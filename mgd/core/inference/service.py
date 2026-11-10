"""进程内推理服务：权重缓存 + 线程锁 + inference_mode + 显式 release。

修复点：旧实现每次点击都从磁盘重载 ~100MB 权重并搬上 GPU，且无并发保护。

治理接线（v2.5.0）：
- 每次成功推理写 ``inference`` 审计事件、失败写 ``inference_failed``（均经
  ``governance.hooks.audit_event``，非阻塞；payload 只含 arch/阈值/sha 前缀/耗时等
  治理元数据，不含 PHI 或图像；case_id 由日志上下文自动带入）；
- 成功路径注入 conformal 诊断（``extra["conformal"]`` + metrics 行），校准读取
  失败降级为无校准，不阻断推理；逻辑抽在模块级 :func:`attach_diagnostic_conformal`，
  可独立单测。
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from ..config import Settings, canonical_weight_candidates, get_settings, model_best_path
from ..governance.hooks import audit_event
from ..imaging.clinical import calculate_clinical_metrics
from ..imaging.postprocess import (
    apply_roi,
    estimate_eyelid_roi_mask,
    eyelid_roi_from_probability,
    postprocess_binary_mask,
    smooth_probability_map,
)
from ..imaging.preprocess import Preprocessed, preprocess_array
from ..imaging.visualize import (
    TrueGradCAM,
    binary_overlay,
    confidence_score,
    generate_contour_visualization,
    generate_heatmap,
    generate_instance_visualization,
    generate_skeleton_visualization,
    gradcam_overlay,
    uncertainty_map,
    uncertainty_overlay,
)
from ..models.loader import load_model, resolve_weight_path
from ..models.manifest import validate_weight
from ..models.registry import normalize_model_name
from ..runtime.device import describe_device, empty_cache, resolve_device
from ..runtime.errors import InferenceError, ManifestError
from ..runtime.logging import bind_context, clear_context, get_logger

logger = get_logger("inference")

TTA_OFF = "关闭"
TTA_2X = "2倍增强 (水平翻转)"
TTA_4X = "4倍增强 (全面翻转)"
TTA_8X = "8倍增强 (多尺度+翻转)"

#: 8 倍 TTA 的第二尺度。1.0 为原尺度，1.125 为轻微放大以补足细腺体上下文。
_TTA_SCALES: tuple[float, ...] = (1.0, 1.125)


def normalize_tta(value: str | None) -> str:
    """把历史/外部 TTA 字符串归一到 off / 2x / 4x / 8x。

    旧配置里出现过 “4x 翻转” 等写法；这里集中兼容，避免再次出现
    “写了 8 倍却只跑 2 倍”的静默降级。
    """
    text = str(value or "").strip()
    if not text or text == TTA_OFF:
        return "off"
    lowered = text.lower()
    if text == TTA_8X or "8倍" in text or lowered.startswith("8x"):
        return "8x"
    if text == TTA_2X or "2倍" in text or lowered.startswith("2x"):
        return "2x"
    if text == TTA_4X or "4倍" in text or lowered.startswith("4x"):
        return "4x"
    logger.warning(
        "未知 TTA 选项 %r，已回退为关闭；可选：%s",
        value, ", ".join((TTA_OFF, TTA_2X, TTA_4X, TTA_8X)),
    )
    return "off"


def _case_id_from_image(image: np.ndarray | str | Path | None) -> str:
    """尽力为一次推理生成稳定 case_id（sha256 前 12 位），无法哈希时记 unknown。

    ndarray → tobytes sha256；路径 → 路径字符串的 sha256（同一路径重复推理 id 一致，便于 grep）。
    """
    try:
        if image is None:
            return "unknown"
        if isinstance(image, (str, Path)):
            payload = str(image).encode("utf-8")
        else:
            payload = np.asarray(image).tobytes()
        return hashlib.sha256(payload).hexdigest()[:12]
    except Exception:
        return "unknown"


def apply_roi_with_fallback(
    mask: np.ndarray,
    prob: np.ndarray,
    roi: np.ndarray,
    min_keep: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """应用 ROI；若 ROI 将预测几乎清零则回退全图并显式标记。

    Otsu 启发式 ROI 在部分红外影像上会与模型预测冲突，直接清零会把
    "有腺体" 误报成 "重度萎缩"。回退不隐藏事实：返回 True 由调用方写入警告。
    """
    before = float(np.asarray(mask).sum())
    candidate_mask, candidate_prob = apply_roi(mask, prob, roi)
    after = float(candidate_mask.sum())
    if before > 0 and after < float(min_keep) * before:
        return mask, prob, True
    return candidate_mask, candidate_prob, False


@dataclass(frozen=True)
class InferenceOptions:
    tile_infer: bool = True
    tile_size: int = 384
    tile_overlap: float = 0.25
    threshold: float = 0.7
    min_area: int = 120
    smooth_on: bool = True
    smooth_sigma: float = 1.2
    roi_only: bool = True
    tta: str = TTA_OFF
    auto_escalate_threshold: bool = False
    gradcam: bool = False
    review_ratio: float = 0.10


@dataclass
class InferenceResult:
    image_rgb: np.ndarray
    prob_map: np.ndarray
    binary_mask: np.ndarray
    overlay: np.ndarray
    heatmap: np.ndarray
    contour: np.ndarray
    skeleton: np.ndarray
    instances: np.ndarray
    metrics: list[list[str]]
    requested_threshold: float
    applied_threshold: float
    threshold_escalated: bool
    warning: str | None = None
    extra: dict = field(default_factory=dict)


def attach_diagnostic_conformal(result: InferenceResult, calibration: dict | None) -> None:
    """把 conformal 校准注入推理结果（就地修改，供单测直接驱动）。

    - ``calibration`` 为 None / 空 → 完全不注入（extra 无 conformal 键、metrics 不加行）；
    - 注入 ``result.extra["conformal"] = {dice_lower_bound, coverage, calibration_id}``，
      其中各字段来自 ``calibration["dice_coverage"]``（bound / coverage）与
      ``calibration["calibration_id"]``；结构缺失一律置 None，不猜测；
    - bound 为有限数时 metrics 追加一行
      ``["Dice 覆盖下界(1-α)", "≥0.xx (校准 id xxxx)"]``；
      口径被 skipped（bound=None）时只写 extra 不加行——绝不让 ``≥nan`` 落进指标表。
    """
    if not calibration:
        return
    dice_block = calibration.get("dice_coverage") or {}
    bound = dice_block.get("bound")
    coverage = dice_block.get("coverage")
    calibration_id = calibration.get("calibration_id")
    payload = {
        "dice_lower_bound": float(bound) if isinstance(bound, (int, float)) else None,
        "coverage": float(coverage) if isinstance(coverage, (int, float)) else None,
        "calibration_id": str(calibration_id) if calibration_id else None,
    }
    result.extra["conformal"] = payload
    bound_value = payload["dice_lower_bound"]
    if bound_value is not None and math.isfinite(bound_value):
        result.metrics.append(
            ["Dice 覆盖下界(1-α)",
             f"≥{bound_value:.2f} (校准 id {payload['calibration_id']})"]
        )


@dataclass(frozen=True)
class WeightCandidate:
    arch: str
    path: Path
    status: str
    reason: str

    @property
    def label(self) -> str:
        suffix = "" if self.status == "verified" else f" [{self.status}]"
        return f"{self.arch} · {self.path.name}{suffix}"


class ModelService:
    """模型缓存与推理入口。同一进程内复用权重，显式 release 释放显存。"""

    def __init__(
        self,
        settings: Settings | None = None,
        device: str | None = None,
        *,
        allow_unverified: bool = False,
        max_cached: int = 3,
    ):
        self.settings = settings or get_settings()
        self.device = resolve_device(device or self.settings.device)
        self.allow_unverified = bool(allow_unverified)
        self.max_cached = max(1, int(max_cached))
        self._cache: dict[tuple, tuple[torch.nn.Module, Path]] = {}
        self._lock = threading.RLock()
        self._infer_lock = threading.Lock()
        # conformal 校准占位：后续版本在此缓存 load_latest_metric_calibration 结果，
        # 当前每次 _predict 直读磁盘（校准文件极小，读代价可忽略）
        self._calibration_cache: dict | None = None
        # 已加载权重的 sha256 前 12 位缓存（load 时从 manifest 捕获，审计 payload 使用）
        self._sha_prefixes: dict[str, str] = {}
        logger.info("ModelService 设备：%s", describe_device(self.device))

    # ---------- 生命周期 ----------
    def cache_info(self) -> list[dict]:
        with self._lock:
            return [
                {"key": key, "path": str(entry[1])} for key, entry in self._cache.items()
            ]

    def release(self, arch: str | None = None) -> None:
        """显式释放缓存；arch 为空则全部释放。"""
        with self._lock:
            if arch is None:
                self._cache.clear()
            else:
                canonical = normalize_model_name(arch)
                for key in [k for k in self._cache if k[0] == canonical]:
                    self._cache.pop(key, None)
        empty_cache(self.device)

    # ---------- 模型 ----------
    def load(self, arch: str, weights_path: str | Path | None = None) -> torch.nn.Module:
        canonical = normalize_model_name(arch)
        path = resolve_weight_path(canonical, weights_path, self.settings)
        key = (canonical, str(path), path.stat().st_mtime_ns, str(self.device))

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached[0]

            status = validate_weight(path, canonical, settings=self.settings)
            if status.manifest is not None and getattr(status.manifest, "sha256", None):
                self._sha_prefixes[str(path)] = str(status.manifest.sha256)[:12]
            if status.status == "pending" and not self.allow_unverified:
                raise ManifestError(
                    f"权重 {path.name} 处于 pending 状态：{status.reason}",
                    hint="先完成口径验证生成 verified manifest，或显式 allow_unverified=True",
                )
            if status.status == "pending":
                logger.warning("加载 pending 权重 %s：%s", path.name, status.reason)

            model, _, _ = load_model(
                canonical,
                path,
                self.device,
                self.settings,
                allow_unverified=True,
            )
            if len(self._cache) >= self.max_cached:
                oldest = next(iter(self._cache))
                self._cache.pop(oldest, None)
            self._cache[key] = (model, path)
            logger.info("已加载权重 %s（设备 %s，缓存 %d）", path.name, self.device, len(self._cache))
            return model

    # ---------- 推理 ----------
    def _prepare(self, image: np.ndarray | str | Path) -> Preprocessed:
        if image is None:
            raise InferenceError("未提供影像")
        if isinstance(image, (str, Path)):
            return preprocess_array(_load_gray(Path(image)), self.settings)
        return preprocess_array(np.asarray(image), self.settings)

    def probability_dual(
        self,
        image: np.ndarray | str | Path,
        arch: str = "UNet",
        options: InferenceOptions | None = None,
        weights_path: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, Preprocessed, np.ndarray | None]:
        """返回 (gland 概率, eyelid 概率或 None, 预处理结果, TTA 概率堆栈或 None)。"""
        options = options or InferenceOptions()
        pre = self._prepare(image)
        model = self.load(arch, weights_path)
        tensor = torch.from_numpy(pre.model_input)[None, None, :, :].float().to(self.device)

        try:
            with self._infer_lock:
                with torch.inference_mode():
                    probabilities = self._forward_probabilities(model, tensor, options)
        except torch.cuda.OutOfMemoryError as exc:
            empty_cache(self.device)
            raise InferenceError(
                "推理显存不足",
                hint="调小分块高度 / 关闭 TTA / 改用 CPU",
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                empty_cache(self.device)
                raise InferenceError("推理显存不足", hint="调小分块高度 / 关闭 TTA") from exc
            raise InferenceError(f"推理失败：{exc}") from exc

        stack = probabilities.float().cpu().numpy()  # (N,C,H,W)
        mean = stack.mean(axis=0)  # (C,H,W)
        gland = mean[0]
        eyelid = mean[1] if mean.shape[0] > 1 else None
        tta_stack = stack[:, 0] if stack.shape[0] > 1 else None
        return gland, eyelid, pre, tta_stack

    def probability(
        self,
        image: np.ndarray | str | Path,
        arch: str = "UNet",
        options: InferenceOptions | None = None,
        weights_path: str | Path | None = None,
    ) -> tuple[np.ndarray, Preprocessed]:
        """只返回腺体概率图与预处理结果（评估/自定义后处理使用）。"""
        gland, _, pre, _ = self.probability_dual(image, arch, options, weights_path)
        return gland, pre

    def _gradcam(self, model: torch.nn.Module, tensor: torch.Tensor, channel: int = 0) -> np.ndarray:
        """半分辨率 Grad-CAM（8GB 显存下避免全尺寸反向传播 OOM），再上采样回原尺寸。"""
        height, width = tensor.shape[-2:]
        target_h = min(int(height), 384)
        scale = target_h / float(height)
        target_w = max(64, int(round(width * scale)))
        small = torch.nn.functional.interpolate(
            tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        cam = TrueGradCAM(model).generate(small, channel=channel)
        import cv2

        return cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)

    def predict(
        self,
        image: np.ndarray | str | Path,
        arch: str = "UNet",
        options: InferenceOptions | None = None,
        weights_path: str | Path | None = None,
    ) -> InferenceResult:
        """推理入口：绑定 stage/case_id 日志上下文，结束即清理（try/finally，不污染下一请求）。"""
        options = options or InferenceOptions()
        bind_context(stage="inference", case_id=_case_id_from_image(image))
        try:
            return self._predict(image, arch, options, weights_path)
        finally:
            clear_context(["case_id", "stage"])

    def _predict(
        self,
        image: np.ndarray | str | Path,
        arch: str,
        options: InferenceOptions,
        weights_path: str | Path | None,
    ) -> InferenceResult:
        """推理主体（治理接线）：

        - 成功 → ``inference`` 审计（arch / applied_threshold / weights sha 前 12 位 /
          耗时 ms / r_loss / 置信度摘要；case_id 由 predict() 绑定的日志上下文自动进入
          payload），并注入 conformal 诊断（读取失败降级，不阻断）；
        - 失败 → ``inference_failed`` 审计（arch / error type），非阻塞，业务异常原样上抛。
        审计与校准均不改变推理结果本身。
        """
        arch_name = normalize_model_name(arch)
        started = time.perf_counter()
        try:
            result = self._run_predict(image, arch_name, options, weights_path)
        except Exception as exc:
            audit_event(
                self.settings,
                "inference_failed",
                actor="system",
                payload={"arch": arch_name, "error_type": type(exc).__name__},
            )
            raise
        attach_diagnostic_conformal(result, self._load_conformal_calibration(arch_name))
        audit_event(
            self.settings,
            "inference",
            actor="system",
            payload=self._inference_audit_payload(arch_name, result, weights_path, started),
        )
        return result

    def _load_conformal_calibration(self, arch_name: str) -> dict | None:
        """读取该 arch 最新 metric 校准；任何失败降级为 None（诊断信息不阻断推理）。

        conformal.pipeline 延迟到函数内导入：推理热路径不为此增加任何 import 期
        依赖（core 顶层 import 门禁不含 conformal）。
        """
        try:
            from ..conformal.pipeline import load_latest_metric_calibration

            calibration = load_latest_metric_calibration(arch_name, self.settings)
        except Exception as exc:
            logger.warning("conformal 校准读取失败（已降级为无校准）：%s", exc)
            return None
        self._calibration_cache = calibration
        return calibration

    def _inference_audit_payload(
        self,
        arch_name: str,
        result: InferenceResult,
        weights_path: str | Path | None,
        started: float,
    ) -> dict:
        """``inference`` 审计 payload：只放治理元数据，不含 PHI / 图像。

        weights sha 前 12 位从 load() 时捕获的 manifest 缓存取（sha256 已在加载期算过，
        这里只做路径解析 + dict 查表）；拿不到（异常 / 未缓存）→ None。
        """
        weights_sha: str | None = None
        try:
            path = resolve_weight_path(arch_name, weights_path, self.settings)
            weights_sha = self._sha_prefixes.get(str(path))
        except Exception:
            weights_sha = None
        clinical = result.extra.get("clinical")
        confidence = result.extra.get("confidence") or {}
        return {
            "arch": arch_name,
            "applied_threshold": float(result.applied_threshold),
            "weights_sha256": weights_sha,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "r_loss_percent": getattr(getattr(clinical, "rloss", None), "r_loss_percent", None),
            "confidence_score": confidence.get("score"),
        }

    def _run_predict(
        self,
        image: np.ndarray | str | Path,
        arch: str,
        options: InferenceOptions,
        weights_path: str | Path | None,
    ) -> InferenceResult:
        gland_prob, eyelid_prob, pre, tta_stack = self.probability_dual(image, arch, options, weights_path)
        prob_map = gland_prob

        post = postprocess_binary_mask(
            prob_map,
            threshold=options.threshold,
            min_area=options.min_area,
            open_kernel=3,
            auto_escalate=options.auto_escalate_threshold,
        )
        mask, prob = post.mask, prob_map.astype(np.float32)
        if options.smooth_on:
            prob = smooth_probability_map(prob, sigma=options.smooth_sigma)

        roi_applied = False
        roi_fallback = False
        dual_task = eyelid_prob is not None
        roi: np.ndarray | None = None
        if options.roi_only:
            if dual_task:
                roi = eyelid_roi_from_probability(eyelid_prob)
            else:
                roi = estimate_eyelid_roi_mask(pre.resized)
            mask, prob, roi_fallback = apply_roi_with_fallback(mask, prob, roi)
            roi_applied = True
            if roi_fallback:
                # 关键：回退后不能拿全图当眼睑分母——那会把 R_loss 拉到 ~90%，
                # 让任何触发回退的病例都被误判成 Meiboscore 3（重度萎缩）。
                # 传 None 走「缺眼睑 ROI → NaN + warning」的诚实分支。
                roi = None

        gray_u8 = pre.resized
        image_rgb = np.repeat(gray_u8[:, :, None], 3, axis=2)
        overlay = binary_overlay(gray_u8, mask)
        heatmap = generate_heatmap(prob, gray_u8)
        contour = generate_contour_visualization(
            prob, gray_u8, min_area=options.min_area, threshold=post.applied_threshold,
        )
        skeleton = generate_skeleton_visualization(mask, gray_u8)
        instances = generate_instance_visualization(mask, gray_u8, min_area=options.min_area)

        clinical = calculate_clinical_metrics(
            mask, roi, min_area=options.min_area, gray_img=gray_u8,
        )
        uncertainty = uncertainty_map(prob_map, tta_stack)
        confidence = confidence_score(prob_map, review_ratio=options.review_ratio)
        uncertainty_image = uncertainty_overlay(uncertainty, gray_u8)

        gradcam_image = None
        if options.gradcam:
            model = self.load(arch, weights_path)
            tensor = torch.from_numpy(pre.model_input)[None, None, :, :].float().to(self.device)
            try:
                with self._infer_lock:
                    cam = self._gradcam(model, tensor, channel=0)
            except torch.cuda.OutOfMemoryError as exc:
                empty_cache(self.device)
                raise InferenceError(
                    "Grad-CAM 显存不足（全分辨率网络的反向传播需要大量激活）",
                    hint="关闭 Grad-CAM，或更换参数量更小的模型/关闭 TTA 后重试",
                ) from exc
            gradcam_image = gradcam_overlay(cam, gray_u8)

        metrics = list(clinical.summary)
        metrics.append(["阈值(用户设定)", f"{post.requested_threshold:.2f}"])
        metrics.append(["阈值(实际应用)", f"{post.applied_threshold:.2f}"])
        metrics.append(["阈值自动抬升", "是" if post.escalated else "否"])
        metrics.append(["边界平滑", f"开启 (σ={options.smooth_sigma:.1f})" if options.smooth_on else "关闭"])
        metrics.append(["ROI 约束", ("全图（ROI 冲突回退）" if roi_fallback
                                    else ("双任务眼睑分支" if dual_task else "Otsu 估计"))
                        if options.roi_only else "全图"])
        # 显示归一后的实际档位：传未知字符串时实际跑 off，不能原样展示误导
        metrics.append(["TTA", normalize_tta(options.tta)])
        metrics.append(["AI 置信度评分", f"{confidence['score']:.1f} / 100"])
        metrics.append(["模糊区占比 (0.4~0.6)", f"{confidence['ambiguous_ratio']:.2%}"])
        metrics.append(["平均像素熵", f"{confidence['mean_entropy']:.3f}"])

        warnings = clinical.warnings
        if roi_fallback:
            warnings.append(
                "ROI 估计与模型预测冲突（ROI 内保留像素不足 5%），已回退全图；请人工复核并优先采用全图结果"
            )
        if post.escalated and post.escalation_reason:
            warnings.append(post.escalation_reason)
        if confidence["need_review"]:
            warnings.append(
                f"模糊区占比 {confidence['ambiguous_ratio']:.2%} 超过 {options.review_ratio:.0%}，建议人工复核"
            )

        return InferenceResult(
            image_rgb=image_rgb,
            prob_map=prob,
            binary_mask=mask,
            overlay=overlay,
            heatmap=heatmap,
            contour=contour,
            skeleton=skeleton,
            instances=instances,
            metrics=metrics,
            requested_threshold=post.requested_threshold,
            applied_threshold=post.applied_threshold,
            threshold_escalated=post.escalated,
            warning="；".join(warnings) if warnings else None,
            extra={
                "device": describe_device(self.device),
                "roi_applied": roi_applied,
                "arch": normalize_model_name(arch),
                "dual_task": dual_task,
                "eyelid_prob": eyelid_prob,
                "roi_mask": roi,
                "uncertainty": uncertainty,
                "uncertainty_image": uncertainty_image,
                "confidence": confidence,
                "gradcam": gradcam_image,
                "instance_df": clinical.instance_df,
                "clinical": clinical,
                "tta_stack": tta_stack,
            },
        )

    def _forward_probabilities(
        self,
        model: torch.nn.Module,
        tensor: torch.Tensor,
        options: InferenceOptions,
    ) -> torch.Tensor:
        """返回 (N,C,H,W) 的 sigmoid 概率堆栈（N=1 或 TTA 次数）。"""
        probabilities: list[torch.Tensor] = []

        def run(batch: torch.Tensor) -> torch.Tensor:
            def logits(x: torch.Tensor) -> torch.Tensor:
                from ..models.registry import unwrap_outputs

                return unwrap_outputs(model(x))[0]

            if not options.tile_infer:
                return logits(batch)
            roi_h = int(max(64, min(int(options.tile_size), self.settings.target_height)))
            roi_w = int(max(128, min(roi_h * 2, self.settings.target_width)))
            overlap = float(max(0.05, min(float(options.tile_overlap), 0.95)))
            return sliding_window_inference(
                inputs=batch,
                roi_size=(roi_h, roi_w),
                sw_batch_size=1,
                predictor=logits,
                overlap=overlap,
            )

        mode = normalize_tta(options.tta)
        flips: tuple[tuple[int, ...], ...] = ((), (3,), (2,), (2, 3))
        if mode == "off":
            views = [(1.0, ())]
        elif mode == "2x":
            views = [(1.0, ()), (1.0, (3,))]
        elif mode == "4x":
            views = [(1.0, dims) for dims in flips]
        else:  # 8x：两个尺度 × 四个翻转
            views = [(scale, dims) for scale in _TTA_SCALES for dims in flips]

        target_size = tensor.shape[-2:]
        for scale, dims in views:
            base = tensor
            if scale != 1.0:
                base = F.interpolate(tensor, scale_factor=scale, mode="bilinear",
                                     align_corners=False, recompute_scale_factor=False)
            if dims:
                base = torch.flip(base, dims=list(dims))
            probability = torch.sigmoid(run(base))
            if dims:
                probability = torch.flip(probability, dims=list(dims))
            if scale != 1.0:
                probability = F.interpolate(probability, size=target_size, mode="bilinear",
                                            align_corners=False)
            probabilities.append(probability)

        return torch.stack([item[0] for item in probabilities], dim=0)


def _load_gray(path: Path) -> np.ndarray:
    from ..imaging.preprocess import load_gray

    return load_gray(path)


def discover_weights(settings: Settings | None = None) -> list[WeightCandidate]:
    """列出 UI 可选的权重及其 manifest 状态。"""
    settings = settings or get_settings()
    candidates: list[WeightCandidate] = []
    from ..models.registry import MODEL_NAMES

    for arch in MODEL_NAMES:
        best = model_best_path(arch, settings)
        if best.is_file():
            status = validate_weight(best, arch, settings=settings)
            candidates.append(WeightCandidate(arch, best, status.status, status.reason))
        if arch == "UNet":
            for path in canonical_weight_candidates(settings):
                if path.is_file():
                    status = validate_weight(path, arch, settings=settings)
                    candidates.append(WeightCandidate(arch, path, status.status, status.reason))
                    break
    return candidates


def select_default_arch(settings: Settings | None = None) -> tuple[str, Path | None]:
    """优先已验证权重，其次已训练 best，最后 canonical（可能 pending）。"""
    candidates = discover_weights(settings)
    verified = [c for c in candidates if c.status == "verified"]
    if verified:
        return verified[0].arch, verified[0].path
    if candidates:
        return candidates[0].arch, candidates[0].path
    return "MGD-Former", None
