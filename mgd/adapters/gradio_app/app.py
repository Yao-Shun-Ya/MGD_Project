"""Gradio 适配层（Windows WebUI 交付端）。

只做「控件 ↔ 核心 API」映射，不含算法细节。视觉按医学软件标准：
深色金属底、低饱和交互色、圆形径向光、克制动效；六个分区：
诊断 / 训练 / 评估 / 实验管理 / 数据集 / 系统状态。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from ... import __version__
from ...core.config import Settings, get_settings, list_trained_models
from ...core.data.pairing import scan_dataset
from ...core.data.splits import SplitSpec, get_splits
from ...core.data.versioning import (
    backup_dataset,
    integrity_report,
    integrity_rows,
    record_version,
)
from ...core.evaluation import (
    EvalReport,
    evaluate_pairs,
    export_csv,
    export_latex,
    summary_rows,
)
from ...core.inference import (
    InferenceOptions,
    ModelService,
    TTA_2X,
    TTA_4X,
    TTA_OFF,
    discover_weights,
    load_presets,
    save_preset,
)
from ...core.models.manifest import validate_weight
from ...core.models.registry import MODEL_NAMES
from ...core.reporting.report import build_report
from ...core.runtime.device import describe_device, resolve_device
from ...core.runtime.doctor import run_doctor
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging, get_logger
from ...core.training.config_file import build_train_config
from ...core.training.engine import TrainConfig, train_iter
from ...core.training.experiments import (
    compare_rows,
    compare_runs,
    export_comparison_csv,
    export_comparison_latex,
    list_runs,
    metric_series,
)
from ...core.training.presets import (
    TRAIN_PROFILES,
    apply_tier,
    detect_device_text,
    detect_vram_gb,
    profile_names,
    profile_overrides,
    tier_table,
)
from ...core.training.runner import run_training

logger = get_logger("adapters.gradio")

#: 同一进程内串行化 GPU 工作，避免「训练 + 推理」并发争抢显存。
WORK_LOCK = threading.Lock()

_TTA_CHOICES = [TTA_OFF, TTA_2X, TTA_4X]
_SCOPE_CHOICES = ["Test（独立测试集）", "Val（验证集）", "Train（训练集自评）"]
_SPLIT_CHOICES = ["Holdout 70/15/15", "K 折交叉验证"]
_VRAM_CHOICES = ["auto", "6", "8", "12", "16", "24", "40"]

CSS = """
:root {
  --mgd-bg: #12151a;
  --mgd-panel: #181c22;
  --mgd-panel-2: #1e242c;
  --mgd-line: #2a323c;
  --mgd-text: #d7dde4;
  --mgd-muted: #8d99a6;
  --mgd-accent: #6f8fa8;
  --mgd-accent-soft: #8fa9bd;
  --mgd-ok: #6f9c7f;
  --mgd-warn: #b09a6a;
  --mgd-fail: #b0776f;
}
body, .gradio-container, .app {
  background: radial-gradient(circle at 50% -10%, #1b222b 0%, var(--mgd-bg) 55%) !important;
  color: var(--mgd-text) !important;
  font-family: "Segoe UI", "Inter", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 18px 22px 32px !important; }
.gradio-container h1, h2, h3, h4, p, span, label, div { color: var(--mgd-text); }
#mgd-header {
  display: flex; align-items: center; gap: 16px; padding: 14px 20px; margin-bottom: 12px;
  border: 1px solid var(--mgd-line); border-radius: 12px;
  background: linear-gradient(180deg, #202730 0%, #171c22 100%);
  box-shadow: inset 0 1px 0 #333c47, 0 10px 26px rgba(0,0,0,.32);
}
#mgd-brand { display:flex; align-items:center; gap:12px; }
#mgd-mark {
  width: 40px; height: 40px; border-radius: 50%;
  background: radial-gradient(circle at 50% 42%, rgba(143,169,189,.55) 0%, rgba(111,143,168,.16) 45%, rgba(18,21,26,0) 72%);
  border: 1px solid #38424e; box-shadow: 0 0 22px rgba(111,143,168,.22);
}
#mgd-title { font-size: 19px; font-weight: 650; letter-spacing: .4px; }
#mgd-sub { font-size: 12px; color: var(--mgd-muted); }
#mgd-chips { margin-left: auto; display:flex; gap:8px; flex-wrap: wrap; }
.mgd-chip {
  font-size: 12px; padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--mgd-line); background: #151a20; color: var(--mgd-muted);
  white-space: nowrap;
}
.tabs { border: none !important; }
.tab-nav button, .tabs > .tab-nav > button {
  color: var(--mgd-muted) !important; background: transparent !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  font-weight: 500 !important;
}
.tab-nav button.selected, .tab-nav button[aria-selected="true"] {
  color: var(--mgd-text) !important; border-bottom-color: var(--mgd-accent) !important;
}
.block, .panel, .form, .gr-box, .gr-group {
  background: var(--mgd-panel) !important; border: 1px solid var(--mgd-line) !important;
  border-radius: 10px !important;
}
input, textarea, select, .gr-input, .gr-dropdown {
  background: #12151a !important; color: var(--mgd-text) !important;
  border: 1px solid #303944 !important; border-radius: 8px !important;
}
input:focus, textarea:focus, select:focus, .gr-input:focus-within {
  outline: none !important; border-color: var(--mgd-accent) !important;
  box-shadow: 0 0 0 3px rgba(111,143,168,.18) !important;
}
button {
  border-radius: 8px !important; border: 1px solid #333d49 !important;
  background: linear-gradient(180deg, #232a33 0%, #1a1f26 100%) !important;
  color: var(--mgd-text) !important; font-weight: 520 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
  transition: background .16s ease, border-color .16s ease, transform .16s ease;
}
button:hover { border-color: #46525f !important; background: linear-gradient(180deg, #2a323c, #1d232b) !important; }
button:focus-visible { outline: 2px solid var(--mgd-accent-soft) !important; outline-offset: 2px; }
button:active { transform: translateY(1px); }
button[variant="primary"], .primary {
  background: linear-gradient(180deg, #33505f 0%, #263c48 100%) !important;
  border-color: #3d5c6c !important;
}
button[variant="primary"]:hover { background: linear-gradient(180deg, #3b5b6d, #2a4350) !important; }
table { color: var(--mgd-text) !important; border-color: var(--mgd-line) !important; }
table th { background: #1c222a !important; color: #aeb9c4 !important; font-weight: 600; }
table td { border-color: #262e37 !important; }
table { overflow-x: auto !important; }
.table-wrap, .table-container, .gr-dataframe, .dataframe { overflow-x: auto !important; max-width: 100% !important; }
label span { color: var(--mgd-muted) !important; }
.mgd-section-title { font-size: 13px; letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--mgd-accent-soft); margin: 4px 0 6px; }
.mgd-help { font-size: 12px; color: var(--mgd-muted); line-height: 1.7; }
.mgd-ok { color: var(--mgd-ok); }
.mgd-warn { color: var(--mgd-warn); }
.mgd-fail { color: var(--mgd-fail); }
"""


def _theme(gr):
    """深色金属主题：同时设置 Gradio 主题 token，避免组件内部残留白底。"""
    try:
        base = gr.themes.Base(
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.zinc,
        )
    except Exception:
        return gr.themes.Base()
    dark_tokens = dict(
        body_background_fill="#12151a",
        background_fill_primary="#181c22",
        background_fill_secondary="#1e242c",
        block_background_fill="#181c22",
        block_label_background_fill="#181c22",
        input_background_fill="#12151a",
        body_text_color="#d7dde4",
        body_text_color_subdued="#8d99a6",
        border_color_primary="#2a323c",
        input_border_color="#303944",
        button_secondary_background_fill="#232a33",
        button_secondary_text_color="#d7dde4",
        table_even_background_fill="#181c22",
        table_odd_background_fill="#1e242c",
    )
    try:
        return base.set(**dark_tokens)
    except TypeError:
        return base


def _launch_style_kwargs(gr) -> dict:
    """Gradio 6 起 theme/css 必须传给 launch()，旧版本仍在 Blocks 构造器上。

    以 launch 签名检测，避免依赖版本号字符串比较。
    """
    import inspect

    try:
        params = inspect.signature(gr.Blocks.launch).parameters
    except (TypeError, ValueError):
        return {}
    if "css" in params:
        kwargs = {"css": CSS, "theme": _theme(gr)}
        if "head" in params:
            # Gradio 6 的 css 只作用于应用容器，视口两侧仍可能是白底；用 head 补齐全局底色
            kwargs["head"] = (
                "<style>html,body{background:#12151a !important;margin:0;}"
                "body{color:#e6e9ee;}</style>"
            )
        return kwargs
    return {}


def _empty_metrics() -> list[list[str]]:
    return [["提示", "尚未推理"]]


def _candidate_map(settings: Settings) -> dict[str, tuple[str, Path | None]]:
    options: dict[str, tuple[str, Path | None]] = {}
    for candidate in discover_weights(settings):
        options[candidate.label] = (candidate.arch, candidate.path)
    for arch in MODEL_NAMES:
        if not any(value[0] == arch for value in options.values()):
            options[f"{arch} · 未训练"] = (arch, None)
    return options


def _resolve_selection(label: str, settings: Settings) -> tuple[str, Path | None]:
    return _candidate_map(settings).get(label, ("MGD-Former", None))


def _scope_key(label: str) -> str:
    text = str(label)
    if text.startswith("Val"):
        return "val"
    if text.startswith("Train"):
        return "train_self"
    return "test"


_WEIGHT_STATUS_CACHE: dict[tuple, object] = {}


def _cached_weight_status(path: Path, arch: str, settings: Settings):
    """manifest 校验含 sha256，按 (路径, mtime, 架构) 缓存。"""
    key = (str(path), path.stat().st_mtime_ns, str(arch))
    status = _WEIGHT_STATUS_CACHE.get(key)
    if status is None:
        status = validate_weight(path, arch, settings=settings)
        _WEIGHT_STATUS_CACHE[key] = status
    return status


def _fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def create_app(settings: Settings | None = None):
    """构建 Gradio Blocks。仅构造，不启动。"""
    import gradio as gr

    settings = settings or get_settings()
    configure_logging(log_file=settings.log_file)
    device_text = describe_device(resolve_device(settings.device))
    vram_gb = detect_vram_gb()
    service = ModelService(settings, allow_unverified=True)
    candidates = _candidate_map(settings)
    default_label = next(iter(candidates))
    presets = load_presets(settings)
    last_report: dict[str, EvalReport] = {}
    last_result: dict = {}
    last_compare: dict[str, list] = {}

    # ================= 诊断 =================
    def infer_fn(image, selection, tta, tile_infer, tile_size, overlap, threshold, min_area,
                 smooth_on, sigma, roi_only, auto_escalate, gradcam):
        if image is None:
            return (None, None, None, None, None, None, None, _empty_metrics(),
                    "**请先上传红外影像（或拖入 640×1280 灰度图）。**")
        arch, path = _resolve_selection(selection, settings)
        options = InferenceOptions(
            tile_infer=bool(tile_infer), tile_size=int(tile_size), tile_overlap=float(overlap),
            threshold=float(threshold), min_area=int(min_area), smooth_on=bool(smooth_on),
            smooth_sigma=float(sigma), roi_only=bool(roi_only), tta=str(tta),
            auto_escalate_threshold=bool(auto_escalate), gradcam=bool(gradcam),
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with WORK_LOCK:
                result = service.predict(image, arch=arch, options=options, weights_path=path)
            elapsed = time.perf_counter() - started
            peak = 0.0
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except MGDError as exc:
            return (None, None, None, None, None, None, None, _empty_metrics(), exc.user_message())
        except Exception as exc:
            return (None, None, None, None, None, None, None, _empty_metrics(), f"推理失败：{exc}")

        last_result.clear()
        last_result.update({"result": result, "arch": arch})
        extra = result.extra
        status = [
            f"**设备**：{extra['device']}　**模型**：{extra['arch']}　"
            f"**任务**：{'双任务（腺体+眼睑）' if extra.get('dual_task') else '单任务（旧版权重兼容）'}",
            f"**阈值**：用户 {result.requested_threshold:.2f} / 实际 {result.applied_threshold:.2f}"
            + ("（已自动抬升）" if result.threshold_escalated else "（未改动）"),
            f"**耗时**：{elapsed * 1000:.0f} ms / 例　"
            f"**显存峰值**：{(f'{peak:.3f} GB（本次推理，不含已加载模型）' if peak else 'CPU 模式，不适用')}",
            f"**分块参数**：{'开启' if tile_infer else '关闭'}　"
            f"{int(tile_size)}×{int(tile_size) * 2}　重叠 {float(overlap):.2f}　"
            f"TTA {tta}",
        ]
        if path is None:
            status.append("**权重状态**：<span class='mgd-warn'>未训练（随机初始化）</span> —— 结果只能用于界面验证，不可用于论文或临床判断。")
        else:
            weight_status = _cached_weight_status(path, arch, settings)
            if weight_status.verified:
                status.append(f"**权重状态**：<span class='mgd-ok'>已验证</span>（{Path(path).name}）")
            else:
                status.append(
                    f"**权重状态**：<span class='mgd-warn'>verified=false</span> —— {weight_status.reason}"
                )
        status.append(
            f"**AI 置信度**：{extra['confidence']['score']:.1f}/100　"
            f"模糊区 {extra['confidence']['ambiguous_ratio']:.2%}"
        )
        status.append("**口径**：形态学量化基于像素未标定的比例，仅作参考。")
        if result.warning:
            status.append(f"<span class='mgd-warn'>⚠ {result.warning}</span>")
        return (
            result.overlay, result.heatmap, result.contour, result.skeleton,
            result.instances, extra.get("uncertainty_image"), extra.get("gradcam"),
            result.metrics, "\n\n".join(status),
        )

    def report_fn():
        if not last_result:
            return None, "请先完成一次分析。"
        payload = last_result["result"]
        try:
            bundle = build_report(payload, model_name=last_result["arch"], patient_id="case")
        except Exception as exc:
            return None, f"报告生成失败：{exc}"
        return str(bundle.html_path), (
            f"HTML：{bundle.html_path.name}　CSV：{bundle.csv_path.name}　{bundle.pdf_note}"
        )

    # ================= 训练 =================
    def preset_fill(profile_name):
        """套用中层预设：把预设值填进表单，用户仍可继续改。"""
        overrides = profile_overrides(profile_name)
        train = overrides.get("train", {})
        evals = overrides.get("eval", {})
        data = overrides.get("data", {})
        label = TRAIN_PROFILES.get(profile_name, {}).get("label", profile_name)
        note = (
            f"**已套用 {label}**：{TRAIN_PROFILES.get(profile_name, {}).get('description', '')}\n\n"
            "> 表单值仍可修改；点「开始训练」时以表单为准。"
        )
        return (
            int(train.get("epochs", 20)),
            float(train.get("lr", 1e-4)),
            train.get("scheduler", "Cosine Annealing"),
            str(evals.get("tta", TTA_OFF)),
            "K 折交叉验证" if data.get("split_mode") == "kfold" else "Holdout 70/15/15",
            note,
        )

    def train_fn(selection, profile_name, vram, epochs, batch_size, lr, optimizer, scheduler,
                 loss_name, split_mode, fold, test_ratio, dual_task, eyelid_weight, tile_mode,
                 tile_height, augmentation, amp, val_ratio, pretrained, deep_supervision,
                 grad_accum, channels_last, cldice_weight, ema_decay, resume_from, seed,
                 publish_best, export_report):
        arch, _ = _resolve_selection(selection, settings)
        overrides = {
            "run": {
                "vram_tier": str(vram), "amp": bool(amp), "seed": int(seed),
                "grad_accum_steps": int(grad_accum), "channels_last": bool(channels_last),
            },
            "data": {
                "split_mode": "kfold" if str(split_mode).startswith("K") else "holdout",
                "fold": int(fold), "test_ratio": float(test_ratio), "val_ratio": float(val_ratio),
                "tile_mode": bool(tile_mode), "tile_height": int(tile_height),
                "augmentation": bool(augmentation),
            },
            "model": {
                "name": arch, "dual_task": bool(dual_task), "eyelid_weight": float(eyelid_weight),
                "pretrained": str(pretrained), "deep_supervision": bool(deep_supervision),
                "cldice_weight": float(cldice_weight), "ema_decay": float(ema_decay),
            },
            "train": {
                "epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr),
                "optimizer": optimizer, "scheduler": scheduler, "loss": loss_name,
                "resume_from": str(resume_from).strip() or None,
            },
            "artifacts": {
                "publish_best": bool(publish_best),
                "export_reference_report": bool(export_report),
            },
        }
        try:
            config, meta = build_train_config(overrides=overrides)
            config, tier_info = apply_tier(config, vram, vram_gb)
            config.validate()
        except (MGDError, ValueError) as exc:
            yield f"<span class='mgd-fail'>配置错误：{exc}</span>", [], None, ""
            return

        rows: list[list] = []
        lines = [
            f"**模型**：{arch}　**显存档**：{tier_info['tier']} GB（{tier_info['status']}）",
            f"**分块**：{tier_info['tile']}　**batch**：{tier_info['batch_size']}×累积 "
            f"{tier_info['grad_accum_steps']}　**AMP**：{'开' if tier_info['amp'] else '关'}",
            f"**档位依据**：{tier_info['reason']}",
        ]
        if tier_info["status"] == "估算":
            lines.append("> 该档位为估算值，尚未在本机实测；首轮请关注显存峰值。")

        queue: Queue = Queue()
        box: dict = {}

        def worker():
            try:
                result, context = run_training(
                    config, meta, settings, on_event=lambda event: queue.put(("event", event))
                )
                box["result"] = result
                box["context"] = context
                queue.put(("done", None))
            except Exception as exc:
                box["error"] = exc
                queue.put(("error", None))
            finally:
                queue.put(("end", None))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            try:
                kind, payload = queue.get(timeout=0.2)
            except Empty:
                # 让出控制权，保持界面可交互
                yield "\n\n".join(lines), rows, None, ""
                continue
            if kind == "end":
                break
            if kind == "error":
                lines.append(f"<span class='mgd-fail'>训练失败：{box.get('error')}</span>")
                yield "\n\n".join(lines), rows, None, ""
                break
            if kind != "event":
                continue
            event = payload
            if event["event"] == "start":
                lines.append(
                    f"**设备**：{event['device']}　训练 {event['train_samples']} / 验证 {event['val_samples']}"
                )
                lines.extend(f"- {item}" for item in event["warnings"])
                yield "\n\n".join(lines), rows, None, ""
            elif event["event"] == "epoch":
                rows.append([
                    event["epoch"],
                    _fmt(event["loss"]),
                    _fmt(event["dice"]),
                    _fmt(event["val_dice"]),
                    _fmt(event.get("val_eyelid_dice")),
                    f"{event['lr']:.2e}",
                    _fmt(event["best"]),
                    _fmt(event.get("samples_per_second"), 2),
                ])
                curve = _curve_image(
                    [row[0] for row in rows], [row[3] for row in rows],
                    [row[2] for row in rows],
                )
                yield "\n\n".join(lines), rows, curve, ""
            elif event["event"] == "done":
                result = event["result"]
                lines.append(
                    f"**完成**：{result.epochs_run} 轮，最佳验证指标 {_fmt(result.best_metric)}"
                    f"（epoch {result.best_epoch}）"
                )
                lines.append(f"**吞吐**：{_fmt(result.samples_per_second, 2)} 样本/秒　"
                             f"**显存峰值**：{_fmt(result.peak_vram_gb, 3)} GB（含模型）")
                lines.append(f"**评估口径**：{result.scope_label}")
                lines.extend(f"- {item}" for item in result.warnings)
        context = box.get("context")
        run_dir = str(context.run_dir) if context else ""
        yield "\n\n".join(lines), rows, _curve_image(
            [row[0] for row in rows], [row[3] for row in rows], [row[2] for row in rows]
        ), run_dir

    def _curve_image(epochs, val_dice, train_dice):
        if not epochs:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(5.2, 2.8), dpi=110)
            fig.patch.set_facecolor("#181c22")
            axis.set_facecolor("#181c22")
            axis.plot(epochs, [_safe_float(v) for v in train_dice], color="#8fa9bd", linewidth=1.2,
                      label="train dice")
            axis.plot(epochs, [_safe_float(v) for v in val_dice], color="#6f9c7f", linewidth=1.6,
                      label="val dice")
            axis.tick_params(colors="#8d99a6", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#2a323c")
            axis.grid(color="#232a33", linewidth=.6)
            axis.legend(facecolor="#181c22", edgecolor="#2a323c", labelcolor="#d7dde4", fontsize=8)
            axis.set_xlabel("epoch", color="#8d99a6", fontsize=8)
            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return image
        except Exception:
            return None

    def _safe_float(value) -> float:
        try:
            result = float(value)
            return result if np.isfinite(result) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    # ================= 评估 =================
    def eval_fn(selection, scope_label):
        arch, path = _resolve_selection(selection, settings)
        scope = _scope_key(scope_label)
        try:
            index = scan_dataset(settings)
            split = get_splits(index.pairs, SplitSpec(), settings)
        except MGDError as exc:
            return _empty_metrics(), _empty_metrics(), exc.user_message()
        pairs = {"test": split.test, "val": split.val, "train_self": split.train}[scope]
        if not pairs:
            return _empty_metrics(), _empty_metrics(), f"{scope_label} 为空，无法评估。"
        try:
            with WORK_LOCK:
                report = evaluate_pairs(pairs, arch=arch, weights_path=path, settings=settings,
                                        scope=scope)
        except MGDError as exc:
            return _empty_metrics(), _empty_metrics(), exc.user_message()
        last_report["report"] = report
        per_sample = [
            [item.stem, _fmt(item.metrics.dice), _fmt(item.metrics.iou), _fmt(item.metrics.hd95),
             _fmt(item.metrics.assd), _fmt(item.metrics.precision), _fmt(item.metrics.recall),
             _fmt(item.cldice), _fmt(item.r_loss_pred, 2),
             item.meiboscore_pred if item.meiboscore_pred >= 0 else "N/A",
             " | ".join(item.metrics.errors) or "-"]
            for item in report.per_sample
        ]
        note = [f"**评估口径**：{report.scope_label}", f"**权重**：{Path(report.weights).name}"]
        if report.is_self_evaluated:
            note.append("> <span class='mgd-warn'>训练集自评，不代表泛化性能，不能直接写进论文表格。</span>")
        note.extend(f"- {item}" for item in report.errors[:3])
        return summary_rows(report), per_sample, "\n".join(note)

    def export_fn(kind):
        report = last_report.get("report")
        if report is None:
            return "请先完成一次评估。"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (export_csv if kind == "csv" else export_latex)(
            report, settings.output_dir / f"eval_{stamp}.{kind if kind == 'csv' else 'tex'}"
        )
        return f"已导出：{path}"

    # ================= 实验管理 =================
    def refresh_runs():
        summaries = list_runs(settings)
        labels = [f"{item['run_id']} · {item['arch'] or '?'}" for item in summaries]
        return gr.update(choices=labels, value=[label for label in labels[:3]]), (
            f"共 {len(summaries)} 个 run。目录：{settings.output_dir / 'runs'}"
        )

    def compare_fn(labels):
        import re

        ids = [str(label).split(" · ")[0] for label in (labels or [])]
        summaries = compare_runs(ids, settings)
        last_compare["summaries"] = summaries
        rows = compare_rows(summaries)
        return rows, _compare_curves(summaries), f"对比 {len(summaries)} 个 run"

    def _compare_curves(summaries):
        if not summaries:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(6.0, 3.0), dpi=110)
            fig.patch.set_facecolor("#181c22")
            axis.set_facecolor("#181c22")
            palette = ["#8fa9bd", "#6f9c7f", "#b09a6a", "#b0776f", "#9d8fb5", "#7fa3a8"]
            for index, item in enumerate(summaries[:6]):
                records = metric_series(item["path"])
                if not records:
                    continue
                epochs = [record.get("epoch") for record in records]
                values = [_safe_float(record.get("val_dice")) for record in records]
                axis.plot(epochs, values, linewidth=1.4, color=palette[index % len(palette)],
                          label=item["run_id"][:22])
            axis.tick_params(colors="#8d99a6", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#2a323c")
            axis.grid(color="#232a33", linewidth=.6)
            axis.legend(facecolor="#181c22", edgecolor="#2a323c", labelcolor="#d7dde4", fontsize=7)
            axis.set_xlabel("epoch", color="#8d99a6", fontsize=8)
            axis.set_ylabel("val dice", color="#8d99a6", fontsize=8)
            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return image
        except Exception:
            return None

    def export_compare(kind):
        summaries = last_compare.get("summaries") or []
        if not summaries:
            return "请先对比至少一个 run。"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = settings.output_dir / f"runs_compare_{stamp}.{kind if kind == 'csv' else 'tex'}"
        path = (export_comparison_csv if kind == "csv" else export_comparison_latex)(summaries, target)
        return f"已导出：{path}"

    # ================= 数据集 =================
    def dataset_check():
        try:
            report = integrity_report(settings)
        except Exception as exc:
            return _empty_metrics(), f"校验失败：{exc}"
        detail = [
            f"**目录**：`{settings.images_dir}` / `{settings.gland_masks_dir}` / `{settings.eyelid_masks_dir}`",
            f"**划分**：{report['split']}",
        ]
        if report["missing_gland_masks"]:
            detail.append("**缺腺体掩膜**：" + ", ".join(report["missing_gland_masks"][:10]))
        if report["missing_eyelid_masks"]:
            detail.append("**缺眼睑掩膜**：" + ", ".join(report["missing_eyelid_masks"][:10]))
        if report["duplicate_stems"]:
            detail.append("**重名**：" + ", ".join(report["duplicate_stems"][:10]))
        if report["naming_anomalies"]:
            detail.append("**命名异常**：" + "；".join(report["naming_anomalies"][:8]))
        if report["ok"]:
            detail.append("<span class='mgd-ok'>完整性校验通过。</span>")
        else:
            detail.append("<span class='mgd-fail'>存在缺失/重复，写版本记录会被拒绝。</span>")
        return integrity_rows(report), "\n\n".join(detail)

    def dataset_version(source, device):
        try:
            payload = record_version(settings, source=source or "本地采集", device=device or "未记录")
        except (ValueError, OSError) as exc:
            return f"<span class='mgd-fail'>未写入：{exc}</span>"
        return (
            "**版本已记录**\n\n"
            f"- 配对：{payload['usable_pairs']}\n"
            f"- 哈希：`{payload['hash']['digest'][:20]}…`（{payload['hash']['mode']}）\n"
            f"- 划分签名：`{(payload['split_signature'] or '')[:16]}…`\n"
            f"- 文件：`{settings.data_dir / 'dataset_version.json'}`"
        )

    def dataset_backup():
        try:
            target = backup_dataset(settings)
        except OSError as exc:
            return f"<span class='mgd-fail'>备份失败：{exc}</span>"
        return f"已备份到 `{target}`（只复制，不覆盖原文件）"

    # ================= 系统状态 =================
    def system_report():
        try:
            report = run_doctor(settings)
            return report.as_text()
        except Exception as exc:
            return f"自检失败：{exc}"

    def weights_table():
        rows = [["模型", "权重文件", "状态", "verified", "原因"]]
        for label, (arch, path) in _candidate_map(settings).items():
            if path is None:
                rows.append([arch, "—", "未训练", "否", "没有可用权重，界面用随机初始化验证流程"])
                continue
            status = _cached_weight_status(path, arch, settings)
            rows.append([arch, Path(path).name, status.status,
                         "是" if status.verified else "否", status.reason])
        return rows

    def tiers_view():
        rows = [["档位", "分块", "batch", "梯度累积", "AMP", "channels_last", "深监督", "TTA",
                 "状态", "依据", "显存峰值", "吞吐"]]
        for item in tier_table():
            rows.append([
                item["档位"], item["分块"], item["batch"], item["梯度累积"], item["AMP"],
                item["channels_last"], item["深监督"], item["TTA"], item["状态"], item["依据"],
                item["显存峰值(GB)"], item["吞吐(样本/秒)"],
            ])
        return rows

    # ================= 界面 =================
    style_kwargs = _launch_style_kwargs(gr)
    blocks_kwargs = {} if "css" in style_kwargs else {"css": CSS, "theme": _theme(gr)}
    with gr.Blocks(title=f"腺而易见 MGD 智能辅助诊疗平台 v{__version__}", **blocks_kwargs) as demo:
        gr.HTML(
            f"""
            <div id="mgd-header">
              <div id="mgd-brand"><div id="mgd-mark"></div>
                <div><div id="mgd-title">腺而易见 · MGD 智能辅助诊疗平台</div>
                     <div id="mgd-sub">睑板腺红外影像分析与学术训练工作台</div></div>
              </div>
              <div id="mgd-chips">
                <span class="mgd-chip">v{__version__}</span>
                <span class="mgd-chip">{device_text}</span>
                <span class="mgd-chip">已训练：{', '.join(list_trained_models(settings)) or '无'}</span>
              </div>
            </div>
            """
        )

        with gr.Tabs():
            # -------- 诊断 -------------
            with gr.Tab("诊断"):
                gr.Markdown(
                    "<div class='mgd-help'>上传 640×1280 红外影像后开始分析。"
                    "结果区会显式标注模型状态（含 verified=false 的原因）、耗时、显存峰值与分块参数；"
                    "未验证权重只能用于流程验证。</div>"
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(type="numpy", label="红外影像")
                        model_dropdown = gr.Dropdown(choices=sorted(candidates), value=default_label,
                                                     label="模型 / 权重")
                        tta_radio = gr.Radio(choices=_TTA_CHOICES, value=TTA_OFF, label="TTA")
                        with gr.Accordion("后处理参数（高级）", open=False):
                            tile_infer = gr.Checkbox(value=True, label="分块推理")
                            tile_size = gr.Slider(64, 640, value=384, step=32, label="分块高度")
                            overlap = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="分块重叠")
                            threshold = gr.Slider(0.05, 0.95, value=0.70, step=0.05, label="分割阈值")
                            min_area = gr.Slider(1, 2000, value=120, step=1, label="最小连通域面积")
                            smooth_on = gr.Checkbox(value=True, label="概率图平滑")
                            sigma = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="平滑 σ")
                            roi_only = gr.Checkbox(value=True, label="仅眼睑 ROI")
                            auto_escalate = gr.Checkbox(value=False, label="允许自动抬升阈值（会如实标注）")
                            gradcam_checkbox = gr.Checkbox(value=False, label="True Grad-CAM（额外反传）")
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(choices=sorted(presets),
                                                          value=sorted(presets)[0], label="推理预设")
                            apply_button = gr.Button("应用预设")
                        preset_name = gr.Textbox(label="保存为新预设", placeholder="输入名称")
                        save_button = gr.Button("保存预设")
                        preset_status = gr.Markdown("")
                        infer_button = gr.Button("开始分析", variant="primary")
                    with gr.Column(scale=5):
                        status_output = gr.Markdown("等待上传影像…")
                        overlay_output = gr.Image(label="分割叠加", type="numpy")
                        with gr.Tabs():
                            with gr.Tab("概率热力图"):
                                heatmap_output = gr.Image(label="概率热力图（非 Grad-CAM）", type="numpy")
                            with gr.Tab("True Grad-CAM"):
                                gradcam_output = gr.Image(label="True Grad-CAM", type="numpy")
                            with gr.Tab("不确定性"):
                                uncertainty_output = gr.Image(label="像素熵不确定性", type="numpy")
                            with gr.Tab("轮廓"):
                                contour_output = gr.Image(label="轮廓", type="numpy")
                            with gr.Tab("骨架"):
                                skeleton_output = gr.Image(label="骨架", type="numpy")
                            with gr.Tab("实例"):
                                instance_output = gr.Image(label="腺体实例", type="numpy")
                        metrics_output = gr.Dataframe(headers=["指标", "数值"], value=_empty_metrics(),
                                                      interactive=False, label="量化指标（表格可横向滚动）")
                        report_button = gr.Button("生成 HTML 临床报告")
                        report_file = gr.File(label="报告下载（HTML / CSV）")
                        report_status = gr.Markdown("")

                infer_button.click(
                    infer_fn,
                    inputs=[image_input, model_dropdown, tta_radio, tile_infer, tile_size, overlap,
                            threshold, min_area, smooth_on, sigma, roi_only, auto_escalate,
                            gradcam_checkbox],
                    outputs=[overlay_output, heatmap_output, contour_output, skeleton_output,
                             instance_output, uncertainty_output, gradcam_output, metrics_output,
                             status_output],
                )
                report_button.click(report_fn, outputs=[report_file, report_status])
                apply_button.click(
                    _preset_apply_handler(settings, load_presets),
                    inputs=[preset_dropdown],
                    outputs=[tile_infer, tile_size, overlap, threshold, min_area, smooth_on, sigma,
                             roi_only, tta_radio, gradcam_checkbox, preset_status],
                )
                save_button.click(
                    _preset_save_handler(settings, load_presets, save_preset),
                    inputs=[preset_name, tile_infer, tile_size, overlap, threshold, min_area,
                            smooth_on, sigma, roi_only, tta_radio, gradcam_checkbox],
                    outputs=[preset_dropdown, preset_status],
                )

            # -------- 训练 -------------
            with gr.Tab("训练"):
                gr.Markdown(
                    "<div class='mgd-help'>三层入口共用同一参数 schema：YAML + CLI <code>--set a.b=c</code>、"
                    "命令行预设（学术 / 快速试跑 / 低显存）、本页表单。普通用户无需写 YAML；"
                    "高级参数在下方折叠区可展开修改。启动时会自动探测显存并推荐档位。</div>"
                )
                vram_recommend = "未探测到 CUDA（CPU 环境）" if vram_gb is None else f"{vram_gb:.1f} GB"
                gr.Markdown(
                    f"<div class='mgd-help'>当前设备：<b>{detect_device_text()}</b>　"
                    f"探测显存：<b>{vram_recommend}</b>　推荐档位由 <code>apply_tier(auto)</code> 以安全边界取整。</div>"
                )
                with gr.Row():
                    train_model = gr.Dropdown(choices=sorted(candidates), value=default_label,
                                              label="模型 / 权重（MGD-Former 为默认主干）")
                    profile_dropdown = gr.Dropdown(choices=profile_names(), value="academic",
                                                   label="预设（中层入口）")
                    apply_profile = gr.Button("套用预设")
                    vram_dropdown = gr.Dropdown(choices=_VRAM_CHOICES, value="auto", label="显存档位")
                profile_note = gr.Markdown("")
                with gr.Accordion("基础参数（默认值即可直接跑）", open=True):
                    with gr.Row():
                        epochs_number = gr.Number(value=20, precision=0, label="总轮数")
                        batch_number = gr.Number(value=1, precision=0, label="batch_size")
                        lr_number = gr.Number(value=1e-4, label="学习率")
                        optimizer_dropdown = gr.Dropdown(choices=["AdamW", "Adam", "SGD"], value="AdamW",
                                                         label="优化器")
                        scheduler_dropdown = gr.Dropdown(
                            choices=["ReduceLROnPlateau (默认)", "Cosine Annealing", "Step Decay",
                                     "CosineAnnealingWarmRestarts"],
                            value="Cosine Annealing", label="学习率调度")
                    with gr.Row():
                        split_mode = gr.Dropdown(choices=_SPLIT_CHOICES, value="Holdout 70/15/15",
                                                 label="划分模式")
                        fold_number = gr.Number(value=0, precision=0, label="K 折 fold")
                        val_ratio = gr.Slider(0.0, 0.4, value=0.15, step=0.05, label="验证集比例")
                        test_ratio = gr.Slider(0.0, 0.4, value=0.15, step=0.05, label="测试集比例")
                        dual_task = gr.Checkbox(value=True, label="双任务（腺体+眼睑）")
                with gr.Accordion("高级参数（可选，与 CLI/YAML 完全一致）", open=False):
                    with gr.Row():
                        train_tile = gr.Checkbox(value=True, label="分块训练")
                        train_tile_size = gr.Slider(64, 640, value=384, step=32, label="分块高度")
                        augmentation = gr.Checkbox(value=True, label="数据增强")
                        amp_checkbox = gr.Checkbox(value=True, label="AMP（仅 CUDA 生效）")
                        pretrained_dropdown = gr.Dropdown(choices=["auto", "none", "ade20k", "imagenet"],
                                                          value="auto", label="编码器预训练")
                        deep_supervision = gr.Checkbox(value=True, label="深监督")
                    with gr.Row():
                        grad_accum = gr.Slider(1, 32, value=1, step=1, label="梯度累积步数")
                        channels_last = gr.Checkbox(value=True, label="channels_last")
                        cldice_weight = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="clDice 权重")
                        ema_decay = gr.Slider(0.9, 1.0, value=0.999, step=0.001, label="EMA 衰减")
                        seed_number = gr.Number(value=42, precision=0, label="随机种子")
                    with gr.Row():
                        eyelid_weight = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="眼睑损失权重")
                        loss_dropdown = gr.Dropdown(
                            choices=["MGD 复合损失", "Dice+CE 组合损失", "Dice Loss", "BCE (pos_weight)",
                                     "Focal Loss", "Tversky Loss", "FocalTversky Loss",
                                     "SoftclDice + Dice", "AnatomicalJointLoss (解剖联合)"],
                            value="MGD 复合损失", label="损失函数")
                        resume_text = gr.Textbox(value="", label="断点续训（run 目录或 .ckpt，留空=新 run）")
                        publish_best = gr.Checkbox(value=False, label="发布 best 到模型目录")
                        export_report = gr.Checkbox(value=True, label="生成 reference_report")
                train_button = gr.Button("开始训练", variant="primary")
                train_status = gr.Markdown("未开始。失败时错误会内联显示在本行下方。")
                history_table = gr.Dataframe(
                    headers=["epoch", "loss", "train_dice", "val_dice", "val_eyelid_dice", "lr",
                             "best", "样本/秒"],
                    interactive=False, label="训练指标流水",
                )
                train_curve = gr.Image(label="验证曲线（train / val dice）", type="numpy")
                run_dir_box = gr.Textbox(label="run 目录（配置快照 / metrics.jsonl / checkpoints）",
                                         interactive=False)
                apply_profile.click(
                    preset_fill, inputs=[profile_dropdown],
                    outputs=[epochs_number, lr_number, scheduler_dropdown, profile_note],
                )
                _bind_train_button(
                    train_button, train_fn,
                    inputs=[train_model, profile_dropdown, vram_dropdown, epochs_number, batch_number,
                            lr_number, optimizer_dropdown, scheduler_dropdown, loss_dropdown,
                            split_mode, fold_number, test_ratio, dual_task, eyelid_weight,
                            train_tile, train_tile_size, augmentation, amp_checkbox, val_ratio,
                            pretrained_dropdown, deep_supervision, grad_accum, channels_last,
                            cldice_weight, ema_decay, resume_text, seed_number, publish_best,
                            export_report],
                    outputs=[train_status, history_table, train_curve, run_dir_box],
                )

            # -------- 评估 -------------
            with gr.Tab("评估"):
                gr.Markdown(
                    "<div class='mgd-help'>复用持久化划分（<code>dataset_splits.json</code>，seed=42）。"
                    "默认 Test 集；训练集自评会明确标注，不能直接写进论文。"
                    "导出 CSV / LaTeX 时会保留逐样本指标，便于 bootstrap 与配对检验。</div>"
                )
                with gr.Row():
                    eval_model = gr.Dropdown(choices=sorted(candidates), value=default_label,
                                             label="模型 / 权重")
                    eval_scope = gr.Dropdown(choices=_SCOPE_CHOICES, value=_SCOPE_CHOICES[0],
                                             label="测试集")
                    eval_button = gr.Button("开始评估", variant="primary")
                eval_note = gr.Markdown("")
                eval_summary = gr.Dataframe(headers=["指标", "数值"], value=_empty_metrics(),
                                            interactive=False, label="汇总")
                eval_per_sample = gr.Dataframe(
                    headers=["样本", "Dice", "IoU", "HD95", "ASSD", "Precision", "Recall",
                             "clDice", "R_loss(%)", "Meiboscore", "错误"],
                    interactive=False, label="逐样本（可横向滚动）",
                )
                with gr.Row():
                    export_csv_button = gr.Button("导出 CSV")
                    export_tex_button = gr.Button("导出 LaTeX")
                export_status = gr.Markdown("")
                eval_button.click(eval_fn, inputs=[eval_model, eval_scope],
                                  outputs=[eval_summary, eval_per_sample, eval_note])
                export_csv_button.click(lambda: export_fn("csv"), outputs=export_status)
                export_tex_button.click(lambda: export_fn("latex"), outputs=export_status)

            # -------- 实验管理 -------------
            with gr.Tab("实验管理"):
                gr.Markdown(
                    "<div class='mgd-help'>每个 run 有唯一 ID、配置快照、代码版本、metrics.jsonl 与产物目录。"
                    "勾选多个 run 可横向对比并导出论文用 CSV / LaTeX；缺 <code>split_signature</code> 的 run 不能进论文表。</div>"
                )
                with gr.Row():
                    refresh_button = gr.Button("刷新 run 列表")
                    compare_button = gr.Button("横向对比", variant="primary")
                runs_status = gr.Markdown("")
                run_selector = gr.Dropdown(choices=[], value=[], multiselect=True,
                                           label="选择 run（默认最近 3 个）")
                compare_table = gr.Dataframe(
                    headers=["run_id", "arch", "epochs", "best_metric", "best_epoch", "val_dice",
                             "显存峰值", "样本/秒", "耗时(s)", "split_signature", "已验证",
                             "创建时间"],
                    interactive=False, label="run 对比",
                )
                compare_curve = gr.Image(label="验证 Dice 曲线", type="numpy")
                with gr.Row():
                    compare_csv = gr.Button("导出对比 CSV")
                    compare_tex = gr.Button("导出对比 LaTeX")
                compare_status = gr.Markdown("")
                refresh_button.click(refresh_runs, outputs=[run_selector, runs_status])
                compare_button.click(compare_fn, inputs=[run_selector],
                                     outputs=[compare_table, compare_curve, compare_status])
                compare_csv.click(lambda: export_compare("csv"), outputs=compare_status)
                compare_tex.click(lambda: export_compare("latex"), outputs=compare_status)

            # -------- 数据集 -------------
            with gr.Tab("数据集"):
                gr.Markdown(
                    "<div class='mgd-help'>版本化记录哈希清单、样本计数、划分签名与来源/设备；"
                    "任何写操作（版本记录、备份）前都会先校验完整性。"
                    "本平台绝不覆盖用户影像、掩膜与权重；备份只复制。</div>"
                )
                with gr.Row():
                    dataset_check_button = gr.Button("完整性校验", variant="primary")
                    dataset_version_button = gr.Button("记录数据集版本")
                    dataset_backup_button = gr.Button("备份数据集（只复制）")
                dataset_table = gr.Dataframe(headers=["项", "数量", "状态"], interactive=False,
                                             label="完整性")
                dataset_note = gr.Markdown("尚未校验。")
                with gr.Row():
                    source_text = gr.Textbox(value="本地采集", label="来源")
                    device_text_box = gr.Textbox(value="未记录", label="采集设备")
                dataset_version_note = gr.Markdown("")
                dataset_backup_note = gr.Markdown("")
                dataset_check_button.click(dataset_check, outputs=[dataset_table, dataset_note])
                dataset_version_button.click(dataset_version, inputs=[source_text, device_text_box],
                                             outputs=dataset_version_note)
                dataset_backup_button.click(dataset_backup, outputs=dataset_backup_note)

            # -------- 系统状态 -------------
            with gr.Tab("系统状态"):
                gr.Markdown(
                    "<div class='mgd-help'>换机器第一件事：点「运行环境自检」。"
                    "报告含 Python / torch / 设备 / 依赖 / 数据 / 权重 manifest / 磁盘 / 写权限；"
                    "失败项会给出可执行的处理建议。</div>"
                )
                doctor_button = gr.Button("运行环境自检", variant="primary")
                doctor_output = gr.Textbox(label="mgd doctor", lines=16, interactive=False)
                gr.Markdown("<div class='mgd-section-title'>权重状态</div>")
                weights_button = gr.Button("刷新权重状态")
                weights_table_box = gr.Dataframe(
                    headers=["模型", "权重文件", "状态", "verified", "原因"],
                    interactive=False, label="权重 manifest",
                )
                gr.Markdown("<div class='mgd-section-title'>显存档位表</div>")
                tiers_table = gr.Dataframe(
                    headers=["档位", "分块", "batch", "梯度累积", "AMP", "channels_last", "深监督",
                             "TTA", "状态", "依据", "显存峰值", "吞吐"],
                    interactive=False, label="显存档位（实测/估算均标注依据）",
                )
                doctor_button.click(system_report, outputs=doctor_output)
                weights_button.click(weights_table, outputs=weights_table_box)
                demo.load(weights_table, outputs=weights_table_box)
                demo.load(tiers_view, outputs=tiers_table)

        demo._mgd_launch_style = style_kwargs  # launch() 取用（Gradio 6）
        return demo


# ----------------- 辅助：把推理预设的 handler 与训练按钮绑定拆出 -----------------
def _bind_train_button(button, handler, inputs, outputs):
    button.click(handler, inputs=inputs, outputs=outputs)


def _preset_apply_handler(settings, loader):
    def apply(name):
        values = loader(settings).get(name)
        if not values:
            return (False, 384, 0.25, 0.70, 120, True, 1.2, True, TTA_OFF, False, "预设不存在")
        return (
            bool(values.get("tile_infer", True)), int(values.get("tile_size", 384)),
            float(values.get("tile_overlap", 0.25)), float(values.get("threshold", 0.70)),
            int(values.get("min_area", 120)), bool(values.get("smooth_on", True)),
            float(values.get("smooth_sigma", 1.2)), bool(values.get("roi_only", True)),
            str(values.get("tta", TTA_OFF)), bool(values.get("gradcam", False)),
            f"已应用预设：{name}",
        )

    return apply


def _preset_save_handler(settings, loader, writer):
    import gradio as gr

    def save(name, tile_infer, tile_size, overlap, threshold, min_area, smooth_on, sigma,
             roi_only, tta, gradcam):
        if not str(name).strip():
            return gr.update(), "请输入预设名称"
        try:
            path = writer(str(name), {
                "tile_infer": bool(tile_infer), "tile_size": int(tile_size),
                "tile_overlap": float(overlap), "threshold": float(threshold),
                "min_area": int(min_area), "smooth_on": bool(smooth_on),
                "smooth_sigma": float(sigma), "roi_only": bool(roi_only), "tta": str(tta),
                "gradcam": bool(gradcam),
            }, settings)
        except MGDError as exc:
            return gr.update(), exc.user_message()
        return gr.update(choices=sorted(loader(settings)), value=str(name)), f"已保存：{path}"

    return save


def launch(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    app = create_app(settings)
    try:
        app.queue(default_concurrency_limit=1)
    except (AttributeError, TypeError):
        try:
            app.queue(concurrency_count=1)
        except (AttributeError, TypeError):
            app.queue()
    auth = ((("mgd", settings.auth_token),) if settings.auth_token else None)
    style_kwargs = getattr(app, "_mgd_launch_style", {})
    app.launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False,
        auth=auth,
        show_error=True,
        inbrowser=False,
        **style_kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="腺而易见 WebUI")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--auth", default=None, help="访问口令（会写入 Settings.auth_token）")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    import dataclasses

    if args.host:
        settings = dataclasses.replace(settings, host=args.host)
    if args.port:
        settings = dataclasses.replace(settings, port=int(args.port))
    if args.auth:
        settings = dataclasses.replace(settings, auth_token=args.auth)
    launch(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
