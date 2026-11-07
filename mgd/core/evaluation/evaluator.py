"""评估：必须在独立划分上跑；指标失败时如实记录，不用 0.0 掩盖。

Phase 5/7：除 Dice/IoU/HD95/ASSD 外，新增 clDice、R_loss（眼睑分母）、
Meiboscore 分级与分级准确率。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings, get_settings
from ..data.pairing import Pair
from ..imaging.clinical import cldice_score, meiboscore_from_rloss, r_loss
from ..imaging.metrics import MetricResult, aggregate_metrics, mask_metrics
from ..imaging.preprocess import load_gray, resize_gray
from ..inference.service import InferenceOptions, ModelService
from ..models.loader import resolve_weight_path
from ..runtime.device import describe_device
from ..runtime.errors import MGDError

SCOPE_LABELS = {
    "test": "独立测试集",
    "val": "独立验证集",
    "holdout": "独立验证/测试集",
    "train": "训练集自评（不代表泛化性能）",
    "train_self": "训练集自评（不代表泛化性能）",
}


@dataclass
class EvalSampleResult:
    stem: str
    metrics: MetricResult
    eyelid_dice: float = float("nan")
    cldice: float = float("nan")
    r_loss_pred: float = float("nan")
    r_loss_gt: float = float("nan")
    meiboscore_pred: int = -1
    meiboscore_gt: int = -1
    meiboscore_correct: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    arch: str
    weights: str
    scope: str
    scope_label: str
    samples: int
    summary: dict
    per_sample: list[EvalSampleResult] = field(default_factory=list)
    device: str = "CPU"
    created_at: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_self_evaluated(self) -> bool:
        return self.scope in ("train", "train_self")


def scope_label(scope: str) -> str:
    return SCOPE_LABELS.get(str(scope), SCOPE_LABELS["holdout"])


def aggregate_extended(results: list[EvalSampleResult]) -> dict:
    summary = aggregate_metrics([item.metrics for item in results])

    def mean(values: list[float]) -> float:
        valid = [v for v in values if isinstance(v, float) and np.isfinite(v)]
        return float(np.mean(valid)) if valid else float("nan")

    summary["eyelid_dice"] = mean([item.eyelid_dice for item in results])
    summary["eyelid_dice_n"] = sum(1 for item in results if np.isfinite(item.eyelid_dice))
    summary["cldice"] = mean([item.cldice for item in results])
    summary["cldice_n"] = sum(1 for item in results if np.isfinite(item.cldice))
    summary["r_loss_pred"] = mean([item.r_loss_pred for item in results])
    summary["r_loss_pred_n"] = sum(1 for item in results if np.isfinite(item.r_loss_pred))
    summary["r_loss_gt"] = mean([item.r_loss_gt for item in results])
    summary["r_loss_gt_n"] = sum(1 for item in results if np.isfinite(item.r_loss_gt))
    graded = [item for item in results if item.meiboscore_gt >= 0 and item.meiboscore_pred >= 0]
    summary["meiboscore_accuracy"] = (
        float(np.mean([item.meiboscore_correct for item in graded])) if graded else float("nan")
    )
    summary["meiboscore_n"] = len(graded)
    return summary


def evaluate_pairs(
    pairs: list[Pair],
    *,
    arch: str = "UNet",
    weights_path: str | Path | None = None,
    settings: Settings | None = None,
    service: ModelService | None = None,
    options: InferenceOptions | None = None,
    scope: str = "test",
    progress=None,
    allow_unverified: bool = True,
) -> EvalReport:
    settings = settings or get_settings()
    if not pairs:
        raise MGDError("评估集为空", hint="请先完成 train/val/test 划分")

    options = options or InferenceOptions(tile_infer=True, tile_size=384, tile_overlap=0.25)
    service = service or ModelService(settings, allow_unverified=allow_unverified)
    resolved = resolve_weight_path(arch, weights_path, settings)
    target_size = settings.target_size

    per_sample: list[EvalSampleResult] = []
    errors: list[str] = []
    warnings: list[str] = []
    for position, pair in enumerate(pairs, start=1):
        try:
            gland_prob, eyelid_prob, _, _ = service.probability_dual(pair.image, arch, options, resolved)
            threshold = float(getattr(options, "threshold", 0.5))
            pred = (gland_prob >= threshold).astype(np.uint8)

            gt_gland = (resize_gray(load_gray(pair.gland_mask), target_size, cv2.INTER_NEAREST) > 0).astype(np.uint8)
            gt_eyelid = (resize_gray(load_gray(pair.eyelid_mask), target_size, cv2.INTER_NEAREST) > 0).astype(np.uint8)

            result = mask_metrics(pred, gt_gland)
            item = EvalSampleResult(stem=pair.stem, metrics=result)
            item.errors.extend(result.errors)

            if eyelid_prob is not None:
                pred_eyelid = (eyelid_prob >= 0.5).astype(np.uint8)
                item.eyelid_dice = mask_metrics(pred_eyelid, gt_eyelid).dice
            item.cldice = cldice_score(pred, gt_gland)

            pred_rloss = r_loss(pred, gt_eyelid)
            gt_rloss = r_loss(gt_gland, gt_eyelid)
            item.r_loss_pred = pred_rloss.r_loss_percent
            item.r_loss_gt = gt_rloss.r_loss_percent
            item.meiboscore_pred = meiboscore_from_rloss(item.r_loss_pred)
            item.meiboscore_gt = meiboscore_from_rloss(item.r_loss_gt)
            item.meiboscore_correct = bool(
                item.meiboscore_pred >= 0 and item.meiboscore_pred == item.meiboscore_gt
            )
            warnings.extend(f"[{pair.stem}] {msg}" for msg in (pred_rloss.warnings + gt_rloss.warnings))

            per_sample.append(item)
        except MGDError as exc:
            errors.append(f"[{pair.stem}] {exc.user_message()}")
        if progress is not None:
            progress(position, len(pairs), pair.stem)

    if not per_sample:
        raise MGDError("所有样本评估失败", hint="\n".join(errors[:5]))

    summary = aggregate_extended(per_sample)
    return EvalReport(
        arch=arch,
        weights=str(resolved),
        scope=str(scope),
        scope_label=scope_label(scope),
        samples=len(per_sample),
        summary=summary,
        per_sample=per_sample,
        device=describe_device(service.device),
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        errors=errors,
        warnings=warnings,
    )


def summary_rows(report: EvalReport) -> list[list[str]]:
    """给 UI 的指标表；第一行永远标注评估口径。"""
    rows: list[list[str]] = [["评估口径", report.scope_label]]
    for key, label in (
        ("dice", "Dice"),
        ("iou", "IoU"),
        ("hd95", "HD95"),
        ("assd", "ASSD"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("cldice", "clDice（骨架连续性）"),
        ("eyelid_dice", "Eyelid Dice（双任务）"),
        ("r_loss_pred", "R_loss 预测均值 (%)"),
        ("r_loss_gt", "R_loss GT 均值 (%)"),
        ("meiboscore_accuracy", "Meiboscore 分级准确率"),
    ):
        value = report.summary.get(key, float("nan"))
        count = report.summary.get(f"{key}_n", 0)
        if key == "meiboscore_accuracy":
            count = report.summary.get("meiboscore_n", 0)
        text = "N/A" if isinstance(value, float) and (np.isnan(value)) else f"{value:.4f}"
        rows.append([label, f"{text} (n={count})"])
    rows.append(["样本数", str(report.samples)])
    rows.append(["设备", report.device])
    rows.append(["权重", Path(report.weights).name])
    return rows


def export_csv(report: EvalReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["stem", "dice", "iou", "hd95", "assd", "precision", "recall", "cldice",
             "eyelid_dice", "r_loss_pred", "r_loss_gt", "meiboscore_pred", "meiboscore_gt",
             "meiboscore_correct", "gt_area", "pred_area", "errors"]
        )
        for item in report.per_sample:
            metrics = item.metrics
            writer.writerow(
                [
                    item.stem,
                    metrics.dice, metrics.iou, metrics.hd95, metrics.assd,
                    metrics.precision, metrics.recall, item.cldice,
                    item.eyelid_dice, item.r_loss_pred, item.r_loss_gt,
                    item.meiboscore_pred, item.meiboscore_gt,
                    int(item.meiboscore_correct),
                    metrics.gt_area, metrics.pred_area,
                    " | ".join(item.errors or metrics.errors),
                ]
            )
        writer.writerow([])
        writer.writerow(["scope", report.scope_label])
        writer.writerow(["arch", report.arch])
        writer.writerow(["weights", report.weights])
        writer.writerow(["created_at", report.created_at])
    return target


def export_latex(report: EvalReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def esc(text: str) -> str:
        for source, replacement in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
            text = str(text).replace(source, replacement)
        return text

    def fmt(value: float) -> str:
        return "N/A" if isinstance(value, float) and np.isnan(value) else f"{value:.4f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + esc(report.arch) + " 评估结果（" + esc(report.scope_label) + "）}",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Sample & Dice & IoU & HD95 & ASSD & clDice & R\_loss(\%) & Meiboscore \\",
        r"\hline",
    ]
    for item in report.per_sample:
        metrics = item.metrics
        lines.append(
            " & ".join(
                [
                    esc(item.stem),
                    fmt(metrics.dice),
                    fmt(metrics.iou),
                    fmt(metrics.hd95),
                    fmt(metrics.assd),
                    fmt(item.cldice),
                    fmt(item.r_loss_pred),
                    str(item.meiboscore_pred) if item.meiboscore_pred >= 0 else "N/A",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
