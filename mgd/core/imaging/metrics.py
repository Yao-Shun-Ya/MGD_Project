"""评价指标。空标签、计算失败等情况返回 NaN + 明确原因，不再用 0.0 冒充。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MetricResult:
    dice: float = float("nan")
    iou: float = float("nan")
    hd95: float = float("nan")
    assd: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    gt_area: int = 0
    pred_area: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "hd95": self.hd95,
            "assd": self.assd,
            "precision": self.precision,
            "recall": self.recall,
            "gt_area": self.gt_area,
            "pred_area": self.pred_area,
        }


def _as_tensor(mask: np.ndarray) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, None]
    elif arr.ndim == 3:
        arr = arr[None]
    return torch.from_numpy(arr)


def _surface_metric(name: str, pred: torch.Tensor, gt: torch.Tensor) -> float:
    if name == "hd95":
        from monai.metrics import HausdorffDistanceMetric

        metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    else:
        from monai.metrics import SurfaceDistanceMetric

        metric = SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="mean")
    metric(pred, gt)
    try:
        value = float(metric.aggregate().item())
    finally:
        metric.reset()
    return value


def mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)) -> MetricResult:
    """单张影像的二值掩膜指标。"""
    pred = (np.asarray(pred_mask) > 0).astype(np.uint8)
    gt = (np.asarray(gt_mask) > 0).astype(np.uint8)
    if pred.shape != gt.shape:
        raise ValueError(f"掩膜形状不一致：pred={pred.shape}, gt={gt.shape}")

    result = MetricResult(gt_area=int(gt.sum()), pred_area=int(pred.sum()))
    inter = int(np.logical_and(pred, gt).sum())
    pred_sum, gt_sum = int(pred.sum()), int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        result.errors.append("预测与标签均为空，Dice 无定义")
        return result

    union = pred_sum + gt_sum
    result.dice = float(2 * inter / union) if union else float("nan")
    result.iou = float(inter / (union - inter)) if (union - inter) else float("nan")
    result.precision = float(inter / pred_sum) if pred_sum else 0.0
    result.recall = float(inter / gt_sum) if gt_sum else 0.0

    if gt_sum == 0:
        result.errors.append("标签为空，表面距离指标无定义")
        return result
    if pred_sum == 0:
        result.errors.append("预测为空，表面距离指标无定义")
        return result

    try:
        pred_t, gt_t = _as_tensor(pred), _as_tensor(gt)
        result.hd95 = _surface_metric("hd95", pred_t, gt_t)
        result.assd = _surface_metric("assd", pred_t, gt_t)
    except (RuntimeError, ValueError, ZeroDivisionError, IndexError) as exc:
        result.errors.append(f"表面距离计算失败：{type(exc).__name__}: {exc}")

    return result


def aggregate_metrics(results: list[MetricResult]) -> dict:
    """忽略 NaN 求均值，并给出参与统计的样本数。"""
    keys = ("dice", "iou", "hd95", "assd", "precision", "recall")
    summary: dict = {"samples": len(results)}
    for key in keys:
        values = [getattr(item, key) for item in results]
        valid = [v for v in values if isinstance(v, float) and not np.isnan(v)]
        summary[key] = float(np.mean(valid)) if valid else float("nan")
        summary[f"{key}_n"] = len(valid)
    return summary


def clinical_metrics(mask: np.ndarray) -> list[list[str]]:
    """面向报告的基础形态学量化。像素未标定，只给比例，不给毫米。"""
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 0]
    coverage = float(binary.mean())
    rows = [
        ["检测腺体数量", str(len(areas))],
        ["腺体覆盖面积占比", f"{coverage:.2%}"],
        ["前景像素面积", str(int(binary.sum()))],
        ["平均单腺体面积(px)", f"{np.mean(areas):.1f}" if areas else "0"],
        ["量化口径", "像素未标定，仅作比例参考"],
    ]
    return rows
