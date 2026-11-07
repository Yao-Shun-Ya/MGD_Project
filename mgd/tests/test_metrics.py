"""指标：空标签不得用 0.0 冒充，聚合必须忽略 NaN 并给出样本数。"""
from __future__ import annotations

import numpy as np
import pytest

from mgd.core.imaging.metrics import (
    MetricResult,
    aggregate_metrics,
    clinical_metrics,
    mask_metrics,
)


def _rect(height: int, width: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def test_perfect_prediction_metrics():
    gt = _rect(64, 128, 10, 30, 20, 60)
    result = mask_metrics(gt, gt)
    assert result.dice == pytest.approx(1.0)
    assert result.iou == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.gt_area == gt.sum()
    assert result.pred_area == gt.sum()
    assert not result.errors


def test_disjoint_prediction_metrics():
    gt = _rect(64, 128, 10, 30, 20, 60)
    pred = _rect(64, 128, 40, 50, 80, 100)
    result = mask_metrics(pred, gt)
    assert result.dice == pytest.approx(0.0)
    assert result.iou == pytest.approx(0.0)
    assert result.precision == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)


def test_empty_gt_returns_zero_dice_but_surface_is_nan():
    pred = _rect(64, 128, 10, 30, 20, 60)
    gt = np.zeros((64, 128), dtype=np.uint8)
    result = mask_metrics(pred, gt)
    assert result.dice == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)
    assert np.isnan(result.hd95)
    assert np.isnan(result.assd)
    assert any("标签为空" in msg for msg in result.errors)


def test_both_empty_returns_nan_with_reason():
    result = mask_metrics(np.zeros((64, 128)), np.zeros((64, 128)))
    assert np.isnan(result.dice)
    assert np.isnan(result.iou)
    assert any("均为空" in msg for msg in result.errors)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="形状不一致"):
        mask_metrics(np.zeros((64, 128)), np.zeros((32, 64)))


def test_aggregate_ignores_nan_and_reports_counts():
    a = MetricResult(dice=0.8, iou=0.7)
    b = MetricResult(dice=float("nan"), iou=0.3)
    summary = aggregate_metrics([a, b])
    assert summary["samples"] == 2
    assert summary["dice"] == pytest.approx(0.8)
    assert summary["dice_n"] == 1
    assert summary["iou"] == pytest.approx(0.5)
    assert summary["iou_n"] == 2
    assert np.isnan(summary["hd95"])
    assert summary["hd95_n"] == 0


def test_clinical_metrics_reports_uncalibrated_units():
    mask = _rect(64, 128, 10, 30, 20, 60)
    mask[40:44, 100:110] = 1
    rows = clinical_metrics(mask)
    labels = {row[0]: row[1] for row in rows}
    assert labels["检测腺体数量"] == "2"
    assert labels["量化口径"].startswith("像素未标定")
