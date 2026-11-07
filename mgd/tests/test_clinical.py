"""临床指标：R_loss 眼睑分母、空掩膜 NaN、实例形态与截断。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from mgd.core.imaging.clinical import (
    INSTANCE_COLUMNS,
    calculate_clinical_metrics,
    cldice_score,
    instance_metrics,
    meiboscore_from_rloss,
    r_loss,
)


def _mask_with_lines() -> tuple[np.ndarray, np.ndarray]:
    h, w = 240, 480
    eyelid = np.zeros((h, w), np.uint8)
    cv2.ellipse(eyelid, (w // 2, h // 2), (180, 80), 0, 0, 360, 1, -1)
    gland = np.zeros((h, w), np.uint8)
    for y in range(80, 161, 20):
        cv2.line(gland, (60, y), (420, y), 1, 7)
    # 一条很短的腺体，用于截断识别
    cv2.line(gland, (100, 30), (100, 60), 1, 5)
    return gland, eyelid


def test_r_loss_uses_eyelid_as_denominator():
    h, w = 100, 200
    eyelid = np.zeros((h, w), np.uint8)
    eyelid[20:80, 50:150] = 1  # 6000 px
    gland = np.zeros_like(eyelid)
    gland[20:80, 50:100] = 1  # 3000 px
    result = r_loss(gland, eyelid)
    assert result.eyelid_area_px == 6000
    assert result.gland_in_roi_px == 3000
    assert result.r_loss_percent == pytest.approx(50.0)


def test_r_loss_nan_when_eyelid_missing_or_empty():
    gland = np.ones((20, 20), np.uint8)
    missing = r_loss(gland, None)
    assert np.isnan(missing.r_loss_percent)
    assert missing.warnings

    empty = r_loss(gland, np.zeros((20, 20), np.uint8))
    assert np.isnan(empty.r_loss_percent)
    assert empty.meiboscore == -1


def test_meiboscore_mapping():
    assert [meiboscore_from_rloss(v) for v in (0, 10, 40, 90)] == [0, 1, 2, 3]
    assert meiboscore_from_rloss(float("nan")) == -1


def test_instance_metrics_and_truncation():
    gland, eyelid = _mask_with_lines()
    result = calculate_clinical_metrics(gland, eyelid)
    assert list(result.instance_df.columns) == INSTANCE_COLUMNS
    assert len(result.instance_df) >= 4
    assert any(result.instance_df["是否截断"] == "是")
    summary = {row[0]: row[1] for row in result.summary}
    assert summary["检测腺体数量"] != "0"
    assert "%" in summary["截断腺体占比"]


def test_empty_masks_do_not_crash_and_use_nan():
    result = calculate_clinical_metrics(np.zeros((64, 128), np.uint8), np.zeros((64, 128), np.uint8))
    assert len(result.instance_df) == 0
    summary = {row[0]: row[1] for row in result.summary}
    assert summary["腺体缺失率 R_loss (%)"] == "N/A"
    assert summary["检测腺体数量"] == "0"
    assert result.warnings


def test_cldice_perfect_and_disjoint():
    binary = np.zeros((80, 80), np.uint8)
    binary[20:60, 20:60] = 1
    assert cldice_score(binary, binary) > 0.99
    empty = np.zeros_like(binary)
    assert cldice_score(empty, binary) < 0.05


def test_instance_metrics_terminates_with_single_pixel_tail():
    """骨架化停在孤立像素时必须有终止保护（历史上会死循环）。"""
    gland = np.zeros((64, 128), np.uint8)
    gland[10:30, 10:30] = 1
    gland[10, 30] = 1
    table = instance_metrics(gland, min_area=1)
    assert len(table) == 1
    assert set(INSTANCE_COLUMNS).issubset(set(table.columns))
