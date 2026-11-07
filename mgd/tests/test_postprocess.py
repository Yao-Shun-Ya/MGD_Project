"""后处理：阈值不得静默改写，抬升必须显式且可追溯。"""
from __future__ import annotations

import numpy as np

from mgd.core.imaging.postprocess import (
    apply_roi,
    estimate_eyelid_roi_mask,
    postprocess_binary_mask,
    smooth_probability_map,
)
from mgd.core.imaging.visualize import generate_skeleton_visualization


def _prob_map() -> np.ndarray:
    """64x128：核心 8 行 @0.95（12.5%），光晕 16 行 @0.72，合计 37.5%。"""
    prob = np.full((64, 128), 0.30, dtype=np.float32)
    prob[0:8, :] = 0.95
    prob[8:24, :] = 0.72
    return prob


def test_default_does_not_escalate_threshold():
    prob = _prob_map()
    result = postprocess_binary_mask(prob, threshold=0.70, min_area=120)
    assert result.requested_threshold == 0.70
    assert result.applied_threshold == 0.70
    assert result.escalated is False
    assert result.escalation_reason is None
    assert result.mask.shape == prob.shape
    assert result.mask.dtype == np.float32
    # 0.70 阈值会同时保留核心与光晕
    assert result.mask.sum() > 8 * 128


def test_explicit_escalation_changes_applied_threshold_and_reports_reason():
    result = postprocess_binary_mask(
        _prob_map(), threshold=0.70, min_area=120, auto_escalate=True, max_coverage=0.25
    )
    assert result.escalated is True
    assert result.applied_threshold == 0.75
    assert result.requested_threshold == 0.70
    assert "0.75" in result.escalation_reason
    assert "覆盖率" in result.escalation_reason


def test_min_area_filters_small_components():
    prob = np.zeros((64, 128), dtype=np.float32)
    prob[10:20, 10:30] = 0.9
    prob[50:53, 100:103] = 0.9  # 9 px，低于 min_area
    result = postprocess_binary_mask(prob, threshold=0.5, min_area=120)
    assert result.mask[50:53, 100:103].sum() == 0
    assert result.mask[10:20, 10:30].sum() > 0


def test_smooth_probability_map_keeps_range_and_constant():
    prob = _prob_map()
    smoothed = smooth_probability_map(prob, sigma=1.2)
    assert smoothed.dtype == np.float32
    assert smoothed.min() >= 0.0 and smoothed.max() <= 1.0

    flat = np.full((16, 16), 0.5, dtype=np.float32)
    blurred_flat = smooth_probability_map(flat, sigma=1.0)
    assert np.allclose(blurred_flat, 0.5, atol=1e-6)


def test_sigma_zero_is_noop():
    prob = _prob_map()
    assert np.array_equal(smooth_probability_map(prob, sigma=0.0), prob)


def test_roi_mask_shape_and_values():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(64, 128), dtype=np.uint8)
    roi = estimate_eyelid_roi_mask(gray)
    assert roi.shape == gray.shape
    assert roi.dtype == np.float32
    assert set(np.unique(roi)).issubset({0.0, 1.0})


def test_apply_roi_zeroes_outside_region():
    mask = np.ones((64, 128), dtype=np.float32)
    prob = np.ones((64, 128), dtype=np.float32)
    roi = np.zeros((64, 128), dtype=np.float32)
    roi[10:30, 20:60] = 1.0

    masked, masked_prob = apply_roi(mask, prob, roi)
    assert masked[0, 0] == 0.0
    assert masked[15, 30] == 1.0
    assert masked_prob[0, 0] == 0.0


def test_skeletonize_terminates_on_isolated_pixel():
    """孤立单像素无法被腐蚀（默认边界视为 +inf），必须有终止保护。"""
    gray = np.zeros((32, 64), dtype=np.uint8)
    mask = np.zeros((32, 64), dtype=np.float32)
    mask[5, 7] = 1.0
    mask[10:20, 10:30] = 1.0
    out = generate_skeleton_visualization(mask, gray)
    assert out.shape == (32, 64, 3)
    assert out.dtype == np.uint8
