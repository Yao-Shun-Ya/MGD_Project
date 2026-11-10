"""WP-5 split conformal 不确定性 + 风险控制：离线单测。

覆盖：蒙特卡洛覆盖率（向量化）、有限样本分位公式、参数校验、
阈值 / 收缩选择、store 往返。全部合成数据，无网络、无 torch。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mgd.core.config import set_settings
from mgd.core.conformal.calibrator import ConformalCalibrator
from mgd.core.conformal.risk import select_shrink_for_fp, select_threshold_for_fnr
from mgd.core.conformal.store import (
    calibrator_from_store,
    calibrator_to_store,
    load_calibration,
    save_calibration,
)
from mgd.core.runtime.errors import ConfigError


# ---------------------------------------------------------------------------
# calibrator：蒙特卡洛覆盖率（向量化，保证 <3s）
# ---------------------------------------------------------------------------

def test_monte_carlo_coverage_on_target():
    """200 次实验 × n=100 校准 / 50 测试点：平均覆盖率应落在 1-alpha±0.08。"""
    rng = np.random.default_rng(424242)
    n_exp, n_calib, n_test = 200, 100, 50
    alpha, sigma = 0.10, 0.05

    cal_truth = rng.uniform(0.0, 1.0, size=(n_exp, n_calib))
    cal_pred = cal_truth + rng.normal(0.0, sigma, size=(n_exp, n_calib))
    test_truth = rng.uniform(0.0, 1.0, size=(n_exp, n_test))
    test_pred = test_truth + rng.normal(0.0, sigma, size=(n_exp, n_test))

    # 逐实验走真实 ConformalCalibrator（fit/quantile），覆盖评估向量化
    quantiles = np.empty(n_exp, dtype=float)
    for e in range(n_exp):
        cal = ConformalCalibrator(alpha=alpha).fit(
            cal_pred[e].tolist(), cal_truth[e].tolist()
        )
        quantiles[e] = cal.quantile()

    covered = np.abs(test_truth - test_pred) <= quantiles[:, None]
    mean_coverage = float(covered.mean())
    assert 1 - alpha - 0.08 <= mean_coverage <= 1 - alpha + 0.08

    # 抽第 0 个实验用 interval() 逐点复核：与向量化口径完全一致
    cal0 = ConformalCalibrator(alpha=alpha).fit(
        cal_pred[0].tolist(), cal_truth[0].tolist()
    )
    q0 = cal0.quantile()
    hits = 0
    for t, p in zip(test_truth[0].tolist(), test_pred[0].tolist()):
        lo, hi = cal0.interval(p)
        assert lo == pytest.approx(p - q0)
        assert hi == pytest.approx(p + q0)
        hits += int(lo <= t <= hi)
    assert hits / n_test == pytest.approx(float(covered[0].mean()))


# ---------------------------------------------------------------------------
# calibrator：有限样本分位公式与边界
# ---------------------------------------------------------------------------

def test_quantile_matches_finite_sample_formula():
    scores = [0.42, 0.11, 0.73, 0.05, 0.29, 0.66, 0.18, 0.91, 0.33, 0.57,
              0.24, 0.80, 0.09, 0.49, 0.62, 0.15, 0.38, 0.71, 0.03]  # n=19
    alpha = 0.10
    n = len(scores)
    truths = [float(i) for i in range(n)]
    preds = [t + s for t, s in zip(truths, scores)]  # |pred-truth| = s
    cal = ConformalCalibrator(alpha=alpha).fit(preds, truths)
    k = math.ceil((n + 1) * (1 - alpha))  # 手算：ceil(20*0.9)=18 → 第 18 小
    assert cal.quantile() == pytest.approx(sorted(scores)[k - 1])
    assert cal.scores == pytest.approx(scores)


def test_quantile_boundaries():
    # n == 1/alpha 恰好可评估：k = ceil(11*0.9) = 10 → 最大得分
    scores10 = [0.5] * 9 + [1.5]
    cal10 = ConformalCalibrator(alpha=0.10).fit(
        [1.0 + s for s in scores10], [1.0] * 10
    )
    assert cal10.quantile() == pytest.approx(1.5)
    # alpha=0.5 最小样本量 n=2：k = ceil(3*0.5) = 2 → 第 2 小
    cal50 = ConformalCalibrator(alpha=0.5).fit([1.2, 1.6], [1.0, 1.0])
    assert cal50.quantile() == pytest.approx(0.6)


def test_interval_symmetric_and_rejects_nonfinite():
    cal = ConformalCalibrator(alpha=0.10).fit(
        [0.1 * i for i in range(1, 16)], [0.0] * 15
    )
    q = cal.quantile()
    lo, hi = cal.interval(2.0)
    assert lo == pytest.approx(2.0 - q)
    assert hi == pytest.approx(2.0 + q)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            cal.interval(bad)


def test_fit_rejects_bad_inputs():
    cal = ConformalCalibrator(alpha=0.10)
    with pytest.raises(ValueError):
        cal.fit([1.0, 2.0], [0.5])  # 长度不等
    with pytest.raises(ValueError):
        cal.fit([], [])  # 空
    with pytest.raises(ValueError):
        cal.fit([1.0, float("nan"), 2.0], [0.5, 0.5, 0.5])  # 含 NaN
    with pytest.raises(ValueError):
        cal.fit([1.0] * 9, [0.5] * 9)  # n=9 < 1/0.1=10
    with pytest.raises(ValueError):
        cal.fit([1.0] * 9, [0.5, float("inf"), 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        ConformalCalibrator(alpha=0.7).fit([1.0] * 12, [0.5] * 12)  # alpha > 0.5
    with pytest.raises(ValueError):
        ConformalCalibrator(alpha=0.0).fit([1.0] * 12, [0.5] * 12)

    small = ConformalCalibrator(alpha=0.10).fit([1.0] * 12, [0.5] * 12)
    with pytest.raises(ValueError, match="校准样本不足"):
        ConformalCalibrator(alpha=0.10).quantile()  # 未拟合
    with pytest.raises(ValueError, match="校准样本不足"):
        ConformalCalibrator(alpha=0.10).fit([1.0] * 5, [0.5] * 5)
    # 失败的 fit 不得污染旧状态
    assert len(small.scores) == 12


def test_to_dict_from_dict_roundtrip():
    truths = [0.1 * i for i in range(12)]
    preds = [t + 0.07 * ((i % 4) + 1) for i, t in enumerate(truths)]
    cal = ConformalCalibrator(alpha=0.10).fit(preds, truths)
    d = cal.to_dict()
    assert d["alpha"] == pytest.approx(0.10)
    assert len(d["scores"]) == 12

    restored = ConformalCalibrator.from_dict(d)
    assert restored.alpha == pytest.approx(cal.alpha)
    assert restored.scores == cal.scores
    assert restored.quantile() == pytest.approx(cal.quantile())

    with pytest.raises(ValueError):
        ConformalCalibrator.from_dict({"alpha": 0.9, "scores": [0.1] * 12})
    with pytest.raises(ValueError):
        ConformalCalibrator.from_dict({"alpha": 0.1, "scores": [float("nan")] * 12})
    with pytest.raises(ValueError):
        ConformalCalibrator.from_dict({"scores": [0.1] * 12})  # 缺 alpha


# ---------------------------------------------------------------------------
# risk：select_threshold_for_fnr
# ---------------------------------------------------------------------------

def _gt_maps(n=12, size=(16, 16), seed=7):
    """带正像素的合成 GT（强制 [0,0] 为正，避免触发空 GT 跳过逻辑）。"""
    rng = np.random.default_rng(seed)
    maps = []
    for _ in range(n):
        m = (rng.random(size) > 0.6).astype(np.float32)
        m[0, 0] = 1.0
        maps.append(m)
    return maps


def test_select_threshold_fnr_perfect_prediction():
    gts = _gt_maps()
    probs = [g.copy() for g in gts]
    result = select_threshold_for_fnr(probs, gts, epsilon=0.10, alpha=0.10)
    assert result["feasible"] is True
    assert result["threshold"] == pytest.approx(0.05)  # 全部可行 → 网格最小 t
    assert result["achieved_fnr"] <= 0.10 + 1e-12
    assert result["n_images"] == 12
    assert result["skipped_empty_gt"] == 0
    assert result["guarantee"] == "P(FNR<=epsilon)>=1-alpha"


def test_select_threshold_fnr_all_wrong():
    gts = _gt_maps()
    probs = [1.0 - g for g in gts]  # 全错预测
    result = select_threshold_for_fnr(probs, gts, epsilon=0.10, alpha=0.10)
    assert result["feasible"] is False
    assert result["best_q"] == pytest.approx(1.0)
    assert result["reason"]


def test_select_threshold_fnr_insufficient_images():
    gts = _gt_maps(n=5)  # n=5 < 1/0.1=10 → 不可评估，即使预测完美
    probs = [g.copy() for g in gts]
    result = select_threshold_for_fnr(probs, gts, epsilon=0.10, alpha=0.10)
    assert result["feasible"] is False
    assert result["best_q"] is None


def test_select_threshold_fnr_skips_empty_gt():
    gts = _gt_maps()
    gts[3] = np.zeros_like(gts[3])  # 无正像素 → 跳过并计数
    probs = [g.copy() for g in gts]
    result = select_threshold_for_fnr(probs, gts, epsilon=0.10, alpha=0.10)
    assert result["feasible"] is True
    assert result["n_images"] == 11
    assert result["skipped_empty_gt"] == 1


# ---------------------------------------------------------------------------
# risk：select_shrink_for_fp
# ---------------------------------------------------------------------------

def test_select_shrink_fp_perfect_prediction():
    gts = _gt_maps()
    probs = [g.copy() for g in gts]
    result = select_shrink_for_fp(probs, gts, epsilon=0.05, alpha=0.10)
    assert result["feasible"] is True
    assert result["threshold"] == pytest.approx(0.05)  # 网格中最低的可行 t
    assert result["achieved_fp_rate"] <= 0.05 + 1e-12
    assert result["n_images"] == 12
    assert result["skipped_empty_keep"] == 0
    assert result["guarantee"] == "P(FP_rate<=epsilon)>=1-alpha"


def test_select_shrink_skips_empty_keep_images():
    gts = _gt_maps()
    probs = [g.copy() for g in gts]
    probs[0] = np.zeros_like(probs[0])  # 该图 keep 永远为空 → 每个阈值都跳过
    result = select_shrink_for_fp(probs, gts, epsilon=0.05, alpha=0.10)
    assert result["feasible"] is True
    assert result["threshold"] == pytest.approx(0.05)
    assert result["n_images"] == 11
    assert result["skipped_empty_keep"] == 1


def test_risk_input_validation():
    gts = _gt_maps(n=12)
    with pytest.raises(ValueError):
        select_threshold_for_fnr(gts[:2], gts)  # 数量不一致
    with pytest.raises(ValueError):
        select_threshold_for_fnr([], [])  # 空
    with pytest.raises(ValueError):
        select_shrink_for_fp([], [])
    bad = [g.copy() for g in gts]
    bad[1] = bad[1][:-1, :]  # 形状不一致
    with pytest.raises(ValueError):
        select_threshold_for_fnr([g.copy() for g in gts], bad)
    with pytest.raises(ValueError):
        select_shrink_for_fp([g.copy() for g in gts], bad)
    bad_prob = [g.copy() for g in gts]
    bad_prob[0] = bad_prob[0] * 2.0  # 概率 > 1
    with pytest.raises(ValueError):
        select_threshold_for_fnr(bad_prob, gts)
    with pytest.raises(ValueError):
        select_shrink_for_fp(bad_prob, gts)


def test_gt_binarization_tolerates_0_255():
    gts = _gt_maps()
    gts255 = [(g * 255).astype(np.uint8) for g in gts]  # {0,255} 存储
    probs = [g.astype(np.float32) for g in gts]  # {0,1} 概率
    fnr = select_threshold_for_fnr(probs, gts255, epsilon=0.10, alpha=0.10)
    assert fnr["feasible"] is True
    assert fnr["achieved_fnr"] <= 1e-12
    fp = select_shrink_for_fp(probs, gts255, epsilon=0.05, alpha=0.10)
    assert fp["feasible"] is True
    assert fp["threshold"] == pytest.approx(0.05)
    assert fp["achieved_fp_rate"] <= 1e-12


# ---------------------------------------------------------------------------
# store：持久化往返 + 版本/缺失处理
# ---------------------------------------------------------------------------

def _fitted_calibrator() -> ConformalCalibrator:
    truths = [float(i) * 0.1 for i in range(15)]
    scores = [0.05 * (i % 7) + 0.02 for i in range(15)]
    return ConformalCalibrator(alpha=0.10).fit(
        [t + s for t, s in zip(truths, scores)], truths
    )


def test_store_roundtrip(tmp_path):
    cal = _fitted_calibrator()
    path = save_calibration("demo_cal", calibrator_to_store(cal), directory=tmp_path)
    assert path == tmp_path / "demo_cal.json"

    payload = load_calibration("demo_cal", directory=tmp_path)
    assert payload["version"] == 1
    assert payload["name"] == "demo_cal"
    assert "created_at" in payload

    restored = calibrator_from_store(payload)
    assert restored.alpha == pytest.approx(cal.alpha)
    assert restored.scores == cal.scores
    assert restored.quantile() == pytest.approx(cal.quantile())


def test_load_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_calibration("nope", directory=tmp_path)


def test_load_rejects_newer_version(tmp_path):
    (tmp_path / "future.json").write_text(
        json.dumps({"version": 2, "name": "future"}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="版本过新"):
        load_calibration("future", directory=tmp_path)
    with pytest.raises(ConfigError):
        calibrator_from_store(
            {"version": 2, "kind": "conformal_calibrator",
             "payload": {"alpha": 0.1, "scores": [0.1] * 12}}
        )


def test_save_and_load_use_settings_default_dir(settings):
    set_settings(settings)
    path = save_calibration("default_dir_probe", {"alpha": 0.1, "scores": [0.3] * 12})
    assert path == settings.calibration_dir / "default_dir_probe.json"
    assert path.is_file()
    payload = load_calibration("default_dir_probe")
    assert payload["version"] == 1
    assert payload["scores"] == [0.3] * 12


def test_store_rejects_nan_scores_at_save_time(tmp_path):
    # scores 被外部直接改出 NaN：calibrator_to_store 在保存端即拦截（提示先清洗），
    # 而不是等读取端 from_dict 才发现
    bad_cal = ConformalCalibrator(alpha=0.10, scores=[0.1, float("nan"), 0.3] * 5)
    with pytest.raises(ValueError, match="清洗"):
        calibrator_to_store(bad_cal)

    # save_calibration 对含 NaN/Inf 的任意 payload 也在写盘前拦截
    # （json.dumps(..., allow_nan=False)），且不留半截文件
    with pytest.raises(ValueError):
        save_calibration(
            "nan_payload", {"alpha": 0.1, "scores": [float("inf")]}, directory=tmp_path
        )
    assert not (tmp_path / "nan_payload.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
