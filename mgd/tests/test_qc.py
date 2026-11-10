"""WP-4 qc 采集质控单元测试。

全部离线、合成数据、CPU 运行，目标耗时 < 10 秒。覆盖：
1. 干净灰图（128±轻微噪声）→ 无 flag、score ≥ 90；
2. 全黑 / 全白 → underexposed / overexposed（常数图 specular 按 0 处理，
   float 退化输入按直接截断归一化）；
3. 椒盐高光块（254 灰阶，约 1.9% 像素）→ specular（且不误报 overexposed）；
4. 高斯模糊（sigma=15）→ blur（先断言未模糊图不触发 blur，防恒真）；
5. 眼睑 ROI 极小（面积比 ≈0.02）→ poor_eversion；ROI 覆盖全图 → over_flip；
6. 阈值改变结论翻转：边缘棋盘图 laplacian_var ∈ (60, 5000)，
   blur_min=5000 → 触发 blur，默认 60 → 不触发；
7. decide：30 → reject、60 → review、85 → accept、边界值、非法 policy → ValueError；
8. eyelid 缺失 → eversion 相关 metrics 为 None 且 flags 不含 4 个 ROI 类 flag；
9. 附加：离焦（逐列厚度 CV 超限）与睫毛遮挡（ROI 暗块）各精确触发单一 flag。
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from mgd.core.qc import (
    QCPolicy,
    QualityReport,
    QualityThresholds,
    assess_quality,
    decide,
    default_thresholds,
)

SEED = 20261110


def _noisy_gray(height: int = 128, width: int = 256, seed: int = SEED) -> np.ndarray:
    """干净采集图：128 中灰 + 轻微高斯噪声（σ=6），避免全同值退化。"""
    rng = np.random.default_rng(seed)
    img = rng.normal(128.0, 6.0, size=(height, width))
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _checker_gray(cell: int = 16, lo: int = 68, hi: int = 188,
                  height: int = 128, width: int = 256) -> np.ndarray:
    """确定性边缘图：交替棋盘格（68/188），拉普拉斯方差落在 (60, 5000) 边缘区间。"""
    rows = (np.arange(height) // cell)[:, None]
    cols = (np.arange(width) // cell)[None, :]
    return np.where((rows + cols) % 2 == 0, lo, hi).astype(np.uint8)


def _report(score: float) -> QualityReport:
    """构造决策测试用的最小报告。"""
    return QualityReport(score=score)


# ---------------------------------------------------------------- 1 干净灰图
def test_clean_gray_passes() -> None:
    gray = _noisy_gray()
    report = assess_quality(gray)
    assert report.flags == []
    assert report.reasons == []
    assert report.score >= 90
    assert report.metrics["laplacian_var"] >= 60.0  # 远离 blur 阈值
    assert report.metrics["specular_ratio"] < 0.01
    assert report.thresholds == default_thresholds().as_dict()


# ---------------------------------------------------------------- 2 曝光
def test_underexposed_and_overexposed() -> None:
    black = np.zeros((128, 256), dtype=np.uint8)
    rep_black = assess_quality(black)
    assert "underexposed" in rep_black.flags
    assert rep_black.metrics["dark_ratio"] == 1.0
    # 常数图（std=0）：P99.5==值，specular 判据按 0 处理，不崩、不误报
    assert rep_black.metrics["specular_ratio"] == 0.0
    assert "specular" not in rep_black.flags

    white = np.full((128, 256), 255, dtype=np.uint8)
    rep_white = assess_quality(white)
    assert "overexposed" in rep_white.flags
    assert rep_white.metrics["bright_ratio"] == 1.0
    assert rep_white.metrics["specular_ratio"] == 0.0
    assert "specular" not in rep_white.flags

    # float 全黑：min-max 退化 → 直接截断，仍判欠曝
    rep_float = assess_quality(np.zeros((64, 64)))
    assert "underexposed" in rep_float.flags


# ---------------------------------------------------------------- 3 反光
def test_specular_highlight_block() -> None:
    gray = _noisy_gray()
    gray[10:35, 10:35] = 254  # 625/32768 ≈ 1.9% 像素的高光块（用 254 避开 ==255 过曝判据）
    report = assess_quality(gray)
    assert "specular" in report.flags
    assert report.metrics["specular_ratio"] > 0.01
    assert "overexposed" not in report.flags  # 254 而非 255
    assert report.reasons and "反光" in report.reasons[report.flags.index("specular")]


# ---------------------------------------------------------------- 4 模糊
def test_blur_detection() -> None:
    gray = _noisy_gray()
    sharp = assess_quality(gray)
    assert "blur" not in sharp.flags  # 防恒真断言：未模糊图不触发
    assert sharp.metrics["laplacian_var"] >= 60.0

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=15.0)
    soft = assess_quality(blurred)
    assert "blur" in soft.flags
    assert soft.metrics["laplacian_var"] < 60.0
    # 注意：σ15 模糊使噪声极平，uint8 量化后顶部形成平坦亮台，可能额外触发
    # specular（≥P99.5 平台判据的已知局限，见 quality.py docstring），
    # 故此处只断言扣分 ≥ blur 权重，不断言 flags 恰为 ["blur"]。
    assert soft.score <= sharp.score - 25.0  # engine-defined 权重：blur −25


# ---------------------------------------------------------------- 5 翻转
def test_eversion_bounds() -> None:
    gray = _noisy_gray()

    tiny = np.zeros(gray.shape, dtype=np.uint8)
    tiny[40:65, 40:65] = 1  # 25×25 / 32768 ≈ 0.019 ≈ 0.02，且覆盖列 25 < 30
    rep_tiny = assess_quality(gray, eyelid_mask=tiny)
    assert rep_tiny.flags == ["poor_eversion"]
    assert rep_tiny.metrics["eversion_ratio"] < 0.15
    assert rep_tiny.metrics["eyelid_thickness_cv"] is None  # 覆盖列 <30 → None
    assert rep_tiny.metrics["dark_in_roi_ratio"] is not None

    full = np.ones(gray.shape, dtype=np.uint8)  # ROI 覆盖全图 → 面积比 1.0
    rep_full = assess_quality(gray, eyelid_mask=full)
    assert rep_full.flags == ["over_flip"]
    assert rep_full.metrics["eversion_ratio"] == 1.0
    assert rep_full.metrics["eyelid_thickness_cv"] == 0.0  # 厚度恒定 → 不判离焦
    assert rep_full.metrics["dark_in_roi_ratio"] == 0.0


# ---------------------------------------------------------------- 6 阈值可调
def test_threshold_change_flips_conclusion() -> None:
    edge = _checker_gray()
    base = assess_quality(edge)
    lap_var = base.metrics["laplacian_var"]
    assert 60.0 <= lap_var < 5000.0  # 边缘区间图：默认不触发、抬高后触发
    assert "blur" not in base.flags

    strict = assess_quality(
        edge, thresholds=dataclasses.replace(default_thresholds(), blur_min=5000.0)
    )
    assert "blur" in strict.flags
    assert strict.metrics["laplacian_var"] == pytest.approx(lap_var)
    assert strict.score == base.score - 25.0
    assert strict.thresholds["blur_min"] == 5000.0  # 阈值快照可追溯


# ---------------------------------------------------------------- 7 决策
def test_decide_actions_and_validation() -> None:
    policy = QCPolicy()
    assert decide(_report(30.0), policy)["action"] == "reject"
    assert decide(_report(60.0), policy)["action"] == "review"
    assert decide(_report(85.0), policy)["action"] == "accept"
    # 边界：恰等于阈值不触发更严处置（严格小于）
    assert decide(_report(40.0), policy)["action"] == "review"
    assert decide(_report(70.0), policy)["action"] == "accept"
    # 理由为中文非空
    for action in ("reject", "review", "accept"):
        result = decide(_report({"reject": 30.0, "review": 60.0, "accept": 85.0}[action]), policy)
        assert result["reason"]

    # 非法策略：reject >= review → ValueError（validate 与 decide 都要拦）
    with pytest.raises(ValueError):
        QCPolicy(reject_below=70.0, force_review_below=70.0).validate()
    with pytest.raises(ValueError):
        QCPolicy(reject_below=-1.0).validate()
    with pytest.raises(ValueError):
        QCPolicy(force_review_below=101.0).validate()
    with pytest.raises(ValueError):
        decide(_report(30.0), QCPolicy(reject_below=80.0, force_review_below=50.0))


def test_decide_rejects_nonfinite_score() -> None:
    # NaN 会绕过所有 "<" 比较落进 accept 分支：非有限分数必须显式拒绝判定
    policy = QCPolicy()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="非有限"):
            decide(QualityReport(score=bad), policy)


# ---------------------------------------------------------------- 8 eyelid 缺失
def test_missing_eyelid_mask_leaves_roi_metrics_none() -> None:
    report = assess_quality(_noisy_gray(), eyelid_mask=None)
    for key in ("eversion_ratio", "eyelid_thickness_cv", "dark_in_roi_ratio"):
        assert report.metrics[key] is None
    for flag in ("poor_eversion", "over_flip", "defocus", "eyelash"):
        assert flag not in report.flags
    # 全局项仍可测
    assert report.metrics["laplacian_var"] is not None
    assert report.metrics["dark_ratio"] is not None


# ---------------------------------------------------------------- 9 附加：离焦 / 睫毛
def test_defocus_ragged_eyelid_thickness() -> None:
    gray = _noisy_gray()
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[0:10, ::2] = 1   # 偶数列厚 10
    mask[0:60, 1::2] = 1  # 奇数列厚 60 → CV = 25/35 ≈ 0.714
    report = assess_quality(gray, eyelid_mask=mask)
    cv_value = report.metrics["eyelid_thickness_cv"]
    assert cv_value is not None and cv_value > 0.45
    assert report.flags == ["defocus"]  # 面积比 0.27 在 [0.15, 0.85] 内，不触发翻转项


def test_eyelash_dark_block_in_roi() -> None:
    gray = _noisy_gray()
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[64:, :] = 1          # 下半图为眼睑 ROI（面积比 0.5，厚度恒定）
    gray[70:110, 40:100] = 5  # ROI 内暗块（睫毛模拟），占比 2400/16384 ≈ 0.147
    report = assess_quality(gray, eyelid_mask=mask)
    ratio = report.metrics["dark_in_roi_ratio"]
    assert ratio is not None and ratio > 0.12
    assert report.flags == ["eyelash"]
    assert report.metrics["eyelid_thickness_cv"] == 0.0  # 厚度恒定不误判离焦


# ---------------------------------------------------------------- 10 输入校验
def test_input_validation() -> None:
    with pytest.raises(ValueError):
        assess_quality(np.zeros(16, dtype=np.uint8))  # 一维
    with pytest.raises(ValueError):
        assess_quality(np.zeros((4, 4, 3), dtype=np.uint8))  # 三维
    with pytest.raises(ValueError):
        assess_quality(np.zeros((0, 0), dtype=np.uint8))  # 空
    with pytest.raises(ValueError):
        assess_quality(np.full((8, 8), np.nan))  # NaN
    with pytest.raises(ValueError):
        assess_quality(_noisy_gray(8, 8), eyelid_mask=np.ones((4, 4)))  # 掩膜形状不符
    with pytest.raises(ValueError):
        default_thresholds("no-such-policy")  # 未知策略


def test_eyelid_mask_with_nan_is_rejected() -> None:
    # 含 NaN 的掩膜会在 mask > 0 二值化时被静默当成背景 → 翻转/离焦指标
    # 基于残缺数据误判；必须在评估前拒绝
    gray = _noisy_gray(16, 16)
    mask = np.ones((16, 16), dtype=np.float32)
    mask[0, 0] = np.nan
    with pytest.raises(ValueError, match="眼睑掩膜含 NaN"):
        assess_quality(gray, eyelid_mask=mask)
