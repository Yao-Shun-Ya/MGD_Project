"""WP-4 采集质量评估：单帧灰度图的六类采集缺陷判定与 0-100 质控评分。

对外入口 :func:`assess_quality`：输入灰度图（可选眼睑 ROI 掩膜 ``eyelid_mask``），
返回 :class:`QualityReport`（得分 / flags / 中文理由 / 原始量 / 阈值快照）。

判定公式与阈值来源（g = 归一化到 0-255 的灰度图；m = 眼睑 ROI 掩膜，按 >0 二值化；
T = :class:`QualityThresholds`。**所有默认值均为「工程默认，待临床标定」**）：

- ``blur``：``cv2.Laplacian(g, cv2.CV_64F).var() < T.blur_min``。
  阈值 ``blur_min=60``（工程默认，待临床标定）。
- ``underexposed``：``(g == 0).mean() > T.dark_ratio_max``。
  阈值 ``dark_ratio_max=0.02``（工程默认，待临床标定）。
- ``overexposed``：``(g == 255).mean() > T.bright_ratio_max``。
  阈值 ``bright_ratio_max=0.02``（工程默认，待临床标定）。
- ``specular``：``(g >= np.percentile(g, 99.5)).mean() > T.specular_ratio_max``
  （高光平台占比）。阈值 ``specular_ratio_max=0.01``（工程默认，待临床标定）。
  **与任务书的偏差**：任务书字面公式 ``(g > P99.5).mean()`` 数学上不可达——
  按百分位定义，严格大于 P99.5 的像素质量恒 ≤ 0.5%，永远低于 1% 的阈值，
  flag 永不触发。故按同一「高光块面积」语义改用 ``>=``（统计 ≥P99.5 的顶平台，
  即饱和 / 反光块）；退化规则与任务书一致：灰度全同值（std=0，P99.5==该值）时
  specular 判据按 0 处理，避免常数图误报。已知局限：超平滑（重度模糊）图像经
  uint8 量化后顶部会形成平坦亮台，可能额外触发本项——临床标定时需复核。
- ``poor_eversion``：eyelid_mask 提供 → ``(m > 0).sum() / m.size < T.eversion_min``
  （眼睑 ROI 面积 / 全图面积出下界，翻转不全）。阈值 ``eversion_min=0.15``
  （工程默认，待临床标定）。
- ``over_flip``：同上占比 ``> T.overflip_max``（过度翻转）。阈值 ``overflip_max=0.85``
  （工程默认，待临床标定）。
- ``defocus``：eyelid_mask 提供 → 逐列眼睑厚度（每列 ROI 像素数，只统计掩膜实际
  覆盖、即该列 ROI 像素数 > 0 的列）的变异系数 ``CV = std / mean >
  T.eyelid_thickness_cv_max``。覆盖列数 < 30 → None（样本太少不可判）。
  阈值 ``eyelid_thickness_cv_max=0.45``（工程默认，待临床标定）。
- ``eyelash``：eyelid_mask 提供 → ROI 内 ``(g < 40)`` 暗像素占比 >
  T.dark_in_roi_ratio_max。阈值 ``dark_in_roi_ratio_max=0.12``（工程默认，待临床标定）。
  **该判据为未经临床验证的工程启发式**（睫毛在红外睑板腺影像中表现为眼睑 ROI
  内的细暗结构，此处只做暗像素占比统计，不做形态学确认）。

不可测项语义：eyelid_mask 缺失、ROI 为空、覆盖列不足时，对应指标记 None、
不计入分数、不进 flags——**绝不瞎猜**。

计分（engine-defined 权重表，勿当临床结论）：从 100 起按触发的 flag 扣固定
权重——blur −25；underexposed / overexposed 各 −20；specular −15；
poor_eversion / over_flip 各 −20；defocus −15；eyelash −10；下限 0。
权重表见 :data:`FLAG_WEIGHTS`。

输入归一化约定：gray 必须二维。uint8 直接使用；其余 dtype（float / 其他整型 /
布尔）先转 float64 做 min-max 归一化后 ×255（含 NaN/Inf 时抛 ValueError）；
退化输入（max == min，min-max 无定义）退回「直接截断到 [0, 255]」，保留
全黑 / 全白的曝光语义。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import cv2
import numpy as np

from mgd.core.runtime.logging import get_logger

logger = get_logger("qc.quality")

__all__ = [
    "EYELASH_DARK_LEVEL",
    "FLAG_ORDER",
    "FLAG_WEIGHTS",
    "MIN_THICKNESS_COLUMNS",
    "QualityReport",
    "QualityThresholds",
    "assess_quality",
    "default_thresholds",
]

#: flags 的固定输出顺序（确定性，便于下游快照 / 比对）。
FLAG_ORDER: tuple[str, ...] = (
    "blur",
    "underexposed",
    "overexposed",
    "specular",
    "poor_eversion",
    "over_flip",
    "defocus",
    "eyelash",
)

#: 每个 flag 的固定扣分权重（engine-defined，见模块 docstring 的计分表）。
FLAG_WEIGHTS: dict[str, float] = {
    "blur": 25.0,
    "underexposed": 20.0,
    "overexposed": 20.0,
    "specular": 15.0,
    "poor_eversion": 20.0,
    "over_flip": 20.0,
    "defocus": 15.0,
    "eyelash": 10.0,
}

#: 睫毛启发式的暗像素灰度级（/255）。
EYELASH_DARK_LEVEL = 40

#: defocus 逐列厚度统计要求的最小掩膜覆盖列数（少于该数记 None）。
MIN_THICKNESS_COLUMNS = 30


@dataclass(frozen=True)
class QualityThresholds:
    """质控阈值组。所有默认值：工程默认，待临床标定。"""

    blur_min: float = 60.0             # 拉普拉斯方差（归一化到 0-255 灰度后）
    dark_ratio_max: float = 0.02       # 像素==0 的占比上限
    bright_ratio_max: float = 0.02     # 像素==255 的占比上限
    specular_ratio_max: float = 0.01   # ≥P99.5 高光平台的像素占比上限
    eversion_min: float = 0.15         # 眼睑 ROI 面积 / 全图面积 下限（翻转不全）
    overflip_max: float = 0.85         # 同上上限（过度翻转）
    eyelid_thickness_cv_max: float = 0.45  # 眼睑区域厚度变异系数上限（判离焦）
    dark_in_roi_ratio_max: float = 0.12    # 睫毛启发式：眼睑 ROI 内 <40/255 暗像素占比上限

    def as_dict(self) -> dict[str, float]:
        """阈值快照（写入 :attr:`QualityReport.thresholds`，保证报告可追溯）。"""
        return dataclasses.asdict(self)


@dataclass
class QualityReport:
    """单帧采集质控报告。

    score:       0-100（100 = 无任何 flag；扣分权重表见 :data:`FLAG_WEIGHTS`）
    flags:       触发的缺陷名，顺序遵循 :data:`FLAG_ORDER`
    reasons:     中文解释，与 flags 一一对应（每条含数值）
    metrics:     原始量（不可测的项为 None，绝不瞎猜）
    thresholds:  本次评估使用的阈值快照
    """

    score: float
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


#: 已注册的阈值策略（目前仅 "default"；后续策略在此扩充）。
_POLICY_TABLE: dict[str, QualityThresholds] = {
    "default": QualityThresholds(),
}


def default_thresholds(policy: str = "default") -> QualityThresholds:
    """按策略名返回阈值组。

    目前仅支持 ``"default"``（工程默认，待临床标定）；未知策略抛 ValueError。
    上游 :class:`mgd.core.config.Settings` 的 ``qc_policy`` 字段直接作为本参数传入。
    """
    key = str(policy).strip()
    if key not in _POLICY_TABLE:
        known = ", ".join(sorted(_POLICY_TABLE))
        raise ValueError(f"未知质控策略 {policy!r}（当前仅支持：{known}）")
    return _POLICY_TABLE[key]


def _to_gray255(gray: np.ndarray) -> np.ndarray:
    """输入校验并归一化到 0-255 uint8（约定见模块 docstring）。"""
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError(f"gray 必须是二维灰度数组，收到 ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("gray 是空数组，无法评估")
    if arr.dtype == np.uint8:
        return arr
    f = arr.astype(np.float64, copy=False)
    if not bool(np.all(np.isfinite(f))):
        raise ValueError("gray 含 NaN/Inf，无法归一化到 0-255")
    lo = float(f.min())
    hi = float(f.max())
    if hi > lo:
        scaled = (f - lo) / (hi - lo) * 255.0
    else:
        # 退化（全同值）：min-max 无定义 → 直接截断，保留全黑/全白的曝光语义。
        scaled = f
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def _eyelid_stats(
    gray255: np.ndarray, eyelid_mask: np.ndarray
) -> tuple[float, float | None, float | None]:
    """由眼睑掩膜提取 (eversion_ratio, eyelid_thickness_cv, dark_in_roi_ratio)。

    ROI 为空时 eversion_ratio=0.0（可测：翻转完全失败），厚度 CV 与暗像素占比
    无定义 → None。
    """
    mask = np.asarray(eyelid_mask)
    if mask.ndim != 2 or mask.shape != gray255.shape:
        raise ValueError(
            f"eyelid_mask 必须是与 gray 同形状的二维数组，收到 shape={mask.shape}"
        )
    if not bool(np.all(np.isfinite(mask))):
        # NaN 掩膜会在 mask > 0 二值化时被静默当成背景，导致翻转/离焦指标
        # 基于残缺数据误判——含非有限值直接拒绝。
        raise ValueError("眼睑掩膜含 NaN/Inf，无法评估翻转/离焦/睫毛指标")
    roi = mask > 0
    roi_pixels = int(np.count_nonzero(roi))
    eversion_ratio = roi_pixels / float(gray255.size)

    thickness_cv: float | None = None
    dark_in_roi_ratio: float | None = None
    if roi_pixels > 0:
        thickness = roi.sum(axis=0).astype(np.float64)
        covered = thickness[thickness > 0]  # 只统计掩膜实际覆盖的列
        if covered.size >= MIN_THICKNESS_COLUMNS:
            mean = float(covered.mean())
            if mean > 0.0:
                thickness_cv = float(covered.std() / mean)
        dark_in_roi_ratio = float(np.mean(gray255[roi] < EYELASH_DARK_LEVEL))
    return eversion_ratio, thickness_cv, dark_in_roi_ratio


def assess_quality(
    gray: np.ndarray,
    eyelid_mask: np.ndarray | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """评估单帧采集质量。

    参数：
        gray: 二维灰度图。uint8 直接使用；float / 其他 dtype 先 min-max 归一化
            ×255（见模块 docstring 的归一化约定；NaN/Inf 抛 ValueError）。
        eyelid_mask: 可选眼睑 ROI 掩膜，按 >0 二值化，须与 gray 同形状。
            缺失时 eversion / defocus / eyelash 相关指标为 None、不计分、不进 flags。
        thresholds: 阈值组；None 时使用 :func:`default_thresholds`。

    返回：
        :class:`QualityReport`。

    判定公式 / 阈值来源 / 扣分权重表（engine-defined）/ 退化规则 / specular
    判据偏差说明，均见模块 docstring。要点：

    - blur:           Laplacian方差 < blur_min（权重 −25）
    - underexposed:   ==0 占比 > dark_ratio_max（权重 −20）
    - overexposed:    ==255 占比 > bright_ratio_max（权重 −20）
    - specular:       ≥P99.5 平台占比 > specular_ratio_max（权重 −15）
    - poor_eversion:  ROI 面积占比 < eversion_min（权重 −20）
    - over_flip:      ROI 面积占比 > overflip_max（权重 −20）
    - defocus:        逐列厚度 CV > eyelid_thickness_cv_max（权重 −15）
    - eyelash:        ROI 内 <40/255 暗像素占比 > dark_in_roi_ratio_max
                      （权重 −10；未经临床验证的工程启发式）

    异常：
        ValueError: gray 非二维 / 为空 / 含 NaN-Inf，或 eyelid_mask 形状不匹配 /
            含 NaN-Inf。
    """
    th = thresholds if thresholds is not None else default_thresholds()
    gray255 = _to_gray255(gray)

    laplacian_var = float(np.var(cv2.Laplacian(gray255, cv2.CV_64F)))
    dark_ratio = float(np.mean(gray255 == 0))
    bright_ratio = float(np.mean(gray255 == 255))
    if float(gray255.min()) == float(gray255.max()):
        # 灰度全同值（std=0）：P99.5==该值，specular 判据按 0 处理（不崩、不误报）。
        specular_ratio = 0.0
    else:
        p995 = float(np.percentile(gray255, 99.5))
        specular_ratio = float(np.mean(gray255 >= p995))

    if eyelid_mask is None:
        eversion_ratio: float | None = None
        thickness_cv: float | None = None
        dark_in_roi_ratio: float | None = None
        logger.debug("eyelid_mask 未提供：eversion/defocus/eyelash 相关项记 None")
    else:
        eversion_ratio, thickness_cv, dark_in_roi_ratio = _eyelid_stats(gray255, eyelid_mask)

    metrics: dict[str, float | None] = {
        "laplacian_var": laplacian_var,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "specular_ratio": specular_ratio,
        "eversion_ratio": eversion_ratio,
        "eyelid_thickness_cv": thickness_cv,
        "dark_in_roi_ratio": dark_in_roi_ratio,
    }

    triggered: dict[str, str] = {}
    if laplacian_var < th.blur_min:
        triggered["blur"] = (
            f"模糊：拉普拉斯方差 {laplacian_var:.1f} 低于下限 {th.blur_min:g}（图像细节不足）"
        )
    if dark_ratio > th.dark_ratio_max:
        triggered["underexposed"] = (
            f"欠曝：全黑（==0）像素占比 {dark_ratio:.4f} 超过上限 {th.dark_ratio_max:g}"
        )
    if bright_ratio > th.bright_ratio_max:
        triggered["overexposed"] = (
            f"过曝：全白（==255）像素占比 {bright_ratio:.4f} 超过上限 {th.bright_ratio_max:g}"
        )
    if specular_ratio > th.specular_ratio_max:
        triggered["specular"] = (
            f"反光：≥P99.5 高光平台像素占比 {specular_ratio:.4f} "
            f"超过上限 {th.specular_ratio_max:g}"
        )
    if eversion_ratio is not None:
        if eversion_ratio < th.eversion_min:
            triggered["poor_eversion"] = (
                f"翻转不全：眼睑 ROI 面积占比 {eversion_ratio:.4f} 低于下限 {th.eversion_min:g}"
            )
        elif eversion_ratio > th.overflip_max:
            triggered["over_flip"] = (
                f"过度翻转：眼睑 ROI 面积占比 {eversion_ratio:.4f} 超过上限 {th.overflip_max:g}"
            )
    if thickness_cv is not None and thickness_cv > th.eyelid_thickness_cv_max:
        triggered["defocus"] = (
            f"疑似离焦：眼睑逐列厚度变异系数 {thickness_cv:.3f} "
            f"超过上限 {th.eyelid_thickness_cv_max:g}（阈值待临床标定）"
        )
    if dark_in_roi_ratio is not None and dark_in_roi_ratio > th.dark_in_roi_ratio_max:
        triggered["eyelash"] = (
            f"疑似睫毛遮挡：眼睑 ROI 内暗像素（<{EYELASH_DARK_LEVEL}/255）占比 "
            f"{dark_in_roi_ratio:.4f} 超过上限 {th.dark_in_roi_ratio_max:g}"
            f"（工程启发式，未经临床验证）"
        )

    flags = [name for name in FLAG_ORDER if name in triggered]
    reasons = [triggered[name] for name in flags]
    score = max(0.0, 100.0 - sum(FLAG_WEIGHTS[name] for name in flags))
    logger.debug("质控评估完成：score=%.1f flags=%s", score, flags)
    return QualityReport(
        score=score,
        flags=flags,
        reasons=reasons,
        metrics=metrics,
        thresholds=th.as_dict(),
    )
