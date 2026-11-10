"""WP-4 采集质控（quality gate）。

组成：
- :mod:`.quality`：六类采集缺陷判定（模糊 / 曝光 / 反光 / 翻转不全-过度翻转 /
  离焦 / 睫毛遮挡）与 0-100 质控评分；
- :mod:`.policy`：处置策略（accept / review / reject，reject 优先）。

统一约定：eyelid 缺失或不可测项一律 None——不计分、不进 flags，绝不瞎猜；
所有默认阈值均为工程默认，待临床标定；策略名来自
:class:`mgd.core.config.Settings` 的 ``qc_policy`` 字段。
"""
from __future__ import annotations

from .policy import QCPolicy, decide
from .quality import (
    EYELASH_DARK_LEVEL,
    FLAG_ORDER,
    FLAG_WEIGHTS,
    MIN_THICKNESS_COLUMNS,
    QualityReport,
    QualityThresholds,
    assess_quality,
    default_thresholds,
)

__all__ = [
    "EYELASH_DARK_LEVEL",
    "FLAG_ORDER",
    "FLAG_WEIGHTS",
    "MIN_THICKNESS_COLUMNS",
    "QCPolicy",
    "QualityReport",
    "QualityThresholds",
    "assess_quality",
    "decide",
    "default_thresholds",
]
