"""WP-4 质控处置策略：把 :class:`QualityReport` 映射为 accept / review / reject。

处置规则（严格小于比较；**reject 优先于 review**）：

- ``report.score < policy.reject_below``            → ``reject``
- ``policy.reject_below <= score < force_review_below`` → ``review``
- ``score >= force_review_below``                    → ``accept``

阈值 ``reject_below=40``、``force_review_below=70`` 均为 **工程默认，待临床标定**。
决策前先执行 :meth:`QCPolicy.validate`，非法策略（不满足
``0 <= reject_below < force_review_below <= 100``）抛 ValueError。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from mgd.core.qc.quality import QualityReport

__all__ = ["QCPolicy", "decide"]


@dataclass(frozen=True)
class QCPolicy:
    """质控处置策略（阈值为工程默认，待临床标定）。

    reject_below:        得分低于该值 → reject
    force_review_below:  reject 未触发且得分低于该值 → review；否则 accept
    """

    reject_below: float = 40.0
    force_review_below: float = 70.0

    def validate(self) -> None:
        """校验 ``0 <= reject_below < force_review_below <= 100``，否则 ValueError。"""
        if not (0.0 <= self.reject_below < self.force_review_below <= 100.0):
            raise ValueError(
                "QCPolicy 非法：需满足 0 <= reject_below < force_review_below <= 100，"
                f"收到 reject_below={self.reject_below!r}, "
                f"force_review_below={self.force_review_below!r}"
            )


def decide(report: QualityReport, policy: QCPolicy) -> dict:
    """按策略对质控报告做处置决策。

    返回 ``{"action": "accept" | "review" | "reject", "reason": 中文理由}``；
    reject 优先于 review；决策前先 :meth:`QCPolicy.validate`（非法策略抛 ValueError）。
    分数为 NaN/Inf 时抛 ValueError 拒绝判定（NaN 会绕过所有 ``<`` 比较、
    落入 accept 分支——非有限分数必须显式拦截，绝不静默放行）。
    """
    policy.validate()
    score = float(report.score)
    if not math.isfinite(score):
        raise ValueError("质控分数非有限值，拒绝判定")
    flags_text = "、".join(report.flags) if report.flags else "无"
    if score < policy.reject_below:
        action = "reject"
        reason = (
            f"质控得分 {score:.1f} 低于拒绝线 {policy.reject_below:g}，"
            f"建议拒绝采集（触发项：{flags_text}）"
        )
    elif score < policy.force_review_below:
        action = "review"
        reason = (
            f"质控得分 {score:.1f} 低于人工复核线 {policy.force_review_below:g}，"
            f"转人工复核（触发项：{flags_text}）"
        )
    else:
        action = "accept"
        reason = f"质控得分 {score:.1f} 达到接受标准（≥ {policy.force_review_below:g}）"
    return {"action": action, "reason": reason}
