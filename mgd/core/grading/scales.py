"""WP-2 双分级体系定义：Arita Meiboscore 与 Pult Meiboscale。

通用算法：loss ≤ 0 → 0 级；否则级别 = 1 + 不超过 loss 的切点数
（loss ≥ cut 即计入——Pult 的 ε 切点 25.0001 本义即 "略高于 25% 就升一级"）；
超出最高级封顶 levels−1；NaN → −1 并记 warning，不抛异常。

Arita 采用 (100/3, 200/3) 精确切点，与 2026-11 修复后的
``imaging.clinical.meiboscore_from_rloss`` 边界口径一致（该函数不受本模块影响）。

Arita 的 "≤5% 归 0" 由 ``GradeSpec.zero_below`` 声明（本项目操作化，
与 ``imaging.clinical.meiboscore_from_rloss`` 的 ``r_loss_percent <= 5.0 → 0``
口径对齐）；Pult 无此规则（``zero_below=None``），分级完全由切点决定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..runtime.logging import get_logger

logger = get_logger("grading.scales")


@dataclass(frozen=True)
class GradeSpec:
    """一套分级体系的静态定义（frozen，不可变）。"""

    key: str
    name_zh: str
    levels: int
    cuts: tuple[float, ...]  # 缺失率百分比切点，升序
    definition: str
    source: str
    # 缺失率 (0, zero_below] 区间强制归 0 级（None = 无此规则）。
    # ≤5% 归 0 为本项目操作化，与 imaging.clinical.meiboscore_from_rloss 对齐。
    zero_below: float | None = None


ARITA_MEIBOSCORE = GradeSpec(
    "arita_meiboscore",
    "Arita 睑板腺评分",
    4,
    (100.0 / 3.0, 200.0 / 3.0),
    "0 = 无缺失（本项目操作化：缺失率 ≤5% 归 0）；1 = 缺失 <1/3；2 = 1/3–2/3；3 = >2/3（按眼睑面积）",
    "Arita et al., 2008/2009；≤5% 归 0 为本项目操作化",
    zero_below=5.0,
)
PULT_MEIBOSCALE = GradeSpec(
    "pult_meiboscale",
    "Pult Meiboscale",
    5,
    (25.0001, 50.0001, 75.0001),
    "0 = 0%；1 = ≤25%；2 = 26–50%；3 = 51–75%；4 = >75%",
    "Pult & Nichols, 2016",
)

GRADE_SPECS: dict[str, GradeSpec] = {
    "arita_meiboscore": ARITA_MEIBOSCORE,
    "pult_meiboscale": PULT_MEIBOSCALE,
}

_TABLE_HEADER = ["分级体系", "级别"]

# 互斥声明固定文案：切点不同 → 同一缺失率不同级别 → 数值不可直接互换。
_INCOMPATIBILITY_NOTE = (
    "互斥声明：两套分级体系切点不同——Arita Meiboscore 以眼睑面积缺失 1/3、2/3 为界"
    "（精确 33.333…% 与 66.666…%），Pult Meiboscale 以 25%、50%、75% 为界；"
    "同一缺失率在两套体系下会得到不同级别，评分数值不可直接互换、不可相互换算。"
    "例如缺失率 30% 时：Arita 评 1 级（缺失 <1/3），Pult Meiboscale 评 2 级（26–50%）。"
)


def grade_from_loss_percent(loss_percent: float, spec: GradeSpec) -> int:
    """缺失率百分比 → 该体系级别。

    - NaN（或非数值输入）→ −1，记 warning，不抛异常；
    - loss ≤ 0 → 0；
    - ``0 < loss ≤ spec.zero_below`` → 0（如 Arita 的 ≤5% 归 0：本项目操作化，
      与 clinical.meiboscore_from_rloss 对齐；zero_below=None 的体系无此分支）；
    - 否则 1 + 不超过 loss 的切点数，封顶 levels − 1。
    """
    try:
        value = float(loss_percent)
    except (TypeError, ValueError):
        logger.warning("grade_from_loss_percent：输入 %r 非数值，%s 记为 -1", loss_percent, spec.key)
        return -1
    if math.isnan(value):
        logger.warning("grade_from_loss_percent：缺失率为 NaN，%s 记为 -1", spec.key)
        return -1
    if value <= 0.0:
        return 0
    if spec.zero_below is not None and value <= spec.zero_below:
        return 0  # 本项目操作化：如 Arita 缺失率 ≤5% 归 0（与 clinical 口径对齐）
    passed = sum(1 for cut in spec.cuts if value >= cut)
    return min(1 + passed, spec.levels - 1)


def grade_all(loss_percent: float) -> dict[str, int]:
    """两套体系同时分级：{"arita_meiboscore": x, "pult_meiboscale": y}。"""
    return {key: grade_from_loss_percent(loss_percent, spec) for key, spec in GRADE_SPECS.items()}


def level_label(spec: GradeSpec, grade: int) -> str:
    """从 ``spec.definition`` 提取第 grade+1 段的语义描述，形如 "2 级（1/3–2/3）"。

    definition 按全角分号"；"拆段，取第 grade+1 段（1 起计）中 "=" 之后的语义部分；
    取不到（grade < 0 或越界）时退回 "级别 {grade}"，绝不编造。
    """
    if 0 <= grade < spec.levels:
        parts = spec.definition.split("；")
        if grade < len(parts):
            segment = parts[grade]
            _, marker, desc = segment.partition("=")
            desc = desc.strip() if marker else segment.strip()
            if desc:
                return f"{grade} 级（{desc}）"
    return f"级别 {grade}"


def grade_table(loss_percent: float) -> list[list[str]]:
    """报告用两列对照表（含表头行）：[[分级体系, 级别], ...]。"""
    rows: list[list[str]] = [list(_TABLE_HEADER)]
    for key, spec in GRADE_SPECS.items():
        grade = grade_from_loss_percent(loss_percent, spec)
        rows.append([spec.name_zh, level_label(spec, grade)])
    return rows


def incompatibility_note() -> str:
    """固定互斥声明（含 30% → Arita 1 级 / Pult 2 级 的实例）。"""
    return _INCOMPATIBILITY_NOTE
