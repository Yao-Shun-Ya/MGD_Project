"""WP-2 双分级体系（Arita Meiboscore / Pult Meiboscale）与互斥声明。

两套体系切点不同（1/3、2/3 精确切点 vs 25/50/75），同一缺失率会得到
不同级别，数值不可直接互换——互斥声明见 ``incompatibility_note``。
"""
from __future__ import annotations

from .report import GradeResult, build_grade_report, grade_rows, incompatibility_rows
from .scales import (
    ARITA_MEIBOSCORE,
    GRADE_SPECS,
    PULT_MEIBOSCALE,
    GradeSpec,
    grade_all,
    grade_from_loss_percent,
    grade_table,
    incompatibility_note,
    level_label,
)

__all__ = [
    "ARITA_MEIBOSCORE",
    "GRADE_SPECS",
    "PULT_MEIBOSCALE",
    "GradeResult",
    "GradeSpec",
    "build_grade_report",
    "grade_all",
    "grade_from_loss_percent",
    "grade_rows",
    "grade_table",
    "incompatibility_note",
    "incompatibility_rows",
    "level_label",
]
