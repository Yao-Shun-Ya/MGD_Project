"""WP-2 分级报告：两套体系各一条结果 + 报告表格行 + 互斥声明行。"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..runtime.logging import get_logger
from .scales import (
    GRADE_SPECS,
    GradeSpec,
    grade_from_loss_percent,
    grade_table,
    incompatibility_note,
    level_label,
)

logger = get_logger("grading.report")

_NAN_WARNING = "缺失率为 NaN，无法分级，级别记为 -1"

_ROWS_HEADER = ["分级体系", "级别", "分级定义", "来源"]


@dataclass(frozen=True)
class GradeResult:
    """单一分级体系对一个缺失率的分级结果（frozen，不可变）。"""

    key: str
    name_zh: str
    loss_percent: float
    grade: int
    level_label: str
    definition: str
    source: str
    warnings: list[str] = field(default_factory=list)


def build_grade_report(loss_percent: float) -> list[GradeResult]:
    """两套体系各生成一条 GradeResult；NaN → grade=-1 并附 warning，不抛异常。"""
    results: list[GradeResult] = []
    for key, spec in GRADE_SPECS.items():
        grade = grade_from_loss_percent(loss_percent, spec)
        warnings = [_NAN_WARNING] if grade < 0 else []
        if grade < 0:
            logger.warning("build_grade_report：%s 在缺失率 %r 下无法分级", key, loss_percent)
        results.append(_to_result(key, spec, loss_percent, grade, warnings))
    return results


def _to_result(
    key: str, spec: GradeSpec, loss_percent: float, grade: int, warnings: list[str]
) -> GradeResult:
    return GradeResult(
        key=key,
        name_zh=spec.name_zh,
        loss_percent=loss_percent,
        grade=grade,
        level_label=level_label(spec, grade),
        definition=spec.definition,
        source=spec.source,
        warnings=warnings,
    )


def grade_rows(loss_percent: float) -> list[list[str]]:
    """供报告表格渲染（含表头）：体系名 / 级别 / 定义 / 来源 四列。"""
    rows: list[list[str]] = [list(_ROWS_HEADER)]
    for result in build_grade_report(loss_percent):
        rows.append([result.name_zh, result.level_label, result.definition, result.source])
    return rows


def incompatibility_rows(loss_percent: float) -> list[list[str]]:
    """两列对照表（grade_table）+ 末行"不可直接互换"提示行。"""
    rows = grade_table(loss_percent)
    rows.append(["互斥声明", incompatibility_note()])
    return rows
