"""WP-2 双分级体系：Arita/Pult 切点边界、NaN 行为、互斥声明与报告行。

全部离线运行（合成数值，无 IO、无网络）。
"""
from __future__ import annotations

import dataclasses

import pytest

from mgd.core.grading import (
    ARITA_MEIBOSCORE,
    GRADE_SPECS,
    PULT_MEIBOSCALE,
    GradeSpec,
    build_grade_report,
    grade_all,
    grade_from_loss_percent,
    grade_rows,
    grade_table,
    incompatibility_note,
    incompatibility_rows,
    level_label,
)
from mgd.core.imaging.clinical import meiboscore_from_rloss


def test_dual_scale_mapping_and_non_equivalence():
    """0/20/30/60/90% 两套体系的级别；30% → Arita 1 / Pult 2 证明两套不等价。"""
    cases = [(0.0, 0, 0), (20.0, 1, 1), (30.0, 1, 2), (60.0, 2, 3), (90.0, 3, 4)]
    for loss, arita, pult in cases:
        assert grade_from_loss_percent(loss, ARITA_MEIBOSCORE) == arita
        assert grade_from_loss_percent(loss, PULT_MEIBOSCALE) == pult
        grades = grade_all(loss)
        assert grades == {"arita_meiboscore": arita, "pult_meiboscale": pult}

    arita_30, pult_30 = grade_all(30.0).values()
    assert arita_30 != pult_30  # 两套体系不等价的直接证据

    # 通用算法：loss ≤ 0 一律 0 级
    assert grade_all(-0.5) == {"arita_meiboscore": 0, "pult_meiboscale": 0}


def test_boundary_assignment():
    """边界精确归属：Arita 精确 1/3、2/3 切点；Pult ε 切点（>25% 即升级）。"""
    arita_cases = [(33.3333, 1), (33.34, 2), (66.6666, 2), (66.6667, 3), (100.0, 3)]
    for loss, expected in arita_cases:
        assert grade_from_loss_percent(loss, ARITA_MEIBOSCORE) == expected

    pult_cases = [
        (25.0, 1), (25.0001, 2),
        (50.0, 2), (50.0001, 3),
        (75.0, 3), (75.0001, 4),
        (100.0, 4),
    ]
    for loss, expected in pult_cases:
        assert grade_from_loss_percent(loss, PULT_MEIBOSCALE) == expected

    # 66.6667% > 200/3 = 66.6666…%（66.6667 × 3 = 200.0001 > 200），
    # Arita 精确切点下归 3 级，与 clinical.meiboscore_from_rloss 口径一致。
    assert meiboscore_from_rloss(66.6667) == 3
    assert grade_from_loss_percent(66.6667, ARITA_MEIBOSCORE) == meiboscore_from_rloss(66.6667)


def test_nan_returns_minus_one_without_raise():
    """NaN → -1（两套体系一致），不抛异常，报告与表格仍可构建。"""
    nan = float("nan")
    for spec in GRADE_SPECS.values():
        assert grade_from_loss_percent(nan, spec) == -1
    assert grade_all(nan) == {"arita_meiboscore": -1, "pult_meiboscale": -1}

    report = build_grade_report(nan)
    assert len(report) == 2
    assert all(r.grade == -1 for r in report)
    assert all(r.warnings for r in report)
    assert len(grade_table(nan)) == 3  # 表头 + 两套体系，NaN 不崩溃


def test_incompatibility_note_and_table_shapes():
    """互斥声明文案 + 三种表格的行列数。"""
    note = incompatibility_note()
    assert "不可直接互换" in note
    assert "30%" in note  # 具体例子：30% → Arita 1 / Pult 2
    assert "1 级" in note and "2 级" in note

    table = grade_table(30.0)
    assert table[0] == ["分级体系", "级别"]
    assert len(table) == 3  # 表头 + 两套体系
    assert all(len(row) == 2 for row in table)

    rows = grade_rows(30.0)
    assert rows[0] == ["分级体系", "级别", "分级定义", "来源"]
    assert len(rows) == 3  # 表头 + 两套体系
    assert all(len(row) == 4 for row in rows)
    assert all(row[3] for row in rows[1:])  # 来源列非空

    irows = incompatibility_rows(30.0)
    assert len(irows) == 4  # 表头 + 对照两行 + 提示行
    assert all(len(row) == 2 for row in irows)
    assert "不可直接互换" in irows[-1][1]


def test_grade_spec_and_result_frozen():
    """GradeSpec / GradeResult 均不可变：赋值抛 FrozenInstanceError。"""
    with pytest.raises(dataclasses.FrozenInstanceError):
        ARITA_MEIBOSCORE.levels = 3  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        PULT_MEIBOSCALE.cuts = (25.0,)  # type: ignore[misc]

    result = build_grade_report(30.0)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.grade = 0  # type: ignore[misc]


def test_build_grade_report_and_level_labels():
    """build_grade_report：两套体系各一条、source 非空；level_label 取自 definition 语义段。"""
    report = build_grade_report(30.0)
    assert len(report) == 2
    assert [r.key for r in report] == ["arita_meiboscore", "pult_meiboscale"]
    assert all(r.source for r in report)
    assert all(r.loss_percent == 30.0 for r in report)
    assert all(r.warnings == [] for r in report)

    labels = {r.key: r.level_label for r in report}
    assert labels["arita_meiboscore"] == "1 级（缺失 <1/3）"
    assert labels["pult_meiboscale"] == "2 级（26–50%）"

    # 取不到语义段时退回 "级别 {grade}"，不编造
    assert level_label(ARITA_MEIBOSCORE, -1) == "级别 -1"
    assert level_label(ARITA_MEIBOSCORE, 99) == "级别 99"
    # GradeSpec 定义完整性：切点数不超过 levels−1，语义段数与级别数一致
    for spec in GRADE_SPECS.values():
        assert isinstance(spec, GradeSpec)
        assert len(spec.cuts) <= spec.levels - 1
        assert spec.cuts == tuple(sorted(spec.cuts))
        assert len(spec.definition.split("；")) == spec.levels


def test_arita_zero_below_five_percent():
    """[H1] Arita ≤5% 归 0（本项目操作化，与 clinical.meiboscore_from_rloss 对齐）。

    zero_below 声明在 GradeSpec 上：0 < loss ≤ 5.0 一律 0 级，>5% 恢复切点判定；
    Pult 无该规则（zero_below=None），4.9% 仍按 ε 切点判 1 级。
    """
    assert ARITA_MEIBOSCORE.zero_below == 5.0
    assert PULT_MEIBOSCALE.zero_below is None

    # 边界：4.9% → 0；5.0%（含）→ 0；5.1%（>5%）→ 1
    assert grade_from_loss_percent(4.9, ARITA_MEIBOSCORE) == 0
    assert grade_from_loss_percent(5.0, ARITA_MEIBOSCORE) == 0
    assert grade_from_loss_percent(5.1, ARITA_MEIBOSCORE) == 1
    # Pult 不受影响：4.9% 落在 0% < loss ≤ 25% → 1 级
    assert grade_from_loss_percent(4.9, PULT_MEIBOSCALE) == 1
    assert grade_from_loss_percent(5.1, PULT_MEIBOSCALE) == 1
    # 零值 / 负值仍走既有 0 分支，不受 zero_below 影响
    assert grade_from_loss_percent(0.0, ARITA_MEIBOSCORE) == 0
    assert grade_from_loss_percent(-3.0, ARITA_MEIBOSCORE) == 0
    # 5.1% 与 20% 同级（都 < 1/3），说明 zero_below 只影响 (0, 5%] 区间
    assert grade_from_loss_percent(5.1, ARITA_MEIBOSCORE) == grade_from_loss_percent(
        20.0, ARITA_MEIBOSCORE
    )


def test_arita_zero_below_matches_clinical_meiboscore():
    """[H1] 交叉校验：grade_from_loss_percent 与 clinical.meiboscore_from_rloss 逐点一致。"""
    for loss in (0, 3, 5, 5.1, 20, 30, 60, 90):
        expected = meiboscore_from_rloss(loss)
        assert expected >= 0, loss  # 有限输入不该得到 -1
        assert grade_from_loss_percent(loss, ARITA_MEIBOSCORE) == expected, loss
