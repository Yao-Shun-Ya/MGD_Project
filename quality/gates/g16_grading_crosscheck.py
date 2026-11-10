"""g16 分级交叉验证（冗余校验）：Arita 逐点对齐 clinical.meiboscore_from_rloss；
Pult 单调且范围 [0,4]；30% → Arita 1 / Pult 2（两体系不等价的活证据）；grade_table 3 行。"""
from __future__ import annotations

import _common

if True:
    from mgd.core.grading.scales import (
        ARITA_MEIBOSCORE, GRADE_SPECS, PULT_MEIBOSCALE,
        grade_all, grade_from_loss_percent, grade_table, incompatibility_note,
    )
    from mgd.core.imaging.clinical import meiboscore_from_rloss


def run() -> dict:
    gate = _common.Gate("g16_grading_crosscheck")

    mismatches: list[str] = []
    points = 0
    for loss in range(0, 101):  # 0..100 步长 1，逐点冗余对照
        via_grading = grade_from_loss_percent(float(loss), ARITA_MEIBOSCORE)
        via_clinical = meiboscore_from_rloss(float(loss))
        points += 1
        if via_grading != via_clinical:
            mismatches.append(f"loss={loss}: grading={via_grading} clinical={via_clinical}")
    gate.check(not mismatches,
               f"Arita 与 clinical.meiboscore_from_rloss 逐点一致（{points} 点）",
               ("；".join(mismatches[:4]) + " → ") if mismatches else ""
               + "两处切点必须同为 ≤5 / 100/3 / 200/3；改 clinical 必须同步 GradeSpec")

    previous = -1
    monotonic_ok = True
    range_ok = True
    steps = 0
    value = 0.0
    while value <= 100.0:
        grade = grade_from_loss_percent(value, PULT_MEIBOSCALE)
        if grade < previous:
            monotonic_ok = False
        if not (0 <= grade <= 4):
            range_ok = False
        previous = grade
        steps += 1
        value += 0.25
    gate.check(monotonic_ok, f"Pult 分级随缺失率单调不减（{steps} 个采样点，步长 0.25）",
               "Pult Meiboscale 切点 25.0001/50.0001/75.0001 被改动会破坏单调性")
    gate.check(range_ok, "Pult 分级范围 ⊆ [0, 4]",
               "levels=5，封顶 levels-1=4；越界说明封顶逻辑被破坏")

    grades_30 = grade_all(30.0)
    gate.check(grades_30.get("arita_meiboscore") == 1,
               f"30% → Arita 1 级（实际 {grades_30.get('arita_meiboscore')}）",
               "30% < 1/3 → Arita 1 级（互斥声明锚点）")
    gate.check(grades_30.get("pult_meiboscale") == 2,
               f"30% → Pult 2 级（实际 {grades_30.get('pult_meiboscale')}）",
               "30% ∈ (25, 50] → Pult 2 级（互斥声明锚点）")
    gate.check(grades_30.get("arita_meiboscore") != grades_30.get("pult_meiboscale"),
               "30% 双体系级别不等（两体系不等价的活证据）",
               "若相等，说明有人把两套切点合并了——违反互斥声明")

    table = grade_table(30.0)
    gate.check(len(table) == 3, f"grade_table 输出 3 行（表头 + 双体系，实际 {len(table)}）",
               "GRADE_SPECS 注册数变化时同步报告模板")
    gate.check(table[0] == ["分级体系", "级别"], f"grade_table 表头冻结（实际 {table[0]}）")
    gate.check("30%" in incompatibility_note() or "30%" in "".join(str(x) for x in table),
               "互斥声明/对照表含 30% 实例",
               "incompatibility_note 固定文案必须保留 30% 不等价实例")

    gate.note(f"GRADE_SPECS={sorted(GRADE_SPECS)}；Arita {points} 点逐点对照；"
              f"Pult {steps} 点单调/范围扫描")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)
