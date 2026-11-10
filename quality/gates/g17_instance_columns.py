"""g17 冻结契约：INSTANCE_COLUMNS 精确 9 列；atlas.typology 列名常量与之一致；
longitudinal.match 的 REQUIRED_COLUMNS ⊆ INSTANCE_COLUMNS（三处消费端对齐）；
clinical 的 summary 行契约（前 10 行 v2.4.0 口径 + 末 4 行 GA/DI/TI/SI）两分支一致。"""
from __future__ import annotations

import numpy as np

import _common

if True:
    from mgd.core.imaging.clinical import (
        INSTANCE_COLUMNS,
        calculate_clinical_metrics,
    )
    from mgd.core.atlas import typology
    from mgd.core.longitudinal import match

FROZEN = [
    "实例编号", "面积(px)", "长轴(px)", "短轴(px)",
    "弯曲度τ", "平均曲率κ", "最大曲率κ", "相对延伸比", "是否截断",
]
TYPOLOGY_CONSTANTS = ["COLUMN_RELATIVE", "COLUMN_TRUNCATED", "COLUMN_MINOR", "COLUMN_TAU"]

#: clinical.summary 冻结契约：前 10 行为 v2.4.0 口径（顺序不得变），末 4 行为 v2.5.0 追加。
SUMMARY_FROZEN = [
    "腺体缺失率 R_loss (%)", "Meiboscore 分级", "眼睑 ROI 面积(px)", "腺体在 ROI 内面积(px)",
    "检测腺体数量", "腺体覆盖率(全图)", "实例级弯曲度 τ 均值", "实例级平均曲率 κ 均值",
    "截断腺体占比", "量化口径",
    "腺体面积占比 GA (%)", "直径形变指数 DI", "弯曲度指数 TI", "信号指数 SI",
]


def _summary_keys(gland: np.ndarray, eyelid: np.ndarray, gray: np.ndarray) -> list[str]:
    return [row[0] for row in calculate_clinical_metrics(gland, eyelid, gray_img=gray).summary]


def run() -> dict:
    gate = _common.Gate("g17_instance_columns")

    gate.check(len(INSTANCE_COLUMNS) == 9,
               f"INSTANCE_COLUMNS 精确 9 列（实际 {len(INSTANCE_COLUMNS)}）",
               "列数变化是破坏性变更：同步 report/atlas/longitudinal 与本门禁冻结清单")
    gate.check(list(INSTANCE_COLUMNS) == FROZEN,
               f"INSTANCE_COLUMNS 与冻结清单逐项相等（实际 {list(INSTANCE_COLUMNS)}）",
               f"冻结清单 = {FROZEN}")

    for const_name in TYPOLOGY_CONSTANTS:
        value = getattr(typology, const_name, None)
        gate.check(value in INSTANCE_COLUMNS,
                   f"atlas.typology.{const_name}={value!r} ∈ INSTANCE_COLUMNS",
                   "typology 列名常量必须引用 clinical 的冻结列名，不得另造")

    required = list(match.REQUIRED_COLUMNS)
    gate.check(set(required) <= set(INSTANCE_COLUMNS),
               f"longitudinal.match.REQUIRED_COLUMNS ⊆ INSTANCE_COLUMNS（实际 {required}）",
               "随访匹配依赖 clinical 实例表列名；新列先进 INSTANCE_COLUMNS 再进 REQUIRED")
    for column in required:
        gate.check(column in INSTANCE_COLUMNS, f"REQUIRED 列 {column!r} 在冻结清单内")

    columns_from_typology = [getattr(typology, name) for name in TYPOLOGY_CONSTANTS]
    gate.check(len(set(columns_from_typology)) == len(columns_from_typology),
               "typology 四个列名常量互不重复")
    gate.check(len(required) >= 3, f"REQUIRED_COLUMNS 覆盖 ≥3 列（实际 {len(required)}）",
               "匹配至少需要 实例编号/长轴/短轴/弯曲度τ 中的三列以上")

    # ---- clinical.summary 行契约（v2.5.0 冻结）：两套分支行数/键名/顺序必须一致 ----
    gland = np.zeros((64, 128), dtype=np.uint8)
    gland[8:56, 20:30] = 1
    gland[8:56, 44:54] = 1
    eyelid = np.ones((64, 128), dtype=np.uint8)
    gray = np.full((64, 128), 100, dtype=np.uint8)
    normal_keys = _summary_keys(gland, eyelid, gray)
    empty_keys = _summary_keys(np.zeros((64, 128), dtype=np.uint8), eyelid, gray)
    gate.check(normal_keys == SUMMARY_FROZEN,
               f"summary 正常分支 {len(SUMMARY_FROZEN)} 行键名/顺序与冻结清单一致"
               f"（实际 {len(normal_keys)} 行）",
               f"末 4 行必须为 GA/DI/TI/SI；实际尾部 = {normal_keys[10:]}")
    gate.check(empty_keys == SUMMARY_FROZEN,
               f"summary 空掩膜分支 {len(SUMMARY_FROZEN)} 行键名/顺序与冻结清单一致"
               f"（实际 {len(empty_keys)} 行）",
               f"空掩膜分支少写/错序会导致 UI 指标表缺行；实际 = {empty_keys}")
    gate.check(normal_keys[:10] == SUMMARY_FROZEN[:10],
               "summary 前 10 行为 v2.4.0 冻结口径，不得改名或调序",
               f"实际前 10 行 = {normal_keys[:10]}")

    gate.note("冻结清单来自任务书 v2.4.0 交付契约；三处消费端（clinical/atlas/longitudinal）"
              "已逐列核对；summary 14 行（10 + GA/DI/TI/SI）两分支实测比对")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)
