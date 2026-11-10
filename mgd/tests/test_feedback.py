"""WP-10 feedback 医生修正捕获与回流单元测试。

全部离线、合成数据、tmp_path 运行，目标耗时 < 10 秒。覆盖：
1. 纯加法（model 全 0，edit 有块）→ added_px=块面积、removed_px=0、direction=add_only；
2. 纯减法（model 有块，edit 全 0）→ removed_px=块面积、direction=remove_only；
3. 混合修正 → mixed，added_px / removed_px 两个计数都正确；
4. reference 提供：dice_before 与手算一致（100 vs 100 重叠 60 → 0.6）；无 reference →
   dice_before=NaN 且 summarize 忽略 NaN；dice_after_vs_edit 的退化口径（Dice(model, edit)）；
5. summarize 方向分布计数正确；export_candidates：降序排序、max_items 截断、
   空 records → ValueError、原子写且每行可 json.loads（无临时文件残留、NaN → null）；
6. 形状不一致 → ValueError（含 reference_mask 形状校验）；
7. 附加：summarize 空 records 不抛异常、掩膜 >0 二值化、ts 默认 UTC、
   掩膜引用语义、纯加法/纯减法走 export 导出的字段正确性。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from mgd.core.feedback import (
    CorrectionRecord,
    export_candidates,
    record_correction,
    summarize,
)

SIZE = 16


def _blank() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=np.uint8)


def _block(top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = _blank()
    mask[top:top + height, left:left + width] = 255
    return mask


# ------------------------------------------------------------ 1 纯加法
def test_pure_addition() -> None:
    rec = record_correction("c1", _blank(), _block(0, 0, 4, 5))
    assert rec.added_px == 20
    assert rec.removed_px == 0
    assert rec.direction == "add_only"


# ------------------------------------------------------------ 2 纯减法
def test_pure_removal() -> None:
    rec = record_correction("c2", _block(2, 2, 3, 4), _blank())
    assert rec.added_px == 0
    assert rec.removed_px == 12
    assert rec.direction == "remove_only"


# ------------------------------------------------------------ 3 混合
def test_mixed_edit() -> None:
    model = _block(0, 0, 4, 4)                 # 16 px
    edited = _block(0, 0, 2, 4) | _block(8, 8, 2, 2)  # 保留 8 px + 新增 4 px
    rec = record_correction("c3", model, edited)
    assert rec.added_px == 4
    assert rec.removed_px == 8
    assert rec.direction == "mixed"


# ------------------------------------------------- 4 reference 的 Dice 口径
def test_dice_with_reference_matches_hand_calc() -> None:
    model = _block(0, 0, 10, 10)               # 100 px
    ref = _block(0, 0, 10, 6) | _block(0, 10, 10, 4)  # 100 px，与 model 重叠 60 px
    rec = record_correction("c4", model, model.copy(), reference_mask=ref)
    assert math.isclose(rec.dice_before, 0.6)
    # edit == model → Dice(edited, ref) 同为 0.6
    assert math.isclose(rec.dice_after_vs_edit, 0.6)


def test_dice_after_degenerate_definition() -> None:
    # 无 reference：dice_after_vs_edit 退化为 Dice(model, edited)
    model = _block(0, 0, 10, 10)
    edited = _block(0, 5, 10, 10)              # 水平错开 5 列 → 重叠 50 px
    rec = record_correction("c6", model, edited)
    assert math.isclose(rec.dice_after_vs_edit, 0.5)
    assert math.isnan(rec.dice_before)


def test_dice_without_reference_is_nan_and_summarize_ignores() -> None:
    model = _block(0, 0, 5, 5)
    rec = record_correction("c5", model, model.copy())
    assert math.isnan(rec.dice_before)
    assert math.isclose(rec.dice_after_vs_edit, 1.0)

    # 一条有 reference（0.6）+ 一条无 reference（NaN）→ 均值只算 0.6
    ref = _block(0, 0, 10, 6) | _block(0, 10, 10, 4)
    with_ref = record_correction("c4", _block(0, 0, 10, 10), _block(0, 0, 10, 10), reference_mask=ref)
    summary = summarize([with_ref, rec])
    assert math.isclose(summary["mean_dice_before"], 0.6)


# ------------------------------------------------------------ 5 汇总与导出
def test_summarize_direction_counts() -> None:
    records = [
        record_correction("d1", _blank(), _block(0, 0, 2, 2)),            # add_only: added 4
        record_correction("d2", _block(0, 0, 2, 2), _blank()),            # remove_only: removed 4
        record_correction("d3", _block(0, 0, 3, 3), _block(1, 1, 3, 3)),  # mixed: added 5, removed 5
        record_correction("d4", _blank(), _blank()),                      # none
        record_correction("d5", _blank(), _block(4, 4, 1, 1)),            # add_only: added 1
    ]
    summary = summarize(records)
    assert summary["n"] == 5
    assert summary["direction_counts"] == {
        "add_only": 2,
        "remove_only": 1,
        "mixed": 1,
        "none": 1,
    }
    assert summary["net_add_cases"] == 2          # d1、d5（d3 增删相抵不算）
    assert math.isclose(summary["mean_added_px"], 2.0)     # (4+0+5+0+1)/5
    assert math.isclose(summary["mean_removed_px"], 1.8)   # (0+4+5+0+0)/5


def test_summarize_empty_records() -> None:
    summary = summarize([])
    assert summary["n"] == 0
    assert summary["direction_counts"] == {"add_only": 0, "remove_only": 0, "mixed": 0, "none": 0}
    assert summary["net_add_cases"] == 0
    assert math.isnan(summary["mean_added_px"])
    assert math.isnan(summary["mean_removed_px"])
    assert math.isnan(summary["mean_dice_before"])
    assert math.isnan(summary["mean_dice_after_vs_edit"])


def test_export_candidates_sort_truncate_and_jsonl(tmp_path) -> None:
    records = [
        record_correction("small", _blank(), _block(0, 0, 1, 1),
                          ts="2026-01-01T00:00:00+00:00", note="s"),
        record_correction("big", _blank(), _block(0, 0, 5, 5),
                          ts="2026-01-02T00:00:00+00:00", reviewer_id="dr_wang", note="大修"),
        record_correction("mid", _block(0, 0, 3, 3), _blank(),
                          ts="2026-01-03T00:00:00+00:00"),
    ]
    out = export_candidates(records, tmp_path, max_items=2)
    assert out == tmp_path / "correction_candidates.jsonl"
    assert out.exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]   # 每行可 json.loads
    assert [row["case_id"] for row in rows] == ["big", "mid"]  # 25 > 9 > 1，降序
    assert rows[0]["added_px"] == 25
    assert rows[0]["removed_px"] == 0
    assert rows[0]["reviewer_id"] == "dr_wang"
    assert rows[0]["note"] == "大修"
    assert rows[0]["ts"] == "2026-01-02T00:00:00+00:00"
    assert rows[0]["dice_before"] is None         # NaN → null，严格 JSON
    # 无 reference 退化口径：Dice(model=全空, edited=25px) = 0.0
    assert rows[0]["dice_after_vs_edit"] == 0.0
    assert rows[1]["dice_before"] is None         # mid：model 9px / edit 空 → 0.0，before 仍为 NaN
    assert rows[1]["dice_after_vs_edit"] == 0.0
    assert set(rows[0]) == {"case_id", "added_px", "removed_px", "dice_before",
                            "dice_after_vs_edit", "ts", "reviewer_id", "note"}

    # max_items 超过记录数 → 全部导出
    out_all = export_candidates(records, tmp_path, max_items=10)
    assert len(out_all.read_text(encoding="utf-8").strip().splitlines()) == 3

    # 原子写无残留：目录里只有最终文件
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "correction_candidates.jsonl"]
    assert leftovers == []


def test_export_empty_records_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        export_candidates([], tmp_path)
    assert list(tmp_path.iterdir()) == []         # 未落任何文件


# ------------------------------------------------------------ 6 形状校验
def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        record_correction("x", np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 9), dtype=np.uint8))
    with pytest.raises(ValueError):
        record_correction("x", _blank(), _blank(), reference_mask=np.zeros((4, 4), dtype=np.uint8))


# ------------------------------------------------------------ 7 附加约定
def test_masks_binarized_over_zero() -> None:
    model = np.full((SIZE, SIZE), 3, dtype=np.uint8)  # 任意 >0 均视为前景
    rec = record_correction("bin", model, _blank())
    assert rec.removed_px == SIZE * SIZE


def test_default_ts_is_utc_aware_iso() -> None:
    rec = record_correction("ts", _blank(), _blank())
    parsed = datetime.fromisoformat(rec.ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_masks_stored_by_reference() -> None:
    model = _blank()
    rec = record_correction("ref", model, _blank())
    model[0, 0] = 7
    assert rec.model_mask[0, 0] == 7              # 引用语义（docstring 约定），非快照


def test_record_type_and_default_note() -> None:
    rec = record_correction("t", _blank(), _block(0, 0, 1, 1))
    assert isinstance(rec, CorrectionRecord)
    assert rec.reviewer_id == "unknown"
    assert rec.note == ""
