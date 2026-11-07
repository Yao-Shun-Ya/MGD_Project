"""持久化 70/15/15 划分、K 折、小样本回退与空载防火墙。"""
from __future__ import annotations

import json

import pytest

from mgd.core.data.pairing import Pair
from mgd.core.data.splits import (
    SplitSpec,
    get_splits,
    load_or_create_splits,
    splits_path,
    stems_signature,
)
from mgd.core.runtime.errors import DataIntegrityError


def _pairs(tmp_path, n: int) -> list[Pair]:
    return [
        Pair(stem=f"case_{i:03d}", image=tmp_path / f"{i}.png",
             gland_mask=tmp_path / f"{i}.png", eyelid_mask=tmp_path / f"{i}.png")
        for i in range(n)
    ]


def test_holdout_is_70_15_15_and_persisted(settings, tmp_path):
    pairs = _pairs(tmp_path, 100)
    split = get_splits(pairs, SplitSpec(), settings)
    assert (len(split.train), len(split.val), len(split.test)) == (70, 15, 15)

    payload = json.loads(splits_path(settings).read_text(encoding="utf-8"))
    assert payload["signature"] == stems_signature([p.stem for p in pairs])
    assert payload["entries"]["holdout"]["train"]
    assert "source" not in payload["entries"]["holdout"]


def test_second_call_reuses_exact_order(settings, tmp_path):
    pairs = _pairs(tmp_path, 40)
    first = get_splits(pairs, SplitSpec(), settings)
    second = get_splits(pairs, SplitSpec(), settings)
    assert [p.stem for p in first.train] == [p.stem for p in second.train]
    assert [p.stem for p in first.test] == [p.stem for p in second.test]
    record = load_or_create_splits(pairs, SplitSpec(), settings)
    assert record.source == "persisted"


def test_kfold_partitions_cover_every_sample_once(settings, tmp_path):
    pairs = _pairs(tmp_path, 50)
    split = get_splits(pairs, SplitSpec(mode="kfold", fold=0), settings)
    assert len(split.train) == 30
    assert len(split.val) == 10
    assert len(split.test) == 10
    all_stems = [p.stem for p in split.train + split.val + split.test]
    assert sorted(all_stems) == sorted(p.stem for p in pairs)
    assert len(set(all_stems)) == 50


def test_small_dataset_falls_back_with_warning(settings, tmp_path):
    pairs = _pairs(tmp_path, 6)
    split = get_splits(pairs, SplitSpec(), settings)
    assert len(split.train) >= 1 and len(split.val) >= 1 and len(split.test) >= 1
    assert any("降级" in item for item in split.warnings)


def test_two_samples_triggers_empty_load_firewall(settings, tmp_path):
    with pytest.raises(DataIntegrityError, match="防火墙"):
        load_or_create_splits(_pairs(tmp_path, 2), SplitSpec(), settings)


def test_signature_changes_when_file_list_changes(settings, tmp_path):
    pairs = _pairs(tmp_path, 20)
    get_splits(pairs, SplitSpec(), settings)
    changed = pairs[:-1]
    split = get_splits(changed, SplitSpec(), settings)
    assert len(split.train) + len(split.val) + len(split.test) == 19
