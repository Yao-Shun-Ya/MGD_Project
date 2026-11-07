"""配对与划分：按 basename 索引 + 三方一致性硬校验。"""
from __future__ import annotations

from pathlib import Path

import pytest

from mgd.core.config import Settings
from mgd.core.data.pairing import (
    Pair,
    build_data_dicts,
    scan_dataset,
    split_pairs,
)
from mgd.core.runtime.errors import DataIntegrityError


def _touch(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path, target_height=64, target_width=128)


def test_pairing_uses_basename_not_sorted_index(tmp_path):
    settings = _make_settings(tmp_path)
    for index, stem in enumerate(("1", "10", "2")):
        _touch(settings.images_dir / f"{stem}.jpg", bytes([index]))
        _touch(settings.gland_masks_dir / f"{stem}.png", bytes([index + 10]))
        _touch(settings.eyelid_masks_dir / f"{stem}.png")

    index = scan_dataset(settings)
    assert [pair.stem for pair in index.pairs] == ["1", "10", "2"]
    for pair in index.pairs:
        assert pair.image.stem == pair.stem
        assert pair.gland_mask.stem == pair.stem
        assert pair.eyelid_mask.stem == pair.stem
        # 内容按 basename 对应，而不是目录内顺序
        assert pair.image.read_bytes() == bytes([["1", "10", "2"].index(pair.stem)])
        assert pair.gland_mask.read_bytes() == bytes([["1", "10", "2"].index(pair.stem) + 10])


def test_missing_gland_mask_aborts(tmp_path):
    settings = _make_settings(tmp_path)
    _touch(settings.images_dir / "a.jpg")
    _touch(settings.images_dir / "b.jpg")
    _touch(settings.gland_masks_dir / "a.png")
    _touch(settings.eyelid_masks_dir / "a.png")
    _touch(settings.eyelid_masks_dir / "b.png")
    with pytest.raises(DataIntegrityError, match="影像缺少 1 个样本"):
        scan_dataset(settings)


def test_extra_gland_mask_aborts(tmp_path):
    settings = _make_settings(tmp_path)
    _touch(settings.images_dir / "a.jpg")
    _touch(settings.gland_masks_dir / "a.png")
    _touch(settings.gland_masks_dir / "extra.png")
    _touch(settings.eyelid_masks_dir / "a.png")
    _touch(settings.eyelid_masks_dir / "extra.png")
    with pytest.raises(DataIntegrityError, match="腺体标签缺少 1 个样本"):
        scan_dataset(settings)


def test_eyelid_mismatch_aborts(tmp_path):
    settings = _make_settings(tmp_path)
    _touch(settings.images_dir / "a.jpg")
    _touch(settings.gland_masks_dir / "a.png")
    _touch(settings.eyelid_masks_dir / "b.png")
    with pytest.raises(DataIntegrityError, match="眼睑标签"):
        scan_dataset(settings)


def test_duplicate_stem_aborts(tmp_path):
    settings = _make_settings(tmp_path)
    _touch(settings.images_dir / "a.jpg")
    _touch(settings.images_dir / "a.png")
    _touch(settings.gland_masks_dir / "a.png")
    _touch(settings.eyelid_masks_dir / "a.png")
    with pytest.raises(DataIntegrityError, match="同名文件"):
        scan_dataset(settings)


def test_missing_directory_aborts(tmp_path):
    settings = _make_settings(tmp_path)
    _touch(settings.images_dir / "a.jpg")
    with pytest.raises(DataIntegrityError, match="腺体标签目录不存在"):
        scan_dataset(settings)
    assert not settings.eyelid_masks_dir.exists()


def test_empty_directory_aborts(tmp_path):
    settings = _make_settings(tmp_path)
    settings.images_dir.mkdir(parents=True)
    settings.gland_masks_dir.mkdir(parents=True)
    settings.eyelid_masks_dir.mkdir(parents=True)
    with pytest.raises(DataIntegrityError, match="影像目录为空"):
        scan_dataset(settings)


def test_non_image_files_are_skipped_and_reported(tmp_path):
    settings = _make_settings(tmp_path)
    _touch(settings.images_dir / "a.jpg")
    _touch(settings.images_dir / "notes.txt")
    _touch(settings.gland_masks_dir / "a.png")
    _touch(settings.eyelid_masks_dir / "a.png")
    index = scan_dataset(settings)
    assert len(index.pairs) == 1
    assert len(index.skipped) == 1
    assert "notes.txt" in index.skipped[0]


def _pairs(count: int) -> list[Pair]:
    return [
        Pair(stem=f"s{index}", image=Path(f"i{index}.jpg"),
             gland_mask=Path(f"g{index}.png"), eyelid_mask=Path(f"e{index}.png"))
        for index in range(count)
    ]


def test_split_is_disjoint_and_deterministic():
    pairs = _pairs(20)
    first = split_pairs(pairs, val_ratio=0.2, test_ratio=0.1, seed=42)
    second = split_pairs(pairs, val_ratio=0.2, test_ratio=0.1, seed=42)

    assert [p.stem for p in first.train] == [p.stem for p in second.train]
    assert len(first.val) == 4 and len(first.test) == 2
    assert len(first.train) == 14

    all_stems = [p.stem for p in first.train + first.val + first.test]
    assert len(all_stems) == len(set(all_stems)) == 20
    assert set(all_stems) == {p.stem for p in pairs}


def test_split_different_seed_changes_assignment():
    pairs = _pairs(20)
    a = split_pairs(pairs, val_ratio=0.2, seed=1)
    b = split_pairs(pairs, val_ratio=0.2, seed=2)
    assert [p.stem for p in a.val] != [p.stem for p in b.val]


def test_split_with_tiny_dataset_warns_and_keeps_train():
    result = split_pairs(_pairs(2), val_ratio=0.2, seed=0)
    assert len(result.train) == 2
    assert result.val == [] and result.test == []
    assert "无法划分验证" in result.warnings[0]


def test_split_empty_raises():
    with pytest.raises(DataIntegrityError):
        split_pairs([])


def test_build_data_dicts_keys():
    dicts = build_data_dicts(_pairs(2))
    assert set(dicts[0]) == {"image", "label", "eyelid"}
    assert dicts[0]["label"].endswith("g0.png")
