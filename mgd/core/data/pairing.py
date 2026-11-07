"""数据集扫描与配对。

核心修复：按 basename(stem) 建索引三方配对；任何不一致都硬失败（DataIntegrityError），
不再按 sorted 顺序 zip，也不再只是打印警告。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config import SUPPORTED_EXTENSIONS, Settings, get_settings
from ..runtime.errors import DataIntegrityError


@dataclass(frozen=True)
class Pair:
    stem: str
    image: Path
    gland_mask: Path
    eyelid_mask: Path

    def as_dict(self) -> dict[str, str]:
        return {"image": str(self.image), "label": str(self.gland_mask), "eyelid": str(self.eyelid_mask)}


@dataclass
class DatasetIndex:
    pairs: list[Pair] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    def summary(self) -> str:
        lines = [f"样本数：{len(self.pairs)}"]
        if self.skipped:
            lines.append(f"忽略的非图像文件：{len(self.skipped)}")
        return "\n".join(lines)


@dataclass
class SplitResult:
    train: list[Pair]
    val: list[Pair]
    test: list[Pair]
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> str:
        return f"train={len(self.train)} val={len(self.val)} test={len(self.test)}"


def _index_directory(directory: Path, role: str, skipped: list[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise DataIntegrityError(f"{role}目录不存在：{directory}", hint="请检查数据集根目录与子目录命名")
    stems: dict[str, Path] = {}
    for file in sorted(directory.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(str(file))
            continue
        if file.stem in stems:
            raise DataIntegrityError(
                f"{role}目录存在同名文件，无法按 basename 配对：{file.name} 与 {stems[file.stem].name}"
            )
        stems[file.stem] = file
    if not stems:
        raise DataIntegrityError(f"{role}目录为空：{directory}")
    return stems


def _describe_missing(kind: str, missing: set[str], owner: str) -> str:
    preview = "、".join(sorted(missing)[:8])
    suffix = " …" if len(missing) > 8 else ""
    return f"{kind}缺少 {len(missing)} 个样本（对照 {owner}）：{preview}{suffix}"


def scan_dataset(settings: Settings | None = None) -> DatasetIndex:
    """扫描并严格校验三方配对。任何缺失/多余/重名都会抛 DataIntegrityError。"""
    settings = settings or get_settings()
    skipped: list[str] = []

    images = _index_directory(settings.images_dir, "影像", skipped)
    glands = _index_directory(settings.gland_masks_dir, "腺体标签", skipped)
    eyelids = _index_directory(settings.eyelid_masks_dir, "眼睑标签", skipped)

    problems: list[str] = []
    only_images = set(images) - set(glands)
    only_glands = set(glands) - set(images)
    only_gland_vs_eyelid = set(glands) - set(eyelids)
    only_eyelid_vs_gland = set(eyelids) - set(glands)

    if only_images:
        problems.append(_describe_missing("影像", only_images, "腺体标签"))
    if only_glands:
        problems.append(_describe_missing("腺体标签", only_glands, "影像"))
    if only_gland_vs_eyelid:
        problems.append(_describe_missing("腺体标签", only_gland_vs_eyelid, "眼睑标签"))
    if only_eyelid_vs_gland:
        problems.append(_describe_missing("眼睑标签", only_eyelid_vs_gland, "腺体标签"))

    if problems:
        raise DataIntegrityError(
            "数据集三方配对不一致：\n" + "\n".join(f"- {item}" for item in problems),
            hint="影像 / 腺体标签 / 眼睑标签必须使用同一 basename",
        )

    pairs = [
        Pair(stem=stem, image=images[stem], gland_mask=glands[stem], eyelid_mask=eyelids[stem])
        for stem in sorted(images)
    ]
    return DatasetIndex(pairs=pairs, skipped=skipped)


def split_pairs(
    pairs: list[Pair],
    val_ratio: float = 0.15,
    test_ratio: float = 0.0,
    seed: int = 0,
) -> SplitResult:
    """确定性划分；样本过少时只保证 train 非空并给出 warning。"""
    if not pairs:
        raise DataIntegrityError("没有可划分的样本")

    items = list(pairs)
    random.Random(int(seed)).shuffle(items)
    total = len(items)

    val_count = int(round(total * max(0.0, float(val_ratio))) if val_ratio else 0)
    test_count = int(round(total * max(0.0, float(test_ratio))) if test_ratio else 0)
    val_count = min(val_count, max(0, total - 1))
    test_count = min(test_count, max(0, total - 1 - val_count))

    warnings: list[str] = []
    if total < 3 and (val_ratio or test_ratio):
        warnings.append(f"样本仅 {total} 个，无法划分验证/测试集，全部用于训练")
        return SplitResult(train=items, val=[], test=[], warnings=warnings)

    test_start = total - test_count if test_count else total
    val_start = test_start - val_count
    train, val, test = items[:val_start], items[val_start:test_start], items[test_start:]

    if not train:
        raise DataIntegrityError("划分后训练集为空，请降低 val_ratio/test_ratio")
    if val_ratio and not val:
        warnings.append("验证集为空，本轮评估口径只能是训练集自评")
    return SplitResult(train=train, val=val, test=test, warnings=warnings)


def build_data_dicts(pairs: list[Pair]) -> list[dict[str, str]]:
    return [pair.as_dict() for pair in pairs]
