"""Dataset / DataLoader 构建。pin_memory 只在 CUDA 开启。"""
from __future__ import annotations

import torch
from monai.data import DataLoader, Dataset

from .pairing import Pair, build_data_dicts


def build_dataset(pairs: list[Pair], transform) -> Dataset:
    return Dataset(data=build_data_dicts(pairs), transform=transform)


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> DataLoader:
    workers = max(0, int(num_workers))
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=bool(device is not None and device.type == "cuda"),
        persistent_workers=workers > 0,
    )
