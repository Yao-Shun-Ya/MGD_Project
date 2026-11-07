"""MONAI 数据变换：训练 / 验证 / 推理共用 core.imaging 的预处理实现。

注意：这里不使用 MONAI 的 LoadImaged + Resized，而是直接复用 load_gray / resize_gray /
apply_clahe，确保训练与推理逐像素同源（也顺带修正了旧版 EXIF 方向不一致的问题）。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from monai.transforms import Compose, Lambdad, RandFlipd, RandSpatialCropd, ToTensord

from ..config import Settings, get_settings
from ..imaging.preprocess import apply_clahe, load_gray, resize_gray, to_gray

KEYS = ("image", "label", "eyelid")


def load_gray_channel(path):
    """读取路径 → (1,H,W) uint8 灰度，与推理侧 load_gray 完全一致。"""
    return load_gray(path)[np.newaxis, ...]


def shared_image_preprocess(x, settings: Settings):
    """(1,H,W) → 灰度 → 目标尺寸 → /255 + CLAHE，返回 (1,H,W) float32。"""
    gray = to_gray(x[0])
    resized = resize_gray(gray, settings.target_size, interpolation=cv2.INTER_LINEAR)
    enhanced = apply_clahe(resized, clip_limit=settings.clahe_clip_limit)
    return enhanced[np.newaxis, ...]


def shared_label_preprocess(x, settings: Settings):
    gray = to_gray(x[0])
    resized = resize_gray(gray, settings.target_size, interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.float32)[np.newaxis, ...]


def build_train_transforms(
    *,
    tile_mode: bool = False,
    tile_height: int = 256,
    augmentation: bool = True,
    settings: Settings | None = None,
) -> Compose:
    settings = settings or get_settings()
    tile_h = int(max(32, min(int(tile_height), settings.target_height)))
    tile_w = int(max(32, min(tile_h * 2, settings.target_width)))

    transforms = [
        Lambdad(keys=list(KEYS), func=load_gray_channel),
        Lambdad(keys=["image"], func=lambda x: shared_image_preprocess(x, settings)),
        Lambdad(keys=["label", "eyelid"], func=lambda x: shared_label_preprocess(x, settings)),
    ]

    if augmentation:
        transforms.extend(
            [
                RandFlipd(keys=list(KEYS), spatial_axis=0, prob=0.5),
                RandFlipd(keys=list(KEYS), spatial_axis=1, prob=0.5),
            ]
        )

    if tile_mode:
        transforms.append(
            RandSpatialCropd(
                keys=list(KEYS),
                roi_size=[tile_h, tile_w],
                random_center=True,
                random_size=False,
            )
        )

    transforms.append(ToTensord(keys=list(KEYS), dtype=torch.float32))
    return Compose(transforms)


def build_eval_transforms(*, settings: Settings | None = None) -> Compose:
    """整图、无增强，用于验证/测试。"""
    settings = settings or get_settings()
    return Compose(
        [
            Lambdad(keys=list(KEYS), func=load_gray_channel),
            Lambdad(keys=["image"], func=lambda x: shared_image_preprocess(x, settings)),
            Lambdad(keys=["label", "eyelid"], func=lambda x: shared_label_preprocess(x, settings)),
            ToTensord(keys=list(KEYS), dtype=torch.float32),
        ]
    )
