"""预处理：训练与推理共用同一套实现。

管线固定为：灰度化 → EXIF 方向校正 → cv2 双线性缩放到 640x1280 → /255 → CLAHE。
旧工程训练用 MONAI Resized(float) + ScaleIntensity，推理用 cv2.resize(uint8)，
两者插值实现不同；这里统一为同一函数，消除口径漂移。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import exposure

from ..config import Settings, get_settings


@dataclass(frozen=True)
class Preprocessed:
    """一张影像的预处理结果。"""

    source_shape: tuple[int, ...]
    resized: np.ndarray  # uint8 (H, W) 灰度，用于可视化与 ROI
    model_input: np.ndarray  # float32 (H, W) ∈ [0, 1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.model_input.shape  # type: ignore[return-value]


def to_gray(image: np.ndarray) -> np.ndarray:
    """任意通道输入 → uint8 灰度（2D）。"""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    else:
        raise ValueError(f"不支持的图像形状：{arr.shape}")

    if gray.dtype == np.uint8:
        return gray
    if gray.dtype in (np.float32, np.float64):
        scale = 255.0 if gray.max() <= 1.0 else 1.0
        return np.clip(gray * scale, 0, 255).astype(np.uint8)
    return np.clip(gray, 0, 255).astype(np.uint8)


def load_gray(path: str | Path) -> np.ndarray:
    """读取影像为 uint8 灰度，并应用 EXIF 方向（训练与推理一致）。"""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        arr = np.array(img)
    return to_gray(arr)


def resize_gray(gray: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """size 为 (height, width)。已是目标尺寸时直接返回（避免重采样抖动）。"""
    target_h, target_w = int(size[0]), int(size[1])
    if gray.shape[:2] == (target_h, target_w):
        return gray.astype(np.uint8, copy=False)
    return cv2.resize(gray, (target_w, target_h), interpolation=interpolation)


def apply_clahe(gray: np.ndarray, clip_limit: float = 0.02) -> np.ndarray:
    """CLAHE，输入 uint8/float，输出 float32 ∈ [0, 1]。"""
    arr = np.asarray(gray)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    out = exposure.equalize_adapthist(arr, clip_limit=float(clip_limit))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preprocess_array(image: np.ndarray, settings: Settings | None = None) -> Preprocessed:
    settings = settings or get_settings()
    gray = to_gray(image)
    resized = resize_gray(gray, settings.target_size, interpolation=cv2.INTER_LINEAR)
    model_input = apply_clahe(resized, clip_limit=settings.clahe_clip_limit)
    return Preprocessed(source_shape=tuple(np.asarray(image).shape), resized=resized, model_input=model_input)


def preprocess_path(path: str | Path, settings: Settings | None = None) -> Preprocessed:
    return preprocess_array(load_gray(path), settings)
