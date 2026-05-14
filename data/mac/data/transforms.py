"""
数据变换模块
包含图像预处理、增强和变换
"""
import torch
import numpy as np
from PIL import Image
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    RandSpatialCropd,
    ScaleIntensityd,
    ToTensord,
    Lambdad,
    RandFlipd,
)
from monai.data import PILReader
from skimage import exposure
from config import TARGET_HEIGHT, TARGET_WIDTH


# 创建 PILReader 实例
pil_reader = PILReader(reverse_indexing=False)


def load_image_no_rotate(filename):
    """
    使用 PILReader 加载图像，不旋转
    """
    img, meta = pil_reader.read(filename)
    return img


def apply_clahe(img):
    """
    应用 CLAHE 增强
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img_enhanced = exposure.equalize_adapthist(img[0], clip_limit=0.02)
    return img_enhanced[np.newaxis, ...]


def extract_single_channel(x):
    """
    提取单通道
    """
    return x[0:1, :, :]


def binarize_mask(x):
    """
    将掩膜严格二值化
    """
    return (x > 0).float() if isinstance(x, torch.Tensor) else (x > 0).astype('float32')


def build_train_transforms(tile_mode=False, tile_height=256, use_augmentation=False):
    """
    构建训练变换流水线
    
    Args:
        tile_mode: 是否使用分块模式
        tile_height: 分块高度
        use_augmentation: 是否使用数据增强
    
    Returns:
        Compose 变换对象
    """
    tile_height = int(max(64, min(tile_height, TARGET_HEIGHT)))
    tile_width = int(max(128, min(tile_height * 2, TARGET_WIDTH)))
    
    transforms = [
        LoadImaged(keys=["image", "label", "eyelid"], reader="PILReader", reverse_indexing=False),
        EnsureChannelFirstd(keys=["image", "label", "eyelid"]),
        Lambdad(keys=["image", "label", "eyelid"], func=extract_single_channel),
        Resized(keys=["image"], spatial_size=[TARGET_HEIGHT, TARGET_WIDTH], mode="bilinear"),
        Resized(keys=["label", "eyelid"], spatial_size=[TARGET_HEIGHT, TARGET_WIDTH], mode="nearest"),
    ]
    
    if use_augmentation:
        transforms.extend([
            RandFlipd(keys=["image", "label", "eyelid"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label", "eyelid"], spatial_axis=1, prob=0.5),
        ])
    
    transforms.extend([
        ScaleIntensityd(keys=["image"]),
        Lambdad(keys=["label", "eyelid"], func=binarize_mask),
        Lambdad(keys=["image"], func=apply_clahe),
    ])
    
    if tile_mode:
        transforms.append(
            RandSpatialCropd(
                keys=["image", "label", "eyelid"],
                roi_size=[tile_height, tile_width],
                random_center=True,
                random_size=False,
            )
        )
    
    transforms.append(ToTensord(keys=["image", "label", "eyelid"]))
    return Compose(transforms)
