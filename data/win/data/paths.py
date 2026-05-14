"""
数据路径管理模块
"""
import os
import glob
from config import (
    DATA_DIR,
    IMAGE_PATH,
    GLAND_MASK_PATH,
    EYELID_MASK_PATH,
)


def get_data_paths():
    """
    获取训练数据的所有文件路径
    
    Returns:
        (images, gland_masks, eyelid_masks): 三个列表分别包含图像、腺体标签、眼睑标签的路径
    """
    data_dir = DATA_DIR
    
    images = sorted(glob.glob(os.path.join(data_dir, IMAGE_PATH)))
    gland_masks = sorted(glob.glob(os.path.join(data_dir, GLAND_MASK_PATH)))
    eyelid_masks = sorted(glob.glob(os.path.join(data_dir, EYELID_MASK_PATH)))
    
    if not (len(images) == len(gland_masks) == len(eyelid_masks)):
        print(f"⚠️ 警告: 文件数量不一致！图片:{len(images)}, 腺体:{len(gland_masks)}, 眼睑:{len(eyelid_masks)}")
    
    return images, gland_masks, eyelid_masks


def build_data_dicts():
    """
    构建数据字典列表
    
    Returns:
        data_dicts: 包含 image, label, eyelid 路径的字典列表
    """
    images, gland_masks, eyelid_masks = get_data_paths()
    data_dicts = [
        {"image": img, "label": gland, "eyelid": eyelid}
        for img, gland, eyelid in zip(images, gland_masks, eyelid_masks)
    ]
    return data_dicts
