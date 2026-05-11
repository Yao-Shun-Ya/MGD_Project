import os
import glob
import torch
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
from monai.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure 
from PIL import Image
from monai.data import PILReader

# 创建PILReader并设置reverse_indexing=False
pil_reader = PILReader(reverse_indexing=False)

def load_image_no_rotate(filename):
    """使用PILReader加载图像，不旋转"""
    img, meta = pil_reader.read(filename)
    return img

# 0. 临床级增强：CLAHE 预处理函数与单通道提取函数
# ==========================================
def apply_clahe(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img_enhanced = exposure.equalize_adapthist(img[0], clip_limit=0.02)
    return img_enhanced[np.newaxis, ...]

def extract_single_channel(x):
    return x[0:1, :, :]

# 【核心新增】将 0~255 的掩膜严格二值化为 0 和 1
def binarize_mask(x):
    # 将大于 0 的像素全变成 1.0 (兼容 Tensor 和 Numpy，防多进程报错)
    return (x > 0).float() if isinstance(x, torch.Tensor) else (x > 0).astype('float32')

def build_train_transforms(tile_mode=False, tile_height=256, use_augmentation=False):
    tile_height = int(max(64, min(tile_height, TARGET_HEIGHT)))
    tile_width = int(max(128, min(tile_height * 2, TARGET_WIDTH)))
    transforms = [
        # 使用PILReader并设置reverse_indexing=False，避免图像被旋转
        LoadImaged(keys=["image", "label", "eyelid"], reader="PILReader", reverse_indexing=False),
        EnsureChannelFirstd(keys=["image", "label", "eyelid"]),
        Lambdad(keys=["image", "label", "eyelid"], func=extract_single_channel),
        # 维持接近原始标注清晰度（H=640, W=1280），尽量保留细条纹信息。
        # 图像用双线性插值；掩膜必须用最近邻，避免边缘被插值污染。
        Resized(keys=["image"], spatial_size=[TARGET_HEIGHT, TARGET_WIDTH], mode="bilinear"),
        Resized(keys=["label", "eyelid"], spatial_size=[TARGET_HEIGHT, TARGET_WIDTH], mode="nearest"),
    ]
    # =========================================================
    # 🌟 数据增强 (Data Augmentation)：帮助模型真正理解解剖结构
    # =========================================================
    if use_augmentation:
        transforms.extend([
            # 50% 概率上下翻转（彻底解决颠倒后 Dice 为 0 的问题）
            RandFlipd(keys=["image", "label", "eyelid"], spatial_axis=0, prob=0.5),
            # 50% 概率左右翻转（增加形态学多样性）
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



def build_training_loader(batch_size=1, tile_mode=False, tile_height=256, use_augmentation=False, num_workers=2):
    train_transforms = build_train_transforms(
        tile_mode=tile_mode,
        tile_height=tile_height,
        use_augmentation=use_augmentation,
    )
    train_ds = Dataset(data=data_dicts, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    return train_ds, train_loader

# ==========================================
# 1. 设置路径 (包含原图、腺体标签、眼睑标签)
# ==========================================
TARGET_HEIGHT = 640
TARGET_WIDTH = 1280

data_dir = "./data"
image_path = os.path.join(data_dir, "images/*.jpg")
gland_mask_path = os.path.join(data_dir, "gland_masks/*.png")
eyelid_mask_path = os.path.join(data_dir, "eyelid_masks/*.png") 

images = sorted(glob.glob(image_path))
gland_masks = sorted(glob.glob(gland_mask_path))
eyelid_masks = sorted(glob.glob(eyelid_mask_path))

if not (len(images) == len(gland_masks) == len(eyelid_masks)):
    print(f"⚠️ 警告: 文件数量不一致！图片:{len(images)}, 腺体:{len(gland_masks)}, 眼睑:{len(eyelid_masks)}")

data_dicts = [
    {"image": img, "label": gland, "eyelid": eyelid} 
    for img, gland, eyelid in zip(images, gland_masks, eyelid_masks)
]

# ==========================================
# 2. 定义 MONAI 医疗级变换流水线 (已修正执行顺序与 Lambda 问题)
# ==========================================
# ==========================================
# 3. 创建数据集和加载器 (满血模型显存自适应版)
# ==========================================
check_ds, check_loader = build_training_loader(batch_size=1, tile_mode=False, tile_height=256, use_augmentation=False, num_workers=2)

# ==========================================
# 4. 验证测试 (直接运行此脚本时触发)
# ==========================================
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    print("正在预热数据通道，准备加载批次...")
    check_data = next(iter(check_loader))
    
    image = check_data["image"][0][0]
    gland_label = check_data["label"][0][0]
    eyelid_label = check_data["eyelid"][0][0]
    
    print(f"✅ 成功加载一批数据 (Batch Size: {check_data['image'].shape[0]})")
    print(f"图像尺寸: {image.shape}, 腺体标签: {gland_label.shape}, 眼睑标签: {eyelid_label.shape}")
    
    plt.figure("验证加载 - 完整数据闭环 (含 CLAHE 增强)", figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("原始红外影像 (Image + CLAHE)")
    plt.imshow(image.numpy(), cmap="gray")
    
    plt.subplot(1, 3, 2)
    plt.title("专家标注-睑板腺 (Gland Mask)")
    plt.imshow(gland_label.numpy(), cmap="gray")
    
    plt.subplot(1, 3, 3)
    plt.title("专家标注-眼睑范围 (Eyelid Mask)")
    plt.imshow(eyelid_label.numpy(), cmap="gray")
    
    plt.tight_layout()
    plt.show()