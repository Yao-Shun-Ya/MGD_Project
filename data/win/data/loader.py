"""
数据加载器模块
"""
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader
from .paths import build_data_dicts
from .transforms import build_train_transforms
from config import TARGET_HEIGHT, TARGET_WIDTH


class DataManager:
    """
    数据管理类
    统一管理数据集和数据加载器
    """
    
    def __init__(self):
        self.data_dicts = build_data_dicts()
        self.check_ds = None
        self.check_loader = None
        # 初始化默认加载器
        self._init_default_loader()
    
    def _init_default_loader(self):
        """初始化默认的检查加载器"""
        self.check_ds, self.check_loader = self.build_training_loader(
            batch_size=1,
            tile_mode=False,
            tile_height=256,
            use_augmentation=False,
            num_workers=2
        )
    
    def build_training_loader(
        self,
        batch_size=1,
        tile_mode=False,
        tile_height=256,
        use_augmentation=False,
        num_workers=2
    ):
        """
        构建训练数据集和数据加载器
        
        Args:
            batch_size: 批次大小
            tile_mode: 是否分块
            tile_height: 分块高度
            use_augmentation: 是否使用数据增强
            num_workers: 数据加载线程数
        
        Returns:
            (dataset, dataloader): 数据集和加载器
        """
        train_transforms = build_train_transforms(
            tile_mode=tile_mode,
            tile_height=tile_height,
            use_augmentation=use_augmentation,
        )
        
        dataset = Dataset(data=self.data_dicts, transform=train_transforms)
        
        dataloader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=True,
        )
        
        return dataset, dataloader
    
    def get_dataset_size(self):
        """获取数据集大小"""
        return len(self.data_dicts) if self.data_dicts else 0


# 全局数据管理器实例
_data_manager = None


def get_data_manager():
    """获取全局数据管理器单例"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


# 保持向后兼容的变量
check_ds = None
check_loader = None


def init_backward_compatibility():
    """初始化向后兼容的变量"""
    global check_ds, check_loader
    dm = get_data_manager()
    check_ds = dm.check_ds
    check_loader = dm.check_loader


# 初始化
init_backward_compatibility()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("正在预热数据通道，准备加载批次...")
    dm = get_data_manager()
    check_data = next(iter(dm.check_loader))
    
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
