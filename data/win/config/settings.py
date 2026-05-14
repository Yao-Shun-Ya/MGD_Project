"""
项目配置模块
存储所有全局常量和配置
"""
import os

# 图像尺寸配置
TARGET_HEIGHT = 640
TARGET_WIDTH = 1280

# 数据路径配置
DATA_DIR = "./data"
IMAGE_PATH = "images/*.jpg"
GLAND_MASK_PATH = "gland_masks/*.png"
EYELID_MASK_PATH = "eyelid_masks/*.png"

# 模型保存根目录
MODEL_DIR = "./models_weights"

# 默认模型路径（兼容旧版本）
MODEL_BEST_PATH = "meibomian_model_best.pth"
PRESET_FILE = "infer_presets.json"

# 可用模型列表
AVAILABLE_MODELS = ["UNet", "AttentionUNet", "WRCANet"]


def get_model_best_path(model_name):
    """
    获取指定模型的最佳权重路径
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型权重路径
    """
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{model_name.lower()}_best.pth")


def get_model_checkpoint_path(model_name, epoch):
    """
    获取指定模型的检查点路径
    
    Args:
        model_name: 模型名称
        epoch: 训练轮数
    
    Returns:
        检查点路径
    """
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{model_name.lower()}_epoch_{epoch}.pth")


def get_all_trained_models():
    """
    获取所有已训练的模型
    
    Returns:
        已训练模型名称列表
    """
    if not os.path.exists(MODEL_DIR):
        return []
    
    trained_models = []
    for model_name in AVAILABLE_MODELS:
        model_dir = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(model_dir):
            best_path = get_model_best_path(model_name)
            if os.path.exists(best_path):
                trained_models.append(model_name)
    
    return trained_models

# 内置推理预设
BUILTIN_PRESETS = {
    "平衡-默认": {
        "tile_mode": True,
        "tile_size": 192,
        "tile_overlap": 0.25,
        "threshold": 0.70,
        "min_area": 120,
        "smooth_on": True,
        "smooth_sigma": 1.2,
        "roi_only": True,
        "tta_mode": "关闭",
    },
    "保守-低误检": {
        "tile_mode": True,
        "tile_size": 256,
        "tile_overlap": 0.35,
        "threshold": 0.80,
        "min_area": 180,
        "smooth_on": True,
        "smooth_sigma": 1.0,
        "roi_only": True,
        "tta_mode": "关闭",
    },
    "敏感-高召回": {
        "tile_mode": True,
        "tile_size": 192,
        "tile_overlap": 0.30,
        "threshold": 0.60,
        "min_area": 80,
        "smooth_on": True,
        "smooth_sigma": 1.3,
        "roi_only": True,
        "tta_mode": "关闭",
    },
}
