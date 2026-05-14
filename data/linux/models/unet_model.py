"""
模型定义模块
"""
import torch
from monai.networks.nets import UNet


def get_clinical_model(device):
    """
    获取临床级别的 UNet 模型
    
    Args:
        device: 设备类型（CPU 或 GPU）
    
    Returns:
        初始化好的 UNet 模型
    """
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act="PRELU",
        norm="INSTANCE",
        dropout=0.1
    ).to(device)
    return model


def load_model(model_path, device=None):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备类型（如果为 None，自动选择）
    
    Returns:
        加载了权重的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_clinical_model(torch.device("cpu"))
    
    # 先加载到 CPU，避免显存不足问题
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError("加载模型时显存不足，请尝试降低显存占用或使用 CPU")
    
    return model
