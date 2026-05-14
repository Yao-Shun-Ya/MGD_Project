"""
模型定义模块
"""
import torch
from monai.networks.nets import UNet


def get_clinical_model(device, model_name="UNet"):
    """
    获取临床级别的模型
    
    Args:
        device: 设备类型（CPU 或 GPU）
        model_name: 模型架构名称
    
    Returns:
        初始化好的模型
    """
    if model_name == "UNet":
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
    else:
        raise ValueError(f"不支持的模型架构: {model_name}")


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
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    model = get_clinical_model(torch.device("cpu"))
    
    # 先加载到 CPU，避免内存不足问题
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            raise RuntimeError("加载模型时显存不足，请尝试降低显存占用或使用 CPU")
        elif device.type == "mps":
            torch.mps.empty_cache()
            raise RuntimeError("加载模型时 Apple Silicon GPU 内存不足，请尝试降低内存占用或使用 CPU")
        else:
            raise RuntimeError("加载模型时内存不足，请尝试降低内存占用")
    
    return model
