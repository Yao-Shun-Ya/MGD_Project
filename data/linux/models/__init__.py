"""
模型模块
"""
import torch
from .unet_model import get_clinical_model as get_unet_model
from .unet_model import load_model as load_unet_model


def get_clinical_model(device, model_name="UNet"):
    """
    获取临床模型实例
    
    Args:
        device: 设备 (cpu/cuda)
        model_name: 模型名称
    
    Returns:
        模型实例
    """
    model = create_model(model_name)
    return model.to(device)


def create_model(model_name):
    """
    创建模型实例
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型实例
    """
    model_name_lower = model_name.replace("_", "").replace("-", "").lower()
    
    if "unet" in model_name_lower and "attention" not in model_name_lower and "wr" not in model_name_lower:
        # 使用 unet_model.py 中的 UNet
        from monai.networks.nets import UNet
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.1
        )
    elif "attention" in model_name_lower and "unet" in model_name_lower:
        from .attention_unet import AttentionUNet
        return AttentionUNet(in_channels=1, out_channels=1)
    elif "wrcanet" in model_name_lower or "wr" in model_name_lower:
        from .wrcanet import WRCANetForSegmentation
        return WRCANetForSegmentation(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"未知模型: {model_name}")


def load_model(model_path, device, model_name="UNet"):
    """
    加载预训练模型
    
    Args:
        model_path: 模型权重路径
        device: 设备
        model_name: 模型名称
    
    Returns:
        加载好的模型实例
    """
    model = create_model(model_name)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_available_models():
    """
    获取可用的模型列表
    
    Returns:
        模型名称列表
    """
    return ["UNet", "AttentionUNet", "WRCANet"]


__all__ = ["get_clinical_model", "create_model", "load_model", "get_available_models"]
