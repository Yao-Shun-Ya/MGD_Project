"""模型注册表 + 名称归一化。

MGD-Former 为默认主干（SegFormer-B2 编码器 + 高分辨率融合解码器）；
UNet / AttentionUNet / WRCANet / MGD-PriorNet 保留为基线对照组，可加载、可评测。
"""
from __future__ import annotations

import torch.nn as nn

from ..runtime.errors import ConfigError
from .architectures import AttentionUNet, WRCANetForSegmentation, build_unet
from .mgd_former import MGDFormer, build_mgd_former, load_pretrained_encoder
from .mgd_prior_net import MGDPriorNet

MODEL_NAMES: tuple[str, ...] = ("MGD-Former", "UNet", "AttentionUNet", "WRCANet", "MGD-PriorNet")

DEFAULT_MODEL = "MGD-Former"

#: 基线组（论文对照用）；不参与默认路径
BASELINE_MODELS: tuple[str, ...] = ("UNet", "AttentionUNet", "WRCANet", "MGD-PriorNet")

_ALIASES = {
    "mgdformer": "MGD-Former",
    "mgd-former": "MGD-Former",
    "segformer": "MGD-Former",
    "segformerb2": "MGD-Former",
    "mitb2": "MGD-Former",
    "unet": "UNet",
    "attentionunet": "AttentionUNet",
    "attunet": "AttentionUNet",
    "attention": "AttentionUNet",
    "wrcanet": "WRCANet",
    "wrcanetforsegmentation": "WRCANet",
    "mgdprior": "MGD-PriorNet",
    "mgdpriornet": "MGD-PriorNet",
    "mgd": "MGD-PriorNet",
}


def normalize_model_name(name: str) -> str:
    """把任意写法（MGD Former / mgd_former / SEGFORMER）归一为注册名。"""
    raw = str(name or "").strip()
    if raw in MODEL_NAMES:
        return raw
    key = "".join(ch for ch in raw.lower() if ch.isalnum() or ch == "-")
    key = key.replace("-", "")
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise ConfigError(
            f"未知模型 {name!r}",
            hint=f"可选模型：{', '.join(MODEL_NAMES)}",
        )
    return canonical


def is_supported(name: str) -> bool:
    try:
        normalize_model_name(name)
    except ConfigError:
        return False
    return True


def is_new_architecture(name: str) -> bool:
    return normalize_model_name(name) == "MGD-Former"


def create_model(
    name: str,
    in_channels: int = 1,
    out_channels: int = 1,
    *,
    deep_supervision: bool = True,
    drop_path: float = 0.1,
) -> nn.Module:
    """按注册名创建模型，返回 CPU 上的 logits 输出模块。

    默认单通道（与旧版权重兼容）；双任务训练/推理显式传 out_channels=2。
    MGD-PriorNet 固定双通道；MGD-Former 默认双通道。
    """
    canonical = normalize_model_name(name)
    if canonical == "MGD-Former":
        return build_mgd_former(
            in_channels=in_channels,
            out_channels=int(out_channels),
            deep_supervision=bool(deep_supervision),
            drop_path=drop_path,
        )
    if canonical == "MGD-PriorNet":
        return MGDPriorNet(in_channels=in_channels, out_channels=2)
    channels = int(out_channels)
    if canonical == "UNet":
        return build_unet(in_channels, channels)
    if canonical == "AttentionUNet":
        return AttentionUNet(in_channels=in_channels, out_channels=channels)
    return WRCANetForSegmentation(in_channels=in_channels, out_channels=channels)


def load_pretrained(model: nn.Module, arch: str, source: str = "auto") -> dict:
    """仅 MGD-Former 有预训练编码器；其余架构返回「不适用」。"""
    if not isinstance(model, MGDFormer):
        return {"loaded": False, "source": "n/a", "reason": f"{arch} 不使用预训练编码器"}
    return load_pretrained_encoder(model, source)


def unwrap_outputs(outputs) -> tuple:
    """统一输出契约：返回 (主 logits, 辅助 logits 列表)。

    旧模型输出 Tensor；MGD-Former 输出 {"logits": Tensor, "aux": [Tensor, ...]}。
    """
    if isinstance(outputs, dict):
        return outputs.get("logits"), list(outputs.get("aux") or [])
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 2 and isinstance(outputs[1], (list, tuple)):
            return outputs[0], list(outputs[1])
        return outputs[0], []
    return outputs, []


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
