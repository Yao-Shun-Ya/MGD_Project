"""权重加载：路径解析 + manifest 校验 + 结构加载。"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import (
    Settings,
    canonical_weight_candidates,
    get_settings,
    model_best_path,
)
from ..runtime.device import resolve_device
from ..runtime.logging import get_logger
from ..runtime.errors import ManifestError, ModelNotFoundError, WeightLoadError
from .manifest import assert_loadable, validate_weight
from .registry import create_model, normalize_model_name

logger = get_logger("models.loader")


def resolve_weight_path(
    arch: str,
    weights_path: str | Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """权重解析顺序：显式路径 → 本工程训练产出 → canonical 权重。"""
    settings = settings or get_settings()
    if weights_path is not None:
        path = Path(weights_path)
        if not path.is_file():
            raise ModelNotFoundError(f"权重文件不存在：{path}")
        return path.resolve()

    best = model_best_path(normalize_model_name(arch), settings)
    if best.is_file():
        return best

    if normalize_model_name(arch) == "UNet":
        for candidate in canonical_weight_candidates(settings):
            if candidate.is_file():
                return candidate.resolve()

    raise ModelNotFoundError(
        f"未找到 {arch} 的可用权重",
        hint=f"请先训练并生成 {best}，或放置 {settings.home.name} 下的 canonical 权重",
    )


def _load_state_dict(path: Path, map_location="cpu") -> dict:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 2.0 无 weights_only
        payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict):
        # 兼容本工程完整 ckpt 与旧版权重
        for key in ("model_state", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise WeightLoadError(f"权重内容不是 state_dict：{path}")
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in payload.items()}


def _build_matching_model(arch: str, state: dict) -> nn.Module:
    """优先按检测到的通道数构建；检测不到时依次尝试 2 通道 / 1 通道。"""
    detected = detect_out_channels(state)
    candidates = [detected] if detected in (1, 2) else [2, 1]
    last_error: Exception | None = None
    for channels in candidates:
        try:
            model = create_model(arch, in_channels=1, out_channels=channels)
            model.load_state_dict(state, strict=True)
            model.out_channels = int(channels)
            return model
        except (RuntimeError, KeyError, ValueError) as exc:
            last_error = exc
    raise last_error if last_error is not None else WeightLoadError(f"无法构建 {arch}")


def detect_out_channels(state: dict) -> int | None:
    """从 state_dict 检测输出通道数（用于旧版权重向后兼容）。"""
    if any(k.startswith("gland_head.") for k in state):
        return 2
    weight = state.get("outc.conv.weight")
    if weight is not None and hasattr(weight, "shape") and weight.ndim == 4:
        return int(weight.shape[0])
    weight = state.get("model.2.1.conv.unit0.conv.weight")
    if weight is not None and hasattr(weight, "shape") and weight.ndim == 4:
        return int(weight.shape[0])
    return None


def load_model(
    arch: str,
    weights_path: str | Path | None = None,
    device: str | torch.device | None = None,
    settings: Settings | None = None,
    *,
    allow_unverified: bool = False,
) -> tuple[nn.Module, torch.device, Path]:
    """加载模型到目标设备并置为 eval。

    pending 权重默认拒绝加载；allow_unverified 会记录 WARNING 以便审计追溯。
    """
    settings = settings or get_settings()
    canonical_arch = normalize_model_name(arch)
    path = resolve_weight_path(canonical_arch, weights_path, settings)

    status = validate_weight(path, canonical_arch, settings=settings)
    if status.status == "pending" and allow_unverified:
        logger.warning("加载 pending 权重 %s：%s", path.name, status.reason)
    else:
        assert_loadable(status, allow_unverified=allow_unverified)

    target = resolve_device(device if isinstance(device, str) or device is None else str(device))
    try:
        model = _build_matching_model(canonical_arch, _load_state_dict(path))
    except (RuntimeError, KeyError, ValueError) as exc:
        raise WeightLoadError(
            f"权重与 {canonical_arch} 架构不匹配：{path.name}",
            hint=str(exc),
        ) from exc

    try:
        model = model.to(target)
    except torch.cuda.OutOfMemoryError as exc:
        raise WeightLoadError(
            "加载模型时显存不足",
            hint="请释放显存或使用 MGD_DEVICE=cpu",
        ) from exc
    model.eval()
    return model, target, path


def load_any(path: str | Path) -> dict:
    """只读取 state_dict（供 verify 工具做结构检查）。"""
    return _load_state_dict(Path(path))


def arch_from_state_dict(state: dict) -> str | None:
    """从 key 前缀猜架构，仅用于 verify 的初筛提示。"""
    keys = list(state.keys())
    if any(k.startswith("encoder.patch1.") or k.startswith("encoder.blocks.0.") for k in keys):
        return "MGD-Former"
    if any(k.startswith("gland_head.") or k.startswith("eyelid_head.") for k in keys):
        return "MGD-PriorNet"
    if any(k.startswith("enhancer.") for k in keys):
        return "WRCANet"
    if any(k.startswith("up1.attention.") for k in keys):
        return "AttentionUNet"
    if any(k.startswith("model.0.") or k.startswith("conv_0") for k in keys):
        return "UNet"
    if any(k.startswith("inc.") or k.startswith("down1.") for k in keys):
        return "UNet"
    return None
