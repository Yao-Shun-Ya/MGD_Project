"""预训练编码器加载（MiT-B2）。

离线优先：只在本地预训练目录 / HuggingFace 缓存里找权重；默认不联网。
找不到时返回 {"loaded": False, "reason": ...}，训练侧把它写进 run 的降级记录，
绝不静默假装加载成功。
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_CACHE_PREFIX = {
    "ade20k": "models--nvidia--segformer-b2-finetuned-ade-512-512",
    "imagenet": "models--nvidia--mit-b2",
}


def _candidate_files(source: str) -> list[Path]:
    """按优先级收集预训练权重候选：MGD_PRETRAINED_DIR > 文档承诺的默认目录 > HF 缓存。

    修复记录：此前只查环境变量，文档中的默认目录
    `<root>/models_weights/pretrained` 从未被代码覆盖——不设环境变量时
    即使权重就位也会「明确降级为随机初始化」。
    """
    from ..config import get_settings

    candidates: list[Path] = []
    bases: list[Path] = []
    local = os.environ.get("MGD_PRETRAINED_DIR")
    if local:
        bases.append(Path(local))
    try:
        settings = get_settings()
        bases.append(settings.model_dir / "pretrained")
        bases.append(settings.home / "models_weights" / "pretrained")
    except Exception:  # settings 不可用时只依赖环境变量
        pass
    seen: set[Path] = set()
    for base in bases:
        base = base.resolve() if base.exists() else base
        if base in seen:
            continue
        seen.add(base)
        for name in (f"{source}.safetensors", f"{source}.bin", f"{source}.pth"):
            candidates.append(base / name)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cache_dir = hub / _CACHE_PREFIX.get(source, "")
    if cache_dir.is_dir():
        for snapshot in sorted(cache_dir.glob("snapshots/*")):
            for name in ("pytorch_model.safetensors", "model.safetensors", "pytorch_model.bin"):
                candidates.append(snapshot / name)
    return [path for path in candidates if path.is_file()]


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _pick(state: dict[str, torch.Tensor], head: str, names: tuple[str, ...]):
    """按优先级尝试多个 HF 命名变体。"""
    for name in names:
        key = head + name
        if key in state:
            return state[key]
    return None


def _remap(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF MiT / SegFormer 权重 → 本实现的参数名。

    HF 不同版本的命名并不一致（projection/proj、layer_norm/norm、
    dense1/fc1、dense2/fc2、sr/sr_conv），这里集中兼容，避免“权重下载了却加载 0 个张量”。
    """
    prefix = "segformer.encoder." if any(key.startswith("segformer.encoder.") for key in state) else ""
    mapped: dict[str, torch.Tensor] = {}

    # ---- patch embedding：HF proj / layer_norm ----
    for index in range(4):
        head = f"{prefix}patch_embeddings.{index}."
        if _pick(state, head, ("proj.weight", "projection.weight")) is None:
            break
        target = f"encoder.patch{index + 1}"
        for target_suffix, names in (
            ("proj.weight", ("proj.weight", "projection.weight")),
            ("proj.bias", ("proj.bias", "projection.bias")),
            ("norm.weight", ("layer_norm.weight", "norm.weight")),
            ("norm.bias", ("layer_norm.bias", "norm.bias")),
        ):
            value = _pick(state, head, names)
            if value is not None:
                mapped[f"{target}.{target_suffix}"] = value

    # ---- Transformer blocks ----
    for stage in range(4):
        block_index = 0
        while True:
            head = f"{prefix}block.{stage}.{block_index}."
            if _pick(state, head, ("layer_norm_1.weight", "norm1.weight")) is None:
                break
            base_target = f"encoder.blocks.{stage}.{block_index}"
            for target_suffix, names in (
                ("norm1.weight", ("layer_norm_1.weight", "norm1.weight")),
                ("norm1.bias", ("layer_norm_1.bias", "norm1.bias")),
                ("norm2.weight", ("layer_norm_2.weight", "norm2.weight")),
                ("norm2.bias", ("layer_norm_2.bias", "norm2.bias")),
                ("attn.q.weight", ("attention.self.query.weight", "attention.query.weight")),
                ("attn.q.bias", ("attention.self.query.bias", "attention.query.bias")),
                ("attn.proj.weight", ("attention.output.dense.weight", "attention.proj.weight")),
                ("attn.proj.bias", ("attention.output.dense.bias", "attention.proj.bias")),
                ("ffn.fc1.weight", ("mlp.dense1.weight", "mlp.fc1.weight")),
                ("ffn.fc1.bias", ("mlp.dense1.bias", "mlp.fc1.bias")),
                ("ffn.dw.weight", ("mlp.dwconv.dwconv.weight", "mlp.dw.weight")),
                ("ffn.dw.bias", ("mlp.dwconv.dwconv.bias", "mlp.dw.bias")),
                ("ffn.fc2.weight", ("mlp.dense2.weight", "mlp.fc2.weight")),
                ("ffn.fc2.bias", ("mlp.dense2.bias", "mlp.fc2.bias")),
            ):
                value = _pick(state, head, names)
                if value is not None:
                    mapped[f"{base_target}.{target_suffix}"] = value

            # 本实现把 key/value 融合成一个 Linear
            key_weight = _pick(state, head, ("attention.self.key.weight", "attention.key.weight"))
            value_weight = _pick(state, head, ("attention.self.value.weight", "attention.value.weight"))
            if key_weight is not None and value_weight is not None:
                mapped[f"{base_target}.attn.kv.weight"] = torch.cat([key_weight, value_weight], dim=0)
                key_bias = _pick(state, head, ("attention.self.key.bias", "attention.key.bias"))
                value_bias = _pick(state, head, ("attention.self.value.bias", "attention.value.bias"))
                if key_bias is not None and value_bias is not None:
                    mapped[f"{base_target}.attn.kv.bias"] = torch.cat([key_bias, value_bias], dim=0)

            # SR 缩减卷积：HF 旧版 attention.self.sr.sr_conv / 新版 attention.self.sr
            for target_suffix, names in (
                ("attn.reduce.weight", ("attention.self.sr.sr_conv.weight", "attention.self.sr.weight")),
                ("attn.reduce.bias", ("attention.self.sr.sr_conv.bias", "attention.self.sr.bias")),
                ("attn.norm.weight", ("attention.self.sr.layer_norm.weight", "attention.self.layer_norm.weight")),
                ("attn.norm.bias", ("attention.self.sr.layer_norm.bias", "attention.self.layer_norm.bias")),
            ):
                value = _pick(state, head, names)
                if value is not None:
                    mapped[f"{base_target}.{target_suffix}"] = value

            block_index += 1

    # ---- 各 stage 输出 LayerNorm：HF layer_norm.{i} → 本实现 encoder.norms.{i} ----
    # 缺失时 4 个 stage 输出归一化保持随机初始化，迁移效果静默劣化
    for index in range(4):
        for target_suffix, names in (
            (f"encoder.norms.{index}.weight", (f"layer_norm.{index}.weight", f"norm.{index}.weight")),
            (f"encoder.norms.{index}.bias", (f"layer_norm.{index}.bias", f"norm.{index}.bias")),
        ):
            value = _pick(state, prefix, names)
            if value is not None:
                mapped[target_suffix] = value
    return mapped


def try_load_encoder(model, sources: dict[str, str], source: str = "auto") -> dict:
    """返回加载状态；`loaded=False` 时附带可读原因。"""
    requested = str(source or "auto").lower()
    if requested == "none":
        return {"loaded": False, "source": "none", "reason": "配置显式要求不加载预训练（pretrained=none）"}
    keys = [requested] if requested in sources else list(sources)
    for key in keys:
        files = _candidate_files(key)
        if not files:
            continue
        try:
            state = _load_state_dict(files[0])
            mapped = _remap(state)
            own = model.state_dict()                 # 含 encoder.* 前缀
            encoder_own = model.encoder.state_dict()  # 不含前缀，用于覆盖率
            usable = {k: v for k, v in mapped.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
            if not usable:
                continue
            missing, unexpected = model.load_state_dict(usable, strict=False)
            loaded_ratio = len(usable) / max(1, len(encoder_own))
            missing_encoder = [k for k in missing if k.startswith("encoder.")]
            return {
                "loaded": True,
                "source": key,
                "file": str(files[0]),
                "tensors": len(usable),
                "coverage": round(loaded_ratio, 4),
                "missing": missing_encoder,
                "reason": "已加载本地缓存权重（未联网）",
            }
        except Exception as exc:  # 文件损坏 / 版本不兼容都要如实返回
            last_error = f"{files[0].name}: {exc}"
            continue
    return {
        "loaded": False,
        "source": requested,
        "reason": (
            "本地未找到 SegFormer-B2 预训练权重（检查 MGD_PRETRAINED_DIR 或 HuggingFace 缓存）；"
            "本轮编码器为随机初始化。这是**明确降级**，论文中必须注明。"
        ),
    }
