"""完整训练状态检查点。

修复点：旧实现只 torch.save(state_dict)，不保存 optimizer / scheduler / AMP scaler / epoch，
导致「续训」实际是冷启动且总轮数超出。新格式带 format_version 与配置快照。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ... import __version__
from ..models.registry import normalize_model_name
from ..runtime.errors import WeightLoadError

CHECKPOINT_FORMAT = 3


@dataclass
class CheckpointState:
    format_version: int
    model_state: dict[str, Any]
    epoch: int = 0
    optimizer_state: dict | None = None
    scheduler_state: dict | None = None
    scaler_state: dict | None = None
    rng_state: dict | None = None
    best_metric: float = float("nan")
    best_epoch: int | None = None
    history: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    data_signature: str | None = None
    dataset_sizes: dict | None = None
    steps: int = 0
    ema_state: dict | None = None
    arch: str = "UNet"
    mgd_version: str = __version__
    created_at: str = ""
    legacy: bool = False
    path: Path | None = None


def _to_cpu(obj: Any) -> Any:
    """递归把张量搬到 CPU，避免把 GPU 张量序列化进检查点。"""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(value) for value in obj)
    return obj


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    *,
    arch: str,
    scheduler=None,
    scaler=None,
    best_metric: float = float("nan"),
    best_epoch: int | None = None,
    history: list[dict] | None = None,
    config: dict | None = None,
    rng_state: dict | None = None,
    data_signature: str | None = None,
    dataset_sizes: dict | None = None,
    steps: int = 0,
    ema_state: dict | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "mgd_version": __version__,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "arch": normalize_model_name(arch),
        "epoch": int(epoch),
        "model_state": _to_cpu(model.state_dict()),
        "optimizer_state": _to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": float(best_metric),
        "best_epoch": best_epoch,
        "history": history or [],
        "config": config or {},
        "rng_state": rng_state,
        "data_signature": data_signature,
        "dataset_sizes": dataset_sizes or {},
        "steps": int(steps),
        "ema_state": _to_cpu(ema_state) if ema_state else None,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_checkpoint(path: str | Path, map_location="cpu") -> CheckpointState:
    target = Path(path)
    if not target.is_file():
        raise WeightLoadError(f"检查点不存在：{target}")
    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(target, map_location=map_location)

    if not isinstance(payload, dict):
        raise WeightLoadError(f"检查点格式无法识别：{target}")

    if "format_version" not in payload:
        legacy_state = payload
        for key in ("model_state", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                legacy_state = payload[key]
                break
        return CheckpointState(
            format_version=1,
            model_state=legacy_state,
            epoch=0,
            legacy=True,
            path=target,
        )

    return CheckpointState(
        format_version=int(payload["format_version"]),
        model_state=payload.get("model_state", {}),
        epoch=int(payload.get("epoch", 0)),
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        scaler_state=payload.get("scaler_state"),
        rng_state=payload.get("rng_state"),
        data_signature=payload.get("data_signature"),
        dataset_sizes=dict(payload.get("dataset_sizes", {}) or {}),
        steps=int(payload.get("steps", 0) or 0),
        ema_state=payload.get("ema_state"),
        best_metric=float(payload.get("best_metric", float("nan"))),
        best_epoch=payload.get("best_epoch"),
        history=list(payload.get("history", [])),
        config=dict(payload.get("config", {})),
        arch=payload.get("arch", "UNet"),
        mgd_version=payload.get("mgd_version", "unknown"),
        created_at=payload.get("created_at", ""),
        path=target,
    )


def apply_checkpoint(
    state: CheckpointState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    restore_rng: bool = True,
) -> int:
    """恢复全部训练状态，返回下一个 epoch 序号。"""
    try:
        model.load_state_dict(state.model_state, strict=True)
    except RuntimeError as exc:
        raise WeightLoadError("检查点与当前模型架构不匹配", hint=str(exc)) from exc

    if optimizer is not None and state.optimizer_state:
        optimizer.load_state_dict(state.optimizer_state)
    if scheduler is not None and state.scheduler_state:
        scheduler.load_state_dict(state.scheduler_state)
    if scaler is not None and state.scaler_state:
        scaler.load_state_dict(state.scaler_state)
    if restore_rng and state.rng_state:
        from .determinism import restore_rng_state

        restore_rng_state(state.rng_state)
    return int(state.epoch) + 1
