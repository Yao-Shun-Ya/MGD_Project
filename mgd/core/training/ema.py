"""EMA（指数滑动平均）：评估与保存 best 权重时使用 EMA 参数。

只镜像参数与 buffer，不参与梯度；swap 期间原参数被临时替换，退出时还原。
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @property
    def enabled(self) -> bool:
        return 0.0 < self.decay < 1.0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        for key, value in model.state_dict().items():
            if key not in self.shadow:
                continue
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, payload: dict) -> None:
        self.decay = float(payload.get("decay", self.decay))
        self.shadow = {key: value.clone() for key, value in (payload.get("shadow") or {}).items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)

    @contextlib.contextmanager
    def activated(self, model: nn.Module):
        """with ema.activated(model): 期间模型参数为 EMA 值，退出后还原。"""
        if not self.enabled or not self.shadow:
            yield model
            return
        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=False)
