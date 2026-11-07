"""检查点往返：完整训练状态必须能恢复，旧格式要能被识别为 legacy。

注意：本机 torch 2.1 + 新版 transformers 会让 torch.optim 的导入链崩溃
（torch.onnx -> transformers -> torch.utils._pytree 缺 register_pytree_node），
因此这里用带 state_dict 协议的最小替身，而不是真实优化器，保证测试与宿主环境解耦。
"""
from __future__ import annotations

import torch

from mgd.core.training.checkpoint import (
    CHECKPOINT_FORMAT,
    apply_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class _Stateful:
    """满足 optimizer / scheduler 的 state_dict 协议。"""

    def __init__(self, payload: dict):
        self.payload = payload

    def state_dict(self) -> dict:
        return dict(self.payload)

    def load_state_dict(self, payload: dict) -> None:
        self.payload = dict(payload)


def _make_tiny():
    model = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
    optimizer = _Stateful({"lr": 0.02, "step": 5})
    scheduler = _Stateful({"step_size": 2, "gamma": 0.5, "last_epoch": 5})
    return model, optimizer, scheduler


def test_full_checkpoint_roundtrip(tmp_path):
    model, optimizer, scheduler = _make_tiny()
    path = tmp_path / "run" / "unet_epoch_003.mgd.ckpt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=3,
        arch="unet",
        scheduler=scheduler,
        best_metric=0.912,
        best_epoch=2,
        history=[{"epoch": 1, "loss": 0.5}],
        config={"batch_size": 2},
    )
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    state = load_checkpoint(path)
    assert state.format_version == CHECKPOINT_FORMAT
    assert state.legacy is False
    assert state.epoch == 3
    assert state.arch == "UNet"
    assert state.best_metric == 0.912
    assert state.best_epoch == 2
    assert state.history == [{"epoch": 1, "loss": 0.5}]
    assert state.config == {"batch_size": 2}
    assert state.optimizer_state == {"lr": 0.02, "step": 5}
    assert state.scheduler_state is not None

    fresh_model, fresh_optimizer, fresh_scheduler = _make_tiny()
    next_epoch = apply_checkpoint(state, fresh_model, fresh_optimizer, fresh_scheduler)
    assert next_epoch == 4
    assert fresh_optimizer.state_dict() == {"lr": 0.02, "step": 5}
    assert fresh_scheduler.state_dict() == {"step_size": 2, "gamma": 0.5, "last_epoch": 5}
    with torch.no_grad():
        for key, value in model.state_dict().items():
            assert torch.equal(value, fresh_model.state_dict()[key])


def test_legacy_plain_state_dict(tmp_path):
    model, _, _ = _make_tiny()
    path = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), path)

    state = load_checkpoint(path)
    assert state.legacy is True
    assert state.format_version == 1
    assert state.epoch == 0

    fresh_model, _, _ = _make_tiny()
    assert apply_checkpoint(state, fresh_model) == 1
    with torch.no_grad():
        for key, value in model.state_dict().items():
            assert torch.equal(value, fresh_model.state_dict()[key])
