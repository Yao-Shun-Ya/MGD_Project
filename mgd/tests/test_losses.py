"""Phase 4 损失：数值有限、可反传、双任务签名与解剖约束。"""
from __future__ import annotations

import pytest
import torch

from mgd.core.runtime.errors import ConfigError
from mgd.core.training.losses import LOSS_OPTIONS, get_loss_function, soft_skeletonize


@pytest.mark.parametrize("name", LOSS_OPTIONS)
def test_all_losses_finite_and_backward(name):
    logits = torch.randn(2, 2, 32, 64, requires_grad=True)
    gland = (torch.rand(2, 1, 32, 64) > 0.6).float()
    eyelid = (torch.rand(2, 1, 32, 64) > 0.3).float()
    criterion = get_loss_function(name, dual_task=True)
    loss = criterion(logits, gland, eyelid)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_anatomical_joint_requires_dual_task():
    with pytest.raises(ConfigError, match="双通道"):
        get_loss_function("AnatomicalJointLoss (解剖联合)", dual_task=False)


def test_anatomical_out_of_roi_penalty_increases_loss():
    criterion = get_loss_function("AnatomicalJointLoss (解剖联合)", dual_task=True)
    gland = torch.zeros(1, 1, 16, 16)
    eyelid = torch.ones(1, 1, 16, 16)
    inside = torch.full((1, 2, 16, 16), -6.0)
    outside = torch.full((1, 2, 16, 16), 6.0)
    assert criterion(outside, gland, eyelid) > criterion(inside, gland, eyelid)


def test_dual_task_loss_uses_eyelid_weight():
    base = get_loss_function("Dice Loss", dual_task=False)
    logits = torch.zeros(1, 2, 8, 8)
    gland = torch.zeros(1, 1, 8, 8)
    eyelid = torch.ones(1, 1, 8, 8)
    wrapped = get_loss_function("Dice Loss", dual_task=True, eyelid_weight=0.5)
    gland_only = base(logits[:, 0:1], gland)
    eyelid_only = base(logits[:, 1:2], eyelid)
    expected = float(gland_only + 0.5 * eyelid_only)
    assert float(wrapped(logits, gland, eyelid)) == pytest.approx(expected, rel=1e-5)


def test_soft_skeletonize_is_smaller_than_input():
    img = torch.zeros(1, 1, 32, 32)
    img[:, :, 4:28, 14:18] = 1.0
    skeleton = soft_skeletonize(img, iterations=4)
    assert skeleton.shape == img.shape
    assert float(skeleton.sum()) <= float(img.sum()) + 1e-3
