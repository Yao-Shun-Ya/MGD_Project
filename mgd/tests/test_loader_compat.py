"""权重通道探测与旧版单通道权重兼容加载。"""
from __future__ import annotations

from mgd.core.models.loader import _build_matching_model, detect_out_channels
from mgd.core.models.registry import create_model


def test_detect_single_channel_legacy_weight():
    state = create_model("UNet", out_channels=1).state_dict()
    assert detect_out_channels(state) == 1
    model = _build_matching_model("UNet", state)
    assert int(model.out_channels) == 1


def test_detect_dual_channel_weight():
    state = create_model("AttentionUNet", out_channels=2).state_dict()
    assert detect_out_channels(state) == 2
    model = _build_matching_model("AttentionUNet", state)
    assert int(model.out_channels) == 2


def test_mgd_prior_net_is_always_dual():
    state = create_model("MGD-PriorNet").state_dict()
    assert detect_out_channels(state) == 2
    model = _build_matching_model("MGD-PriorNet", state)
    assert int(model.out_channels) == 2
