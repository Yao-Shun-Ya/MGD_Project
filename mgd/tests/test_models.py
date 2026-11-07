"""模型注册与架构。"""
from __future__ import annotations

import pytest
import torch

from mgd.core.models import MODEL_NAMES, create_model, is_supported
from mgd.core.models.loader import arch_from_state_dict
from mgd.core.models.registry import normalize_model_name, unwrap_outputs
from mgd.core.runtime.errors import ConfigError


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape(name):
    model = create_model(name).eval()
    expected_channels = 2 if name == "MGD-PriorNet" else 1
    with torch.no_grad():
        output, _ = unwrap_outputs(model(torch.zeros(1, 1, 64, 128)))
    assert output.shape == (1, expected_channels, 64, 128)


@pytest.mark.parametrize("name", [item for item in MODEL_NAMES if item != "MGD-PriorNet"])
def test_dual_task_output_shape(name):
    model = create_model(name, out_channels=2).eval()
    with torch.no_grad():
        output, _ = unwrap_outputs(model(torch.zeros(1, 1, 64, 128)))
    assert output.shape == (1, 2, 64, 128)


def test_mgd_former_deep_supervision_aux_scales():
    model = create_model("MGD-Former", out_channels=2, deep_supervision=True).eval()
    with torch.no_grad():
        logits, aux = unwrap_outputs(model(torch.zeros(1, 1, 128, 256)))
    assert logits.shape == (1, 2, 128, 256)
    assert [tuple(item.shape[-2:]) for item in aux] == [(32, 64), (16, 32), (8, 16)]


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("UNet", "UNet"),
        ("unet", "UNet"),
        ("U-Net", "UNet"),
        ("AttentionUNet", "AttentionUNet"),
        ("attention_unet", "AttentionUNet"),
        ("Attention UNet", "AttentionUNet"),
        ("ATTENTION-UNET", "AttentionUNet"),
        ("WRCANet", "WRCANet"),
        ("wrcanet_for_segmentation", "WRCANet"),
    ],
)
def test_name_normalization(alias, expected):
    assert normalize_model_name(alias) == expected
    assert is_supported(alias)


def test_unknown_name_raises_with_hint():
    with pytest.raises(ConfigError) as exc_info:
        normalize_model_name("TransformerXYZ")
    assert "可选模型" in exc_info.value.user_message()
    assert not is_supported("TransformerXYZ")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_arch_detection_from_state_dict(name):
    state = create_model(name).state_dict()
    assert arch_from_state_dict(state) == name
