"""v2.3.0 回归测试：TTA 视图数、流水线状态容错、设备边界与默认架构。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mgd.core.inference.service import (
    TTA_2X,
    TTA_4X,
    TTA_8X,
    TTA_OFF,
    InferenceOptions,
    ModelService,
    normalize_tta,
)
from mgd.core.runtime.device import resolve_device
from mgd.core.runtime.errors import DeviceError
from mgd.core.workflow.pipeline import _load_state


class _Echo(torch.nn.Module):
    """只保留第一通道的恒等模型，用于验证 TTA 视图数量。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _service() -> SimpleNamespace:
    return SimpleNamespace(settings=SimpleNamespace(target_height=64, target_width=128))


def _views(tta: str) -> int:
    tensor = torch.rand(1, 1, 32, 64)
    output = ModelService._forward_probabilities(
        _service(), _Echo(), tensor, InferenceOptions(tile_infer=False, tta=tta)
    )
    return int(output.shape[0])


def test_tta_normalization_covers_legacy_strings():
    assert normalize_tta(TTA_OFF) == "off"
    assert normalize_tta("") == "off"
    assert normalize_tta(TTA_2X) == "2x"
    assert normalize_tta(TTA_4X) == "4x"
    assert normalize_tta("4x 翻转") == "4x"
    assert normalize_tta(TTA_8X) == "8x"
    assert normalize_tta("8倍增强") == "8x"
    assert normalize_tta("不存在的选项") == "off"


def test_tta_runs_declared_number_of_views():
    assert _views(TTA_OFF) == 1
    assert _views(TTA_2X) == 2
    assert _views(TTA_4X) == 4
    assert _views(TTA_8X) == 8


def test_pipeline_state_recovers_from_corrupt_file(tmp_path: Path):
    path = tmp_path / "pipeline_state.json"
    path.write_text("{not json", encoding="utf-8")
    state = _load_state(path)
    assert state == {"entries": {}, "runs": []}

    path.write_text(json.dumps(["unexpected"]), encoding="utf-8")
    assert _load_state(path) == {"entries": {}, "runs": []}

    path.write_text(json.dumps({"entries": None, "runs": None}), encoding="utf-8")
    state = _load_state(path)
    assert state["entries"] == {} and state["runs"] == []


def test_pipeline_state_keeps_valid_payload(tmp_path: Path):
    path = tmp_path / "pipeline_state.json"
    payload = {"entries": {"train::a": {"status": "finished"}}, "runs": [{"run_id": "x"}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_state(path) == payload


def test_device_rejects_mps_after_macos_removal():
    with pytest.raises(DeviceError):
        resolve_device("mps")
