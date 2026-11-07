"""ModelService：权重缓存、release、pending 拒绝加载。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mgd.core.config import model_best_path
from mgd.core.inference.service import InferenceOptions, ModelService
from mgd.core.models.manifest import build_manifest, manifest_path_for, write_manifest
from mgd.core.models.registry import create_model
from mgd.core.runtime.errors import ManifestError


def _trained_weight(settings, arch: str = "WRCANet"):
    path = model_best_path(arch, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = create_model(arch, in_channels=1, out_channels=1)
    torch.save(model.state_dict(), path)
    manifest = build_manifest(path, arch, "2.0.0", verified=True, source="test", settings=settings)
    write_manifest(manifest, path)
    return path


def test_cache_reuses_model_and_release_clears(settings):
    _trained_weight(settings)
    service = ModelService(settings, device="cpu")
    first = service.load("WRCANet")
    second = service.load("WRCANet")
    assert first is second
    assert len(service.cache_info()) == 1

    service.release()
    assert service.cache_info() == []


def test_predict_returns_mask_and_threshold_rows(settings):
    # WRCANet 在 64x128 上 CPU 前向较慢；缩到 32x64 仍覆盖完整管线。
    small = settings.with_overrides(target_height=32, target_width=64)
    _trained_weight(small)
    service = ModelService(small, device="cpu")
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=small.target_size, dtype=np.uint8)

    result = service.predict(
        image,
        "WRCANet",
        InferenceOptions(tile_infer=False, roi_only=False, smooth_on=False, threshold=0.5),
    )
    assert result.binary_mask.shape == small.target_size
    assert result.prob_map.shape == small.target_size
    assert result.overlay.shape == (*small.target_size, 3)
    labels = {row[0]: row[1] for row in result.metrics}
    assert labels["阈值(用户设定)"] == "0.50"
    assert labels["阈值(实际应用)"] == "0.50"
    assert labels["阈值自动抬升"] == "否"
    service.release()


def test_pending_weight_rejected_but_opt_in_allowed(settings):
    path = _trained_weight(settings)
    manifest_path_for(path).unlink()

    service = ModelService(settings, device="cpu")
    with pytest.raises(ManifestError, match="pending"):
        service.load("WRCANet")

    permissive = ModelService(settings, device="cpu", allow_unverified=True)
    model = permissive.load("WRCANet")
    assert isinstance(model, torch.nn.Module)
    permissive.release()


def test_roi_fallback_when_roi_wipes_predictions():
    import numpy as np

    from mgd.core.inference.service import apply_roi_with_fallback

    mask = np.zeros((8, 8), np.float32)
    mask[0:4, :] = 1.0
    prob = mask.copy()
    empty_roi = np.zeros((8, 8), np.float32)
    full_roi = np.ones((8, 8), np.float32)

    kept_mask, kept_prob, fallback = apply_roi_with_fallback(mask, prob, empty_roi)
    assert fallback is True
    assert kept_mask.sum() == mask.sum()

    roi_mask, _, fallback_ok = apply_roi_with_fallback(mask, prob, full_roi)
    assert fallback_ok is False
    assert roi_mask.sum() == mask.sum()
