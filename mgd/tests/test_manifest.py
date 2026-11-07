"""权重 manifest：sha256、口径校验、pending 不得默认加载。"""
from __future__ import annotations

import hashlib

import pytest

from mgd.core.config import CANONICAL_WEIGHT_NAME, Settings
from mgd.core.models.manifest import (
    NORMALIZATION_TAG,
    assert_loadable,
    build_manifest,
    manifest_path_for,
    read_manifest,
    sha256_file,
    validate_weight,
    write_manifest,
)
from mgd.core.runtime.errors import ManifestError


def _weights(tmp_path) -> tuple[Settings, "object"]:
    settings = Settings(home=tmp_path, target_height=64, target_width=128)
    path = settings.model_dir / "UNet" / "unet_best.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-weights-bytes")
    return settings, path


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "payload.bin"
    payload = b"a" * 1024 + b"b" * 17
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_manifest_roundtrip_and_verified(tmp_path):
    settings, weights = _weights(tmp_path)
    manifest = build_manifest(weights, "UNet", "2.0.0", verified=True, source="test", settings=settings)
    sidecar = write_manifest(manifest, weights)
    assert sidecar == manifest_path_for(weights)
    loaded = read_manifest(weights)
    assert loaded is not None
    assert loaded.sha256 == manifest.sha256
    assert loaded.input_size == [64, 128]
    assert loaded.normalization == NORMALIZATION_TAG

    status = validate_weight(weights, "UNet", settings=settings)
    assert status.status == "verified"
    assert_loadable(status)


def test_missing_manifest_is_pending(tmp_path):
    settings, weights = _weights(tmp_path)
    status = validate_weight(weights, "UNet", settings=settings)
    assert status.status == "pending"
    assert "manifest" in status.reason
    with pytest.raises(ManifestError, match="pending"):
        assert_loadable(status)


def test_sha_mismatch_is_pending(tmp_path):
    settings, weights = _weights(tmp_path)
    manifest = build_manifest(weights, "UNet", "2.0.0", verified=True, source="test", settings=settings)
    weights.write_bytes(b"tampered")
    status = validate_weight(weights, "UNet", settings=settings)
    assert status.status == "pending"
    assert "sha256" in status.reason


def test_arch_mismatch_is_pending(tmp_path):
    settings, weights = _weights(tmp_path)
    manifest = build_manifest(weights, "UNet", "2.0.0", verified=True, source="test", settings=settings)
    write_manifest(manifest, weights)
    status = validate_weight(weights, "WRCANet", settings=settings)
    assert status.status == "pending"
    assert "架构不符" in status.reason


def test_input_size_mismatch_is_pending(tmp_path):
    settings, weights = _weights(tmp_path)
    manifest = build_manifest(
        weights, "UNet", "2.0.0", verified=True, source="test",
        input_size=(640, 1280), settings=settings,
    )
    write_manifest(manifest, weights)
    status = validate_weight(weights, "UNet", settings=settings)
    assert status.status == "pending"
    assert "输入尺寸不符" in status.reason


def test_unverified_flag_is_pending(tmp_path):
    settings, weights = _weights(tmp_path)
    manifest = build_manifest(weights, "UNet", "2.0.0", verified=False, source="test", settings=settings)
    write_manifest(manifest, weights)
    status = validate_weight(weights, "UNet", settings=settings)
    assert status.status == "pending"
    assert "verified=false" in status.reason
    assert_loadable(status, allow_unverified=True)


def test_corrupt_manifest_is_pending_not_crash(tmp_path):
    settings, weights = _weights(tmp_path)
    manifest_path_for(weights).write_text("{not json", encoding="utf-8")
    status = validate_weight(weights, "UNet", settings=settings)
    assert status.status == "pending"
    assert "manifest 损坏" in status.reason


def test_missing_weight_file_is_missing(tmp_path):
    settings = Settings(home=tmp_path, target_height=64, target_width=128)
    status = validate_weight(settings.model_dir / "UNet" / "nope.pth", "UNet", settings=settings)
    assert status.status == "missing"
    with pytest.raises(ManifestError):
        assert_loadable(status)


def test_canonical_weight_without_manifest_stays_pending(tmp_path):
    """已拍板：canonical 权重在验证前只能是 pending，不允许假装接上。"""
    settings = Settings(home=tmp_path, target_height=640, target_width=1280)
    canonical = settings.home / CANONICAL_WEIGHT_NAME
    canonical.write_bytes(b"canonical")
    status = validate_weight(canonical, "UNet", settings=settings)
    assert status.status == "pending"
    with pytest.raises(ManifestError):
        assert_loadable(status)
