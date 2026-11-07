"""配置与版本单一来源。"""
from __future__ import annotations

from pathlib import Path

import pytest

import mgd
from mgd.core.config import PACKAGE_DIR, REPO_ROOT, Settings, checkpoint_path, model_best_path
from mgd.core.runtime.errors import ConfigError


def test_version_is_single_source():
    assert mgd.__version__ == "2.3.0"
    pyproject = (PACKAGE_DIR.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = { attr = "mgd.__version__" }' in pyproject


def test_paths_anchored_to_code_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.home == REPO_ROOT
    assert settings.data_dir == REPO_ROOT / "data"
    assert settings.model_dir == REPO_ROOT / "models_weights"
    assert settings.log_file.is_absolute()


def test_relative_overrides_are_resolved_against_home(tmp_path):
    settings = Settings(home=tmp_path, data_dir="dataset")
    assert settings.data_dir == (tmp_path / "dataset").resolve()


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("MGD_HOME", str(tmp_path))
    monkeypatch.setenv("MGD_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MGD_PORT", "9000")
    settings = Settings.from_env()
    assert settings.home == tmp_path.resolve()
    assert settings.data_dir == (tmp_path / "d").resolve()
    assert settings.port == 9000


def test_invalid_clahe_raises(tmp_path):
    with pytest.raises(ConfigError):
        Settings(home=tmp_path, clahe_clip_limit=0)


def test_weight_paths_are_absolute_and_not_created(tmp_path):
    settings = Settings(home=tmp_path)
    best = model_best_path("UNet", settings)
    ckpt = checkpoint_path("UNet", 3, settings)
    assert best.is_absolute() and ckpt.is_absolute()
    assert not best.parent.exists()


def test_default_host_is_loopback(tmp_path, monkeypatch):
    monkeypatch.delenv("MGD_HOST", raising=False)
    assert Settings(home=tmp_path).host == "127.0.0.1"


def test_target_size_property(tmp_path):
    settings = Settings(home=tmp_path, target_height=640, target_width=1280)
    assert settings.target_size == (640, 1280)
    assert isinstance(settings.home, Path)
