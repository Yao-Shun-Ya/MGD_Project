"""配置与版本单一来源。"""
from __future__ import annotations

from pathlib import Path

import pytest

import mgd
from mgd.core.config import PACKAGE_DIR, REPO_ROOT, Settings, checkpoint_path, model_best_path
from mgd.core.runtime.errors import ConfigError


def test_version_is_single_source():
    # v2.5.0（接线增量）升版；单一来源契约不变：pyproject 动态读取 mgd.__version__。
    # 不在此硬编码期望版本号——版本与文档的一致性由 g10 与
    # test_version_matches_docs 交叉校验，避免"升版要改三处、漏一处"。
    parts = mgd.__version__.split(".")
    assert len(parts) == 3 and all(part.isdigit() for part in parts), mgd.__version__
    pyproject = (PACKAGE_DIR.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = { attr = "mgd.__version__" }' in pyproject


def test_version_matches_docs():
    """版本唯一源必须与文档声称的当前版本一致（防止文档先行升级）。

    仓库 README 与交付包 README 结构不同：交付包用 "mgd 引擎 vX.Y.Z" 标注引擎版本，
    出现该标注时必须与唯一源相等；FREEZE_BASELINE 为跨包同步的权威基线。
    """
    import re

    version = mgd.__version__
    root = PACKAGE_DIR.parent
    freeze = (root / "docs/engineering/FREEZE_BASELINE.md").read_text(encoding="utf-8")
    assert version in freeze, f"FREEZE_BASELINE.md 未出现当前版本 {version}"

    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    match = re.search(r"mgd 引擎 v(\d+\.\d+\.\d+)", readme)
    if match:
        assert match.group(1) == version, f"README 标注的引擎版本 {match.group(1)} != {version}"


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
