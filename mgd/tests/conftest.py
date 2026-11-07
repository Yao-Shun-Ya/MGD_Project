"""测试公共夹具。全部离线运行：合成数据 + CPU + 无外部网络。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from mgd.core.config import Settings, reset_settings  # noqa: E402
from mgd.core.runtime.logging import reset_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_process_state():
    reset_settings()
    reset_logging()
    yield
    reset_settings()
    reset_logging()


@pytest.fixture
def settings(tmp_path):
    """小尺寸配置（64x128），让 CPU 测试保持可接受的耗时。"""
    return Settings(home=tmp_path, target_height=64, target_width=128, num_workers=0)


def _save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def write_synthetic_dataset(
    settings: Settings,
    stems: tuple[str, ...] = ("s1", "s2", "s3"),
    *,
    seed: int = 0,
) -> list[str]:
    """写入影像 + 腺体标签 + 眼睑标签，返回 stem 列表。"""
    rng = np.random.default_rng(seed)
    height, width = settings.target_size
    for offset, stem in enumerate(stems):
        image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        _save_gray(settings.images_dir / f"{stem}.jpg", image)

        gland = np.zeros((height, width), dtype=np.uint8)
        top = 8 + (offset % 3) * 4
        gland[top:top + 16, 20:60] = 255
        _save_gray(settings.gland_masks_dir / f"{stem}.png", gland)

        eyelid = np.full((height, width), 255, dtype=np.uint8)
        _save_gray(settings.eyelid_masks_dir / f"{stem}.png", eyelid)
    return list(stems)


@pytest.fixture
def dataset_builder(settings):
    def build(stems: tuple[str, ...] = ("s1", "s2", "s3"), seed: int = 0) -> Settings:
        write_synthetic_dataset(settings, stems=stems, seed=seed)
        return settings

    return build
