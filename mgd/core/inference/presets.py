"""推理预设。读写均使用绝对路径 + 原子替换 + UTF-8。"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Settings, get_settings
from ..runtime.errors import MGDError

BUILTIN_PRESETS: dict[str, dict] = {
    "平衡-默认": {
        "tile_infer": True,
        "tile_size": 384,
        "tile_overlap": 0.25,
        "threshold": 0.70,
        "min_area": 120,
        "smooth_on": True,
        "smooth_sigma": 1.2,
        "roi_only": True,
        "tta": "关闭",
    },
    "保守-低误检": {
        "tile_infer": True,
        "tile_size": 384,
        "tile_overlap": 0.35,
        "threshold": 0.80,
        "min_area": 180,
        "smooth_on": True,
        "smooth_sigma": 1.0,
        "roi_only": True,
        "tta": "关闭",
    },
    "敏感-高召回": {
        "tile_infer": True,
        "tile_size": 384,
        "tile_overlap": 0.30,
        "threshold": 0.60,
        "min_area": 80,
        "smooth_on": True,
        "smooth_sigma": 1.3,
        "roi_only": True,
        "tta": "关闭",
    },
}

_ALLOWED_KEYS = {
    "tile_infer", "tile_size", "tile_overlap", "threshold", "min_area",
    "smooth_on", "smooth_sigma", "roi_only", "tta", "auto_escalate_threshold",
}


def _sanitize(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key in _ALLOWED_KEYS}


def load_presets(settings: Settings | None = None) -> dict[str, dict]:
    """内置预设 + 用户预设（用户同名覆盖内置）。"""
    settings = settings or get_settings()
    presets = {name: dict(values) for name, values in BUILTIN_PRESETS.items()}
    path = settings.preset_file
    if path.is_file():
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MGDError(f"预设文件损坏：{path}", hint=str(exc)) from exc
        if isinstance(user, dict):
            for name, values in user.items():
                if isinstance(values, dict):
                    presets[str(name)] = _sanitize(values)
    return presets


def save_preset(name: str, values: dict, settings: Settings | None = None) -> Path:
    name = str(name).strip()
    if not name:
        raise MGDError("预设名称不能为空")
    settings = settings or get_settings()
    presets = load_presets(settings)
    presets[name] = _sanitize(values)

    path = settings.preset_file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
