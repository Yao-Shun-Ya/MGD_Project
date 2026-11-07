"""训练配置文件的唯一入口：默认 YAML → 用户 YAML → 预设 → CLI 覆盖 → TrainConfig。

schema 本身在 specs.py（唯一来源）；本模块只负责合并与快照。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from ..runtime.errors import ConfigError
from .engine import TrainConfig
from .specs import (
    SCHEMA,
    coerce_value,
    parse_set_expression,
    validate_payload,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_TRAIN_CONFIG = CONFIG_DIR / "train_default.yaml"
DEFAULT_INFERENCE_CONFIG = CONFIG_DIR / "inference_default.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}", hint="参考 mgd/configs/train_default.yaml")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败：{path}", hint=str(exc)) from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"配置文件顶层必须是映射：{path}")
    return payload


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolved_dict(
    path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    use_default: bool = True,
) -> dict[str, Any]:
    """返回合并后的原始字典。

    优先级（后者覆盖前者）：默认 YAML < 用户 YAML < 调用方 overrides < CLI --set。
    """
    base: dict[str, Any] = _read_yaml(DEFAULT_TRAIN_CONFIG) if use_default else {}
    if path is not None:
        base = _merge(base, _read_yaml(path))
    validate_payload(base)
    merged = _merge(base, overrides or {})
    for expression in cli_overrides or []:
        section, key, raw = parse_set_expression(expression)
        spec_value = coerce_value_section(section, key, raw)
        merged.setdefault(section, {})[key] = spec_value
    validate_payload(merged)
    return merged


def coerce_value_section(section: str, key: str, raw: Any) -> Any:
    """给 CLI/UI 复用：按 schema 把单个值转成正确类型。"""
    from .specs import SPEC_BY_NAME

    return coerce_value(SPEC_BY_NAME[f"{section}.{key}"], raw)


def build_train_config(
    path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    use_default: bool = True,
) -> tuple[TrainConfig, dict[str, Any]]:
    """返回 (TrainConfig, 元信息)。元信息含 output_dir、resolved 原始字典、来源路径。"""
    raw = resolved_dict(path, overrides, cli_overrides, use_default=use_default)
    kwargs: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    for section, mapping in SCHEMA.items():
        body = raw.get(section) or {}
        for key, target in mapping.items():
            if key not in body or body[key] is None:
                continue
            value = body[key]
            if target == "_output_dir":
                meta["output_dir"] = str(value)
                continue
            from .specs import SPEC_BY_NAME

            kwargs[target] = coerce_value(SPEC_BY_NAME[f"{section}.{key}"], value)
    train_config = TrainConfig(**kwargs)
    train_config.validate()
    meta["config_path"] = str(path or DEFAULT_TRAIN_CONFIG)
    meta["resolved"] = raw
    return train_config, meta


def snapshot_payload(config: TrainConfig, meta: dict[str, Any]) -> dict[str, Any]:
    """写入 run 目录的配置快照（YAML 原始值 + 生效值）。"""
    return {
        "resolved_yaml": meta.get("resolved", {}),
        "train_config": asdict(config),
        "config_path": meta.get("config_path"),
        "fields": [item.name for item in fields(TrainConfig)],
    }


def write_snapshot(path: str | Path, config: TrainConfig, meta: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(snapshot_payload(config, meta), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target
