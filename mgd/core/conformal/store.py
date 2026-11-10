"""共形校准参数的持久化：版本化 JSON + 原子写（项目统一标准）。

- 文件：``<directory>/<name>.json``；``directory`` 默认
  ``get_settings().calibration_dir``（output_dir/"calibration"）。
- 写入前自动补外层包装 ``{"version": 1, "created_at": UTC ISO, "name": name}``。
- 原子写：先写 ``<name>.json.tmp`` 再 :func:`os.replace`，断电 / 中断不会
  留下半截文件（与 splits / workflow 状态写盘同一标准）。
- 读取：文件缺失 → :class:`ConfigError`；``version`` 大于当前支持 →
  :class:`ConfigError` 并提示升级；目录不存在时 :func:`save_calibration`
  自动创建。
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..runtime.errors import ConfigError
from ..runtime.logging import get_logger
from .calibrator import ConformalCalibrator

logger = get_logger("conformal.store")

#: 当前支持的校准文件格式版本
SUPPORTED_STORE_VERSION = 1

#: 校准名称 = JSON 文件名（防路径穿越：只允许安全文件名字符）
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_KIND = "conformal_calibrator"


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise ValueError(
            f"校准名称非法：{name!r}"
            "（只允许字母/数字/点/下划线/连字符，且以字母或数字开头）"
        )
    return name


def _resolve_directory(directory: Path | str | None) -> Path:
    if directory is None:
        directory = get_settings().calibration_dir
    return Path(directory)


def _json_default(obj):
    """numpy 标量 → Python 标量；其余类型拒绝序列化（不允许静默丢数据）。"""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"无法序列化到校准 JSON：{type(obj).__name__}（{obj!r}）")


def save_calibration(name: str, payload: dict, directory: Path | None = None) -> Path:
    """保存校准参数（原子写），返回写入的 JSON 路径。

    外层 ``version`` / ``created_at`` / ``name`` 由本函数统一写入并覆盖，
    保证 store 元数据可信；目录不存在时自动创建（含父目录）。

    :raises ValueError: ``name`` 非法、payload 无法 JSON 序列化，或 payload
        含 NaN/Inf（``allow_nan=False``：非有限值写进 JSON 会成为非法文档，
        必须在写盘前拦截，而不是等读取端才发现）。
    """
    _validate_name(name)
    target_dir = _resolve_directory(directory)
    record = dict(payload) if payload else {}
    record["version"] = SUPPORTED_STORE_VERSION
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    record["name"] = name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.debug("已保存校准参数：%s", path)
    return path


def load_calibration(name: str, directory: Path | None = None) -> dict:
    """读取校准参数；缺失 / 损坏 / 版本过新均抛 :class:`ConfigError`。

    ``version`` 缺失视为 1（兼容手写 payload）。
    """
    _validate_name(name)
    target_dir = _resolve_directory(directory)
    path = target_dir / f"{name}.json"
    if not path.is_file():
        raise ConfigError(
            f"校准文件不存在：{path}",
            hint="请先运行校准流程（save_calibration / calibrator_to_store）生成该文件",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"校准文件损坏或不可读：{path}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"校准文件顶层必须是 JSON 对象：{path}")
    version = payload.get("version", SUPPORTED_STORE_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigError(f"校准文件 version 字段非法：{version!r}（{path}）")
    if version > SUPPORTED_STORE_VERSION:
        raise ConfigError(
            f"校准文件版本过新：version={version} > 当前支持 {SUPPORTED_STORE_VERSION}（{path}）",
            hint="请升级 mgd 运行环境至与新版本校准文件兼容的发行版后再读取，"
                 "或用生成它的同版本工具链处理",
        )
    return payload


def calibrator_to_store(calibrator: ConformalCalibrator) -> dict:
    """把 :class:`ConformalCalibrator` 打包成带外层 ``version`` 包装的可存储 payload。

    前置校验：``scores`` 必须全为有限数值（scores 被外部直接改出 NaN/inf 时
    在保存端拦截，而不是等读取端 :meth:`ConformalCalibrator.from_dict` 才发现）。
    """
    to_dict = getattr(calibrator, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            f"calibrator_to_store 需要 ConformalCalibrator，收到 {type(calibrator).__name__}"
        )
    inner = to_dict()
    scores = inner.get("scores", ()) if isinstance(inner, dict) else ()
    for raw in scores:
        if not math.isfinite(float(raw)):
            raise ValueError(
                f"校准 scores 含 NaN/inf（{raw!r}）：拒绝保存，请先清洗校准数据"
                "（剔除含 NaN/inf 的样本后重新 fit）"
            )
    return {
        "version": SUPPORTED_STORE_VERSION,
        "kind": _KIND,
        "payload": inner,
    }


def calibrator_from_store(payload: dict) -> ConformalCalibrator:
    """从 :func:`save_calibration` 的外层包装中还原 :class:`ConformalCalibrator`。

    兼容扁平结构（``alpha`` / ``scores`` 直接位于外层）；``version`` 过新
    抛 :class:`ConfigError`（与 :func:`load_calibration` 同口径）。
    """
    if not isinstance(payload, dict):
        raise ConfigError(f"校准 payload 必须是 dict，收到 {type(payload).__name__}")
    version = payload.get("version", SUPPORTED_STORE_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigError(f"校准 payload version 字段非法：{version!r}")
    if version > SUPPORTED_STORE_VERSION:
        raise ConfigError(
            f"校准 payload 版本过新：version={version} > 当前支持 {SUPPORTED_STORE_VERSION}",
            hint="请升级 mgd 运行环境后再读取，或用生成它的同版本工具链处理",
        )
    kind = payload.get("kind")
    if kind is not None and kind != _KIND:
        raise ConfigError(f"校准 payload kind 非法：{kind!r}（期望 {_KIND!r}）")
    inner = payload.get("payload")
    if isinstance(inner, dict):
        return ConformalCalibrator.from_dict(inner)
    if "alpha" in payload and "scores" in payload:
        return ConformalCalibrator.from_dict(
            {"alpha": payload["alpha"], "scores": payload["scores"]}
        )
    raise ConfigError(
        "校准 payload 缺少 'payload' 字段（且不是扁平的 alpha/scores 结构）",
        hint="请用 calibrator_to_store 生成存储 payload",
    )
