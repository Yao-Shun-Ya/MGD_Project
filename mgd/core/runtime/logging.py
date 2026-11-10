"""统一日志 v2：结构化上下文 + 三路输出（控制台 / mgd_events.jsonl / errors.jsonl）。

设计目标：任何问题都能一条 grep 立刻定位——
- 控制台给人读：``2026-11-10 12:00:00 | ERROR | mgd.inference | 消息 [run=.. case=.. stage=..]``
- ``<log_file 目录>/mgd_events.jsonl`` 机器全量（严格 JSON，allow_nan=False）
- ``<log_file 目录>/errors.jsonl`` 快速分诊（仅 WARNING 及以上，同结构）

公开签名与 v1 兼容：configure_logging(level=..., log_file=...) 幂等、
get_logger(name)、reset_logging()；上下文绑定用 contextvars（线程/异步安全）。
禁止再用 print 做业务日志。
"""
from __future__ import annotations

import contextvars
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .errors import MGDError

_LOGGER_NAME = "mgd"
_CONFIGURED = False

#: JSONL 轮转参数（模块级常量，测试可 monkeypatch）
_ROTATE_MAX_BYTES = 20 * 1024 * 1024
_ROTATE_BACKUP_COUNT = 10

#: JSONL 产物文件名（log_file 所在目录下）
EVENTS_FILENAME = "mgd_events.jsonl"
ERRORS_FILENAME = "errors.jsonl"

#: 上下文核心字段；其余 bind_context 关键字进入 extra
_CORE_FIELDS = ("run_id", "case_id", "stage", "arch", "scope")

#: 控制台后缀用的短名（JSON 里仍是全名）
_CONSOLE_SHORT_NAMES = {"run_id": "run", "case_id": "case"}


@dataclass(frozen=True)
class LogContext:
    """当前日志上下文（不可变；bind_context 生成合并后的新实例）。"""

    run_id: str | None = None
    case_id: str | None = None
    stage: str | None = None
    arch: str | None = None
    scope: str | None = None
    extra: tuple[tuple[str, str], ...] = ()


_CONTEXT_VAR: contextvars.ContextVar[LogContext] = contextvars.ContextVar(
    "mgd_log_context", default=LogContext()
)


def get_log_context() -> LogContext:
    """读取当前上下文（线程/异步各自隔离）。"""
    return _CONTEXT_VAR.get()


def bind_context(**fields) -> None:
    """合并式绑定：只更新给定字段（run_id/case_id/stage/arch/scope + 任意自定义键进 extra）。"""
    current = get_log_context()
    values = {name: getattr(current, name) for name in _CORE_FIELDS}
    extra = dict(current.extra)
    for key, value in fields.items():
        if value is None:
            continue  # None 表示"未给定"，不覆盖已有值
        if key in _CORE_FIELDS:
            values[key] = str(value)
        else:
            extra[str(key)] = str(value)
    _CONTEXT_VAR.set(LogContext(extra=tuple(extra.items()), **values))


def clear_context(keys: list[str] | None = None) -> None:
    """清理上下文；None = 全清，给定 keys 则只清指定字段（含 extra 里的同名键）。"""
    if keys is None:
        _CONTEXT_VAR.set(LogContext())
        return
    current = get_log_context()
    drop = {str(key) for key in keys}
    values = {name: (None if name in drop else getattr(current, name)) for name in _CORE_FIELDS}
    extra = tuple(item for item in current.extra if item[0] not in drop)
    _CONTEXT_VAR.set(LogContext(extra=extra, **values))


def _context_pairs(ctx: LogContext) -> list[tuple[str, str]]:
    """控制台/检索用：核心字段（短名）+ extra，保持绑定顺序。"""
    pairs: list[tuple[str, str]] = []
    for name in _CORE_FIELDS:
        value = getattr(ctx, name)
        if value:
            pairs.append((_CONSOLE_SHORT_NAMES.get(name, name), value))
    for key, value in ctx.extra:
        pairs.append((key, value))
    return pairs


def _json_safe(value):
    """递归 sanitize：非有限 float → None；容器递归；其余转字符串（g04 门禁纪律）。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_mgd_error(record: logging.LogRecord) -> MGDError | None:
    """从记录中识别 MGDError（exc_info 的异常 / msg 本身 / args 里的实例）。

    不改变调用方的日志习惯：``logger.error(exc_info=True)``、
    ``logger.warning("...：%s", exc)``、``logger.error(exc)`` 都能命中。
    """
    if record.exc_info is not None and isinstance(record.exc_info[1], MGDError):
        return record.exc_info[1]
    if isinstance(record.msg, MGDError):
        return record.msg
    for arg in record.args or ():
        if isinstance(arg, MGDError):
            return arg
    return None


class ConsoleFormatter(logging.Formatter):
    """人读格式；上下文非空才追加 ``[…]`` 后缀，WARNING/ERROR 附加 exc 摘要最后一行。"""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        line = " | ".join(
            (
                self.formatTime(record, self.datefmt),
                record.levelname,
                record.name,
                message,
            )
        )
        pairs = _context_pairs(get_log_context())
        if pairs:
            line += " [" + " ".join(f"{key}={value}" for key, value in pairs) + "]"
        if record.exc_info is not None and record.levelno >= logging.WARNING:
            try:
                text = self.formatException(record.exc_info)
            except Exception:
                text = ""
            stripped = text.strip()
            if stripped:
                line += " | " + stripped.splitlines()[-1]
        return line


class JsonlFormatter(logging.Formatter):
    """机器格式：每行严格 JSON（allow_nan=False，非有限值已 sanitize 成 None）。"""

    def format(self, record: logging.LogRecord) -> str:
        ctx = get_log_context()
        ctx_payload: dict = {}
        for name in _CORE_FIELDS:
            value = getattr(ctx, name)
            if value:
                ctx_payload[name] = value
        for key, value in ctx.extra:
            ctx_payload.setdefault(key, value)
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "module": Path(record.pathname).name if record.pathname else f"{record.module}.py",
            "func": record.funcName,
            "line": record.lineno,
            "msg": message,
            "ctx": ctx_payload,
        }
        error = _extract_mgd_error(record)
        if error is not None:
            payload["error"] = {"type": type(error).__name__, "hint": error.hint}
        if record.exc_info is not None:
            try:
                payload["exc"] = self.formatException(record.exc_info)
            except Exception:
                payload["exc"] = None
        return json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False)


def configure_logging(
    level: str | int | None = "INFO", log_file: str | Path | None = None
) -> logging.Logger:
    """配置命名 logger（幂等，首次生效）。三路输出一次就位：stdout + events + errors。"""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED:
        return logger

    logger.setLevel(level if level is not None else "INFO")

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(ConsoleFormatter())
    logger.addHandler(stream)

    if log_file is not None:
        log_dir = Path(log_file).expanduser().resolve().parent
        log_dir.mkdir(parents=True, exist_ok=True)
        events = RotatingFileHandler(
            log_dir / EVENTS_FILENAME,
            maxBytes=_ROTATE_MAX_BYTES,
            backupCount=_ROTATE_BACKUP_COUNT,
            encoding="utf-8",
        )
        events.setFormatter(JsonlFormatter())
        logger.addHandler(events)

        errors = RotatingFileHandler(
            log_dir / ERRORS_FILENAME,
            maxBytes=_ROTATE_MAX_BYTES,
            backupCount=_ROTATE_BACKUP_COUNT,
            encoding="utf-8",
        )
        errors.setLevel(logging.WARNING)
        errors.setFormatter(JsonlFormatter())
        logger.addHandler(errors)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 mgd 命名空间下的子 logger。"""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def reset_logging() -> None:
    """仅测试使用：卸载所有 handler 并清空日志上下文，允许重新配置。"""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    _CONTEXT_VAR.set(LogContext())
    _CONFIGURED = False
