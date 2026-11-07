"""统一日志。禁止再用 print 做业务日志。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "mgd"
_CONFIGURED = False


def configure_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """配置命名 logger（幂等）。日志同时写 stdout 与文件（UTF-8）。"""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 mgd 命名空间下的子 logger。"""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def reset_logging() -> None:
    """仅测试使用：卸载所有 handler，允许重新配置。"""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    _CONFIGURED = False
