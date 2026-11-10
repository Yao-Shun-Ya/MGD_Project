"""设备探测。这是各平台唯一允许存在差异的一层。

交付范围固定为 Windows WebUI 与 Linux 训练 CLI，运行时只支持 cuda > cpu；
显式指定但不可用时直接报错，不做静默回退。
"""
from __future__ import annotations

import os

import torch

from .errors import DeviceError

_VALID = ("cuda", "cpu")


def resolve_device(override: str | None = None) -> torch.device:
    """解析运行设备。

    override 为空时按 cuda > cpu 顺序探测；环境变量 MGD_DEVICE 优先于自动探测。
    支持 cuda / cpu / cuda:N（N 必须小于可见卡数）；macOS/MPS 不在交付范围。
    """
    requested = (override or os.environ.get("MGD_DEVICE") or "").strip().lower()
    if requested:
        base = requested.split(":", 1)[0]
        if base not in _VALID or (base == "cuda" and ":" in requested
                                  and not requested.split(":", 1)[1].isdigit()):
            raise DeviceError(
                f"未知设备 {requested!r}",
                hint="可选值：cuda / cpu / cuda:N（N 为显卡序号）",
            )
        if base == "cuda":
            if not torch.cuda.is_available():
                raise DeviceError("显式指定 CUDA，但当前环境不可用", hint="改用 MGD_DEVICE=cpu 或安装 CUDA 版 torch")
            if ":" in requested:
                index = int(requested.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    raise DeviceError(
                        f"显卡序号 {index} 不存在（当前 {torch.cuda.device_count()} 张可见）",
                        hint="用 nvidia-smi 确认卡号，或直接写 cuda",
                    )
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """给 UI / 日志用的可读描述。"""
    if device.type == "cuda":
        index = device.index or 0
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:  # 驱动异常时不应阻断主流程
            name = "CUDA"
        return f"CUDA:{index} ({name})"
    return "CPU"


def empty_cache(device: torch.device) -> None:
    """按设备类型释放缓存显存；CPU 下为空操作。"""
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # 缓存清理失败不影响业务结果
        pass
