"""可复现性：固定 seed 与确定性开关。

复现条件（写进 RUNBOOK）：
1. 相同 seed、相同划分（data/dataset_splits.json 的 signature 不变）；
2. 相同 torch / CUDA / cuDNN 版本与相同 GPU 型号；
3. 相同 batch_size、分块参数与 AMP 开关；
4. 环境变量 CUBLAS_WORKSPACE_CONFIG=:4096:8（本模块会自动补默认值）。
即使全部满足，CUDA + AMP 下个别 kernel 仍可能非严格确定；需要严格复现时关闭 AMP。
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if not deterministic:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    try:
        # warn_only：AMP / 个别算子无法保证严格确定性时记录告警而不中断训练
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def rng_state() -> dict[str, Any]:
    """采集 Python / NumPy / torch / CUDA 的随机数状态，供断点续训精确恢复。"""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    """恢复随机数状态；返回是否成功（旧检查点可能没有该字段）。"""
    if not state:
        return False
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        return True
    except Exception:
        return False
