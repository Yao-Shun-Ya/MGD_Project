"""Linux 交付用的交互式训练菜单（仍为薄壳，只负责提问与调用 core）。"""
from __future__ import annotations

import sys

from ...core.config import get_settings
from ...core.data.pairing import scan_dataset
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging
from ...core.training import TrainConfig, train_iter
from .train_cli import main as cli_main


def _ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def run_menu() -> int:
    settings = get_settings()
    configure_logging(log_file=settings.log_file)
    try:
        index = scan_dataset(settings)
    except MGDError as exc:
        print(exc.user_message())
        print("\n修复数据集后可重新运行；也可以直接使用 `python -m mgd.adapters.cli.train_cli --help`。")
        return 2

    print("=" * 56)
    print(" 腺而易见 · MGD 训练控制台")
    print("=" * 56)
    print(index.summary())

    model = _ask("模型 (MGD-Former/UNet/AttentionUNet/WRCANet/MGD-PriorNet)", "MGD-Former")
    epochs = _ask("总轮数", "20")
    batch_size = _ask("batch_size", "1")
    lr = _ask("学习率", "1e-4")
    val_ratio = _ask("验证集比例", "0.15")
    test_ratio = _ask("测试集比例", "0.15")
    split_mode = _ask("划分模式 (holdout/kfold)", "holdout")
    tile = _ask("分块训练 (y/n)", "y")

    # 全部走 --set（schema 唯一来源）：学习率/划分比例等参数在 train_cli 的 parser 里
    # 没有快捷旗标，旧实现直接传旧快捷旗标必然 argparse 报错；--set 在档位之上再落一次，
    # 不会被显存档吞掉。
    argv = [
        "--model", model,
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--set", f"train.lr={lr}",
        "--set", f"data.val_ratio={val_ratio}",
        "--set", f"data.test_ratio={test_ratio}",
        "--set", f"data.split_mode={split_mode}",
    ]
    if tile.lower().startswith("n"):
        argv.extend(["--set", "data.tile_mode=false"])
    return cli_main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(run_menu())
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
        raise SystemExit(130)
