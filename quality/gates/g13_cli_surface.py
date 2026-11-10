"""g13 CLI 门面：6 个 CLI 的 --help 退出码 0 且 <10s；未知 action 退出码非 0。"""
from __future__ import annotations

import subprocess
import sys
import time

import _common

CLIS = [
    ("mgd-qc", "mgd.adapters.cli.qc_cli"),
    ("mgd-atlas", "mgd.adapters.cli.atlas_cli"),
    ("mgd-followup", "mgd.adapters.cli.followup_cli"),
    ("mgd-report", "mgd.adapters.cli.report_cli"),
    ("mgd-train", "mgd.adapters.cli.train_cli"),
    ("mgd-data", "mgd.adapters.cli.data_cli"),
]
HELP_BUDGET_S = 10.0


def run() -> dict:
    gate = _common.Gate("g13_cli_surface")
    for display, module in CLIS:
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=str(_common.PROJECT_ROOT), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=HELP_BUDGET_S + 5,
            )
            elapsed = time.perf_counter() - started
        except subprocess.TimeoutExpired:
            gate.check(False, f"{display} --help 在 {HELP_BUDGET_S + 5}s 内返回（超时）",
                       "CLI 导入链过重或挂起；把重型 import 延迟到子命令内")
            continue
        gate.check(proc.returncode == 0,
                   f"{display} --help 退出码 0（实际 {proc.returncode}）",
                   (proc.stderr or proc.stdout or "").strip()[:200] + " → 修复 argparse 入口")
        gate.check(elapsed < HELP_BUDGET_S,
                   f"{display} --help 耗时 {elapsed:.1f}s < {HELP_BUDGET_S:.0f}s",
                   "导入链过重：延迟导入 core 重型模块（torch/monai）到执行路径")

    proc = subprocess.run(
        [sys.executable, "-m", "mgd.adapters.cli.atlas_cli", "nope"],
        cwd=str(_common.PROJECT_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    )
    gate.check(proc.returncode != 0,
               f"mgd-atlas nope（未知 action）退出码非 0（实际 {proc.returncode}）",
               "argparse choices 会以退出码 2 拒绝未知 action；若为 0 说明校验被绕过")
    gate.note("6 个 CLI 均以 python -m <module> 直调（venv 未装新入口 exe 时的等价路径）；"
              "帮助预算 10s/个；未知 action 走 argparse choices 拒绝")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)
