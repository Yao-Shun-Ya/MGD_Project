"""g01 全量语法编译：`python -m compileall -q mgd` 必须零错误（可导入性的底线）。"""
from __future__ import annotations

import subprocess
import sys

import _common


def run() -> dict:
    gate = _common.Gate("g01_syntax_compile")
    targets = list(_common.iter_py_files("mgd", exclude_tests=False))

    gate.check(len(targets) >= 80, f"mgd/ 下可编译源文件数 ≥ 80（实际 {len(targets)}）",
               "若为 0 说明项目根定位失败：检查 quality/gates/_common.py 的 PROJECT_ROOT")
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "mgd"],
        cwd=str(_common.PROJECT_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=80,
    )
    gate.check(proc.returncode == 0, "compileall -q mgd 退出码为 0（全部源文件可编译）",
               f"按报错文件修复语法：{(proc.stderr or proc.stdout or '').strip()[:400]}")
    noisy = (proc.stderr or "").strip()
    gate.check(not noisy, "compileall 静默模式无 stderr 输出（出现即有编译告警/错误）",
               f"检查输出：{noisy[:300]}")
    gate.note(f"扫描 mgd/ 源文件 {len(targets)} 个（含 tests），compileall 退出码 {proc.returncode}")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)
