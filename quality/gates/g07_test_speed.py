"""g07 测试速度门禁：v2.4.0 增量子包测试套件一条 pytest 命令全绿 + 墙钟 < 60 秒。"""
from __future__ import annotations

import re
import subprocess
import sys
import time

import _common

TEST_FILES = [
    "mgd/tests/test_quantification.py", "mgd/tests/test_grading.py",
    "mgd/tests/test_atlas.py", "mgd/tests/test_explain.py",
    "mgd/tests/test_conformal.py", "mgd/tests/test_qc.py",
    "mgd/tests/test_longitudinal.py", "mgd/tests/test_interop.py",
    "mgd/tests/test_feedback.py", "mgd/tests/test_governance.py",
    "mgd/tests/test_ui_build.py", "mgd/tests/test_cli_new.py",
    "mgd/tests/test_report_extended.py",
]
SPEED_BUDGET_S = 60.0


def run() -> dict:
    gate = _common.Gate("g07_test_speed")
    missing = [f for f in TEST_FILES if not (_common.PROJECT_ROOT / f).is_file()]
    gate.check(not missing, f"13 个目标测试文件全部存在（缺失 {missing}）",
               "文件被重命名时同步更新本门禁的 TEST_FILES 清单")

    command = [sys.executable, "-m", "pytest", *TEST_FILES, "-q", "--tb=no", "-p", "no:warnings"]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command, cwd=str(_common.PROJECT_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=85,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired:
        gate.check(False, f"pytest 在 85s 内完成（超时，预算 {SPEED_BUDGET_S:.0f}s）",
                   "测试套件出现挂起/慢化：逐文件二分定位慢测试；诊断中心级增量必须秒级")
        return gate.result("pytest 超时未返回")

    tail = (proc.stdout or "").strip().splitlines()
    summary_line = tail[-1] if tail else ""
    passed_match = re.search(r"(\d+) passed", summary_line)
    failed_match = re.search(r"(\d+) failed", summary_line)
    error_match = re.search(r"(\d+) error", summary_line)

    gate.check(proc.returncode == 0 and not failed_match and not error_match,
               f"v2.4.0 增量测试套件全部通过（pytest 摘要：{summary_line[:80]}）",
               f"修复失败用例后再交付；复现命令：{' '.join(command)}")
    gate.check(elapsed < SPEED_BUDGET_S,
               f"墙钟 {elapsed:.1f}s < {SPEED_BUDGET_S:.0f}s 预算",
               "超预算说明测试被塞进重训练/重 IO；用 tmp_path + 合成小数据，禁止真实权重")
    gate.note(f"{len(TEST_FILES)} 个测试文件；pytest 摘要：{summary_line[:100]}")
    return gate.result(f"pytest 墙钟 {elapsed:.1f}s（预算 {SPEED_BUDGET_S:.0f}s），退出码 {proc.returncode}")


if __name__ == "__main__":
    _common.run_as_main(run)
