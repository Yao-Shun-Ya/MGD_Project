"""质量守卫总入口：以独立子进程逐个运行 quality/gates/gNN_*.py。

设计（堆料式守卫）：
- 每个门禁独立子进程 + 90 秒超时：单门禁崩溃/挂起不影响其余门禁；
- 汇总表格（门禁 / 检查数 / 耗时 / 结果）+ 尾部汇总（PASS x/18）；
- 任一 FAIL → 退出码 1；结果 JSON 始终写入 quality/reports/last_run.json；
- ``--list`` 列出全部门禁；``--only g01,g03`` 运行子集；``--json`` 追加机器可读输出。

用法（项目根）：
    ./venv/Scripts/python.exe quality/run_gates.py
    ./venv/Scripts/python.exe quality/run_gates.py --list
    ./venv/Scripts/python.exe quality/run_gates.py --only g03,g04 --json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

QUALITY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QUALITY_DIR.parent
GATES_DIR = QUALITY_DIR / "gates"
REPORTS_DIR = QUALITY_DIR / "reports"
RESULT_MARKER = "GATE_RESULT_JSON: "
GATE_FILE_RE = re.compile(r"^g(\d+)_([a-z0-9_]+)\.py$")
GATE_TIMEOUT_SECONDS = 90


def discover_gates() -> list[tuple[int, str, Path]]:
    """发现并按编号排序全部门禁：[(编号, 名字, 路径)]。"""
    gates: list[tuple[int, str, Path]] = []
    for path in GATES_DIR.glob("g*.py"):
        match = GATE_FILE_RE.match(path.name)
        if not match:
            continue
        gates.append((int(match.group(1)), path.stem, path))
    gates.sort(key=lambda item: item[0])
    return gates


def gate_docline(path: Path) -> str:
    """从门禁源码读取模块 docstring 首行（不导入模块，零副作用）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return ""
    doc = ast.get_docstring(tree) or ""
    return doc.strip().splitlines()[0].strip() if doc else ""


def run_gate(path: Path, timeout: int = GATE_TIMEOUT_SECONDS) -> dict:
    """以子进程运行单个门禁并解析其结果 JSON；崩溃/超时都归一为 FAIL。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = round(time.perf_counter() - started, 2)
        return {
            "ok": False, "name": path.stem, "checks": 0,
            "failed": [f"子进程超时（>{timeout}s）"],
            "details": f"门禁子进程超过 {timeout}s 未返回；排查是否存在挂起的子调用或死循环。",
            "elapsed_s": elapsed,
        }
    elapsed = round(time.perf_counter() - started, 2)

    result: dict | None = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                result = json.loads(line[len(RESULT_MARKER):])
                break
            except json.JSONDecodeError:
                result = None
    if result is None:
        combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip().splitlines()
        tail = " | ".join(combined[-8:])[:1500]
        result = {
            "ok": False, "name": path.stem, "checks": 0,
            "failed": [f"门禁崩溃（exit={proc.returncode}），未产出结果 JSON"],
            "details": f"退出码 {proc.returncode}；输出尾部：{tail}",
        }
    result.setdefault("ok", False)
    result.setdefault("name", path.stem)
    result.setdefault("checks", 0)
    result.setdefault("failed", [])
    result.setdefault("details", "")
    result["elapsed_s"] = elapsed
    return result


def display_width(text: str) -> int:
    """终端显示宽度（中文按 2 列计），用于表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def print_table(rows: list[dict], stream=None) -> None:
    stream = stream or sys.stdout
    headers = ["门禁", "检查数", "耗时(s)", "结果", "摘要"]
    summary = lambda r: (r.get("details") or "").split("｜｜")[0].replace("\n", " ")[:38]  # noqa: E731
    table = [[r["name"], str(r.get("checks", 0)), f"{r.get('elapsed_s', 0):.1f}",
              "PASS" if r.get("ok") else "FAIL", summary(r)] for r in rows]
    widths = [max(display_width(row[i]) for row in [headers] + table) for i in range(len(headers))]
    line = "-+-".join("-" * w for w in widths)
    print(" | ".join(pad(h, w) for h, w in zip(headers, widths)), file=stream)
    print(line, file=stream)
    for row in table:
        print(" | ".join(pad(cell, w) for cell, w in zip(row, widths)), file=stream)


def parse_only(arg: str, gates: list[tuple[int, str, Path]]) -> list[tuple[int, str, Path]]:
    wanted = {item.strip().lower() for item in arg.split(",") if item.strip()}
    selected = []
    for num, name, path in gates:
        if name.lower() in wanted or f"g{num:02d}" in wanted or str(num) in wanted:
            selected.append((num, name, path))
    unknown = wanted - {name.lower() for _, name, _ in selected} - {
        f"g{num:02d}" for num, _, _ in selected}
    if unknown:
        print(f"[警告] --only 中未识别的门禁：{', '.join(sorted(unknown))}", file=sys.stderr)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MGD 项目堆料式质量守卫（18 道门禁）")
    parser.add_argument("--list", action="store_true", help="只列出门禁清单，不运行")
    parser.add_argument("--only", default="", help="只运行指定门禁，如 --only g01,g03")
    parser.add_argument("--json", action="store_true",
                        help="stdout 只输出纯 JSON（人类可读进度改走 stderr，便于管道给 jq）")
    parser.add_argument("--timeout", type=int, default=GATE_TIMEOUT_SECONDS,
                        help=f"单门禁超时秒数（默认 {GATE_TIMEOUT_SECONDS}）")
    args = parser.parse_args(argv)

    # --json 时 stdout 必须只有 JSON：所有人类可读进度一律改走 stderr。
    out = sys.stderr if args.json else sys.stdout

    gates = discover_gates()
    if not gates:
        print("[错误] 未在 quality/gates/ 发现任何 gNN_*.py 门禁", file=sys.stderr)
        return 2
    if args.list:
        print(f"共 {len(gates)} 道门禁（quality/gates/）：")
        for num, name, path in gates:
            print(f"  {name:<28} {gate_docline(path)}")
        return 0
    if args.only:
        gates = parse_only(args.only, gates)
        if not gates:
            print("[错误] --only 未匹配到任何门禁", file=sys.stderr)
            return 2

    print(f"MGD 质量守卫：运行 {len(gates)}/{len(discover_gates())} 道门禁，"
          f"单门禁超时 {args.timeout}s，项目根 = {PROJECT_ROOT}\n", file=out)
    results: list[dict] = []
    for _num, _name, path in gates:
        print(f"▶ 运行 {path.stem} ...", flush=True, file=out)
        results.append(run_gate(path, timeout=args.timeout))

    print(file=out)
    print_table(results, out)
    passed = sum(1 for r in results if r.get("ok"))
    total_checks = sum(int(r.get("checks", 0)) for r in results)
    total_seconds = sum(float(r.get("elapsed_s", 0)) for r in results)
    print("-" * 100, file=out)
    print(f"汇总：PASS {passed}/{len(results)} ｜ 总检查项 {total_checks} ｜ 总耗时 {total_seconds:.1f}s",
          file=out)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "passed": passed,
        "total": len(results),
        "total_checks": total_checks,
        "total_seconds": round(total_seconds, 2),
        "results": results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "last_run.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 JSON 已写入：{report_path}", file=out)

    failed = [r for r in results if not r.get("ok")]
    if failed:
        print(file=out)
        for r in failed:
            print(f"✗ {r['name']}（{len(r.get('failed', []))} 项失败）", file=out)
            for item in r.get("failed", [])[:6]:
                print(f"    - {item[:220]}", file=out)
            extra = r.get("details", "")
            if extra and not r.get("failed"):
                print(f"    {extra[:400]}", file=out)
        print(f"\n{len(failed)} 道门禁未通过：详见上方失败项与 last_run.json。", file=out)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
