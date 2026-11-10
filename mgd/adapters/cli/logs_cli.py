"""日志检索 CLI：`mgd-logs <tail|search|summary|files>`——问题定位的"一条命令"。

- `tail [--last 50] [--level ERROR|WARNING|INFO] [--errors-only]` 最近的记录
- `search --case <id> | --run <id> | --grep <文本> [--last N]` 全字段匹配（ctx.* 与 msg）
- `summary [--hours 24]` 按 level/logger/stage 计数 + 最近错误 + 最慢记录
- `files` 日志目录产物与大小

公共参数：`--log-dir`（默认 Settings.log_file 所在目录）、`--json`（原始 JSON 行）。
日志文件缺失 → 友好提示退出 0（不算错误）；损坏行跳过并向 stderr 告警；
MGDError/ValueError/OSError → 退出码 2（与其余 CLI 一致）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ...core.config import get_settings
from ...core.runtime.errors import MGDError

#: JSONL 产物（与 mgd.core.runtime.logging 约定一致）
EVENTS_NAME = "mgd_events.jsonl"
ERRORS_NAME = "errors.jsonl"

_CTX_CORE = ("run_id", "case_id", "stage", "arch", "scope")
_CTX_SHORT = {"run_id": "run", "case_id": "case"}


def _reconfigure_stdio() -> None:
    """管道/重定向输出统一 UTF-8：GBK 控制台代码页下打印中文不再崩溃。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _read_jsonl(path: Path) -> tuple[list[dict], int]:
    """读 JSONL；返回 (records, 损坏行数)。文件缺失抛 FileNotFoundError，由调用方决定退出码。"""
    records: list[dict] = []
    corrupted = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            corrupted += 1
            continue
        if isinstance(payload, dict):
            records.append(payload)
        else:
            corrupted += 1
    return records, corrupted


def _load(path: Path) -> tuple[list[dict], int]:
    """加载日志文件；缺失时友好提示并按"不算错误"处理（返回空）。"""
    if not path.is_file():
        print(f"[提示] 日志文件不存在：{path}（先用业务命令产生日志即可）")
        return [], 0
    records, corrupted = _read_jsonl(path)
    if corrupted:
        print(f"[警告] {path.name}：跳过 {corrupted} 行损坏 JSON", file=sys.stderr)
    return records, corrupted


def _record_ctx(record: dict) -> dict:
    ctx = record.get("ctx")
    return ctx if isinstance(ctx, dict) else {}


def _format_record(record: dict) -> str:
    """人读格式：ts | LEVEL | logger | msg [run=.. case=.. stage=..] | Type hint: ..."""
    parts = [f"{_CTX_SHORT.get(key, key)}={_record_ctx(record)[key]}"
             for key in _CTX_CORE if _record_ctx(record).get(key)]
    parts.extend(
        f"{key}={value}" for key, value in _record_ctx(record).items() if key not in _CTX_CORE
    )
    ts = str(record.get("ts", "")).replace("T", " ")
    line = f"{ts} | {record.get('level', '?')} | {record.get('logger', '?')} | {record.get('msg', '')}"
    if parts:
        line += " [" + " ".join(parts) + "]"
    error = record.get("error")
    if isinstance(error, dict):
        line += f" | {error.get('type', 'Error')}"
        if error.get("hint"):
            line += f" hint: {error['hint']}"
    return line


def _emit(records: list[dict], as_json: bool) -> None:
    if as_json:
        for record in records:
            print(json.dumps(record, ensure_ascii=False, allow_nan=False))
    else:
        for record in records:
            print(_format_record(record))


def _cmd_tail(args: argparse.Namespace, log_dir: Path) -> int:
    source = log_dir / (ERRORS_NAME if args.errors_only else EVENTS_NAME)
    records, _ = _load(source)
    if args.level:
        wanted = str(args.level).upper()
        records = [r for r in records if str(r.get("level", "")).upper() == wanted]
    records = records[max(0, len(records) - max(0, int(args.last))):]
    _emit(records, args.json)
    return 0


def _match(record: dict, field: str | None, needle: str) -> bool:
    needle_lower = needle.lower()
    if field is not None:
        return needle_lower in str(_record_ctx(record).get(field, "")).lower()
    haystacks = [str(record.get("msg", "")), str(record.get("logger", ""))]
    haystacks.extend(str(value) for value in _record_ctx(record).values())
    return any(needle_lower in text.lower() for text in haystacks)


def _cmd_search(args: argparse.Namespace, log_dir: Path) -> int:
    records, _ = _load(log_dir / EVENTS_NAME)
    field = "case_id" if args.case else ("run_id" if args.run else None)
    needle = args.case or args.run or args.grep
    hits = [r for r in records if _match(r, field, str(needle))]
    if args.last and args.last > 0:
        hits = hits[-int(args.last):]
    _emit(hits, args.json)
    print(f"共 {len(hits)} 条命中（{log_dir / EVENTS_NAME}）")
    return 0


def _record_duration(record: dict) -> float | None:
    for source in (record, _record_ctx(record)):
        value = source.get("duration")
        if isinstance(value, (int, float)) and value == value:
            return float(value)
    return None


def _record_ts(record: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(str(record.get("ts", "")))
    except (TypeError, ValueError):
        return None


def _cmd_summary(args: argparse.Namespace, log_dir: Path) -> int:
    records, _ = _load(log_dir / EVENTS_NAME)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(0, int(args.hours)))
    window = []
    for record in records:
        ts = _record_ts(record)
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)  # 无时区的 ts 按UTC归一，避免 naive/aware 比较崩溃
        # ts 解析失败的记录保留在窗口内，不静默丢数据
        if ts is None or ts >= cutoff:
            window.append(record)

    by_level = Counter(str(r.get("level", "?")) for r in window)
    by_logger = Counter(str(r.get("logger", "?")) for r in window)
    by_stage = Counter(str(_record_ctx(r).get("stage") or "-") for r in window)
    latest_error = next((r for r in reversed(window) if str(r.get("level")) == "ERROR"), None)
    timed = [( _record_duration(r), r) for r in window]
    slowest = max((item for item in timed if item[0] is not None),
                  key=lambda item: item[0], default=None)

    if args.json:
        print(json.dumps({
            "log_dir": str(log_dir),
            "hours": int(args.hours),
            "total": len(window),
            "by_level": dict(by_level),
            "by_logger": dict(by_logger),
            "by_stage": dict(by_stage),
            "latest_error": latest_error,
            "slowest": slowest[1] if slowest is not None else None,
            "slowest_duration": slowest[0] if slowest is not None else None,
        }, ensure_ascii=False, allow_nan=False))
        return 0

    def counts(counter: Counter) -> str:
        return "、".join(f"{key}={value}" for key, value in sorted(counter.items())) or "-"

    print(f"窗口：最近 {args.hours} 小时（{log_dir}），共 {len(window)} 条")
    print(f"按级别：{counts(by_level)}")
    print(f"按模块：{counts(by_logger)}")
    print(f"按阶段：{counts(by_stage)}")
    if latest_error is not None:
        print(f"最近错误：{_format_record(latest_error)}")
    else:
        print("最近错误：无")
    if slowest is not None:
        print(f"最慢记录：{slowest[0]:.3f}s → {_format_record(slowest[1])}")
    return 0


def _cmd_files(args: argparse.Namespace, log_dir: Path) -> int:
    if not log_dir.is_dir():
        print(f"[提示] 日志目录不存在：{log_dir}")
        return 0
    entries = sorted((p for p in log_dir.iterdir() if p.is_file()), key=lambda p: p.name)
    if args.json:
        print(json.dumps(
            [{"name": p.name, "bytes": p.stat().st_size} for p in entries],
            ensure_ascii=False,
        ))
        return 0
    if not entries:
        print(f"[提示] 目录为空：{log_dir}")
        return 0
    total = 0
    for path in entries:
        size = path.stat().st_size
        total += size
        print(f"{path.name:40s} {size / 1024:10.1f} KB")
    print(f"{'合计':40s} {total / 1024:10.1f} KB（{len(entries)} 个文件）")
    return 0


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = argparse.ArgumentParser(
        prog="mgd-logs",
        description="检索 MGD 日志：tail（最近）/ search（定位 case 或 run）/ summary（概览）/ files（产物）",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{tail,search,summary,files}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-dir", default=None, help="日志目录（默认 Settings.log_file 所在目录）")
    common.add_argument("--json", action="store_true", help="输出原始 JSON 行 / JSON 摘要")

    tail = sub.add_parser("tail", parents=[common], help="最近 N 条日志（默认读 mgd_events.jsonl）")
    tail.add_argument("--last", type=int, default=50, help="显示最后 N 条（默认 50）")
    tail.add_argument("--level", choices=["ERROR", "WARNING", "INFO"], default=None,
                      help="只看指定级别（精确匹配）")
    tail.add_argument("--errors-only", action="store_true", help="改读 errors.jsonl（WARNING 及以上）")

    search = sub.add_parser("search", parents=[common], help="按 case/run/文本 全字段检索")
    group = search.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", default=None, help="匹配 ctx.case_id（子串，不区分大小写）")
    group.add_argument("--run", default=None, help="匹配 ctx.run_id（子串，不区分大小写）")
    group.add_argument("--grep", default=None, help="匹配 msg 与全部 ctx 字段（子串，不区分大小写）")
    search.add_argument("--last", type=int, default=0, help="只显示最后 N 条命中（默认全部）")

    summary = sub.add_parser("summary", parents=[common], help="按 level/logger/stage 计数 + 最近错误 + 最慢记录")
    summary.add_argument("--hours", type=int, default=24, help="统计窗口（小时，默认 24）")

    sub.add_parser("files", parents=[common], help="列出日志目录产物与大小")

    args = parser.parse_args(argv)
    try:
        log_dir = (
            Path(args.log_dir).expanduser() if args.log_dir else get_settings().log_file.parent
        )
        handler: dict[str, Callable[[argparse.Namespace, Path], int]] = {
            "tail": _cmd_tail,
            "search": _cmd_search,
            "summary": _cmd_summary,
            "files": _cmd_files,
        }
        return handler[args.command](args, log_dir)
    except (MGDError, ValueError, OSError) as exc:
        print(f"[失败] {getattr(exc, 'user_message', lambda: str(exc))()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
