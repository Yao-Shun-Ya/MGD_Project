"""随访 CLI：`mgd-followup init|timeline|add-measurement|compare`。

基于 ClinicStore（sqlite3，路径来自 Settings.db_path）。
timeline / compare 是只读动作：数据库不存在时给出友好提示并退出码 2，
绝不隐式建库；init / add-measurement 保持可创建。
退出码：0 成功；2 输入/运行错误。
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

from ...core.config import get_settings
from ...core.longitudinal.store import ClinicStore
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging


def _fmt(value) -> str:
    return "N/A" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value}"


def _reconfigure_stdio() -> None:
    """管道/重定向输出统一 UTF-8（M-2）：GBK 控制台代码页下打印中文不再崩溃。

    pytest 捕获缓冲等非 TextIOWrapper 流没有 reconfigure，静默保持原样。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = argparse.ArgumentParser(description="腺而易见随访（患者/就诊/测量/复核）")
    parser.add_argument("action", choices=["init", "timeline", "add-measurement", "compare"])
    parser.add_argument("--patient", default=None, help="患者编号（timeline/add-measurement/compare 必填）")
    parser.add_argument("--date", default=None, help="就诊日期（ISO，add-measurement 必填）")
    parser.add_argument("--eye", default="OS", help="眼别（OS/OD）")
    parser.add_argument("--eyelid", default="lower", help="眼睑（upper/lower）")
    parser.add_argument("--key", default=None, help="测量键（如 ga_percent）")
    parser.add_argument("--value", type=float, default=None, help="测量值")
    parser.add_argument("--unit", default="", help="单位")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--db", default=None, help="覆盖 Settings.db_path")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(log_file=settings.log_file)
    db_path = Path(args.db).expanduser() if args.db else settings.db_path

    # 只读动作绝不隐式建库（L-7）：库文件不存在 → 友好提示 + 退出码 2，不产生任何文件
    if args.action in ("timeline", "compare"):
        if db_path is None or not Path(db_path).is_file():
            print(
                f"[失败] 随访数据库不存在：{db_path or '（未配置 db_path）'}；"
                "timeline/compare 不会自动建库，请先执行 `mgd-followup init` 或 add-measurement",
                file=sys.stderr,
            )
            return 2

    try:
        store = ClinicStore(db_path)
        try:
            if args.action == "init":
                store.migrate()
                print(f"随访数据库就绪：{db_path}")
                return 0

            if not args.patient:
                print(f"[失败] {args.action} 需要 --patient <患者编号>", file=sys.stderr)
                return 2

            if args.action == "timeline":
                rows = store.timeline(args.patient)
                if args.json:
                    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
                else:
                    if not rows:
                        print(f"患者 {args.patient} 暂无就诊记录（库：{db_path}）")
                        return 0
                    for row in rows:
                        # 就诊行键是 `id`（ClinicStore.timeline 复用 list_visits 列），不是 visit_id
                        print(f"就诊 #{row['id']} {row.get('visit_date', '')} "
                              f"{row.get('eye', '')}/{row.get('eyelid', '')}")
                        for key, value in (row.get("measurements") or {}).items():
                            print(f"    {key}: {_fmt(value)}")
                return 0

            if args.action == "add-measurement":
                if not args.date or args.key is None or args.value is None:
                    print("[失败] add-measurement 需要 --date/--key/--value", file=sys.stderr)
                    return 2
                store.migrate()
                store.upsert_patient(args.patient)
                visit_id = store.add_visit(args.patient, args.date, eye=args.eye, eyelid=args.eyelid)
                store.add_measurement(visit_id, args.key, float(args.value), unit=args.unit)
                print(f"已记录：患者 {args.patient} 就诊 #{visit_id} {args.key}={args.value}{args.unit}")
                return 0

            # compare：最近两次就诊的测量差值
            rows = store.timeline(args.patient)
            if len(rows) < 2:
                print(f"[失败] 患者 {args.patient} 就诊记录不足 2 次，无法对比", file=sys.stderr)
                return 2
            base, follow = rows[-2], rows[-1]
            keys = sorted(set(base.get("measurements") or {}) | set(follow.get("measurements") or {}))
            diff_rows = []
            for key in keys:
                base_value = (base.get("measurements") or {}).get(key)
                follow_value = (follow.get("measurements") or {}).get(key)
                if base_value is None or follow_value is None or base_value != base_value or follow_value != follow_value:
                    delta = float("nan")
                else:
                    delta = follow_value - base_value
                diff_rows.append([key, _fmt(base_value), _fmt(follow_value), _fmt(delta)])
            if args.json:
                print(json.dumps({"base_visit": base.get("id"), "follow_visit": follow.get("id"),
                                  "rows": diff_rows}, ensure_ascii=False, indent=2))
            else:
                print(f"基线 #{base.get('id')} → 随访 #{follow.get('id')}")
                for row in diff_rows:
                    print(f"  {row[0]:24s} {row[1]:>12s} → {row[2]:>12s}　Δ {row[3]}")
            return 0
        finally:
            store.close()
    except (MGDError, ValueError, OSError, sqlite3.Error) as exc:
        # sqlite3.Error：库文件损坏（如写入垃圾 bytes）等数据库层错误 → 友好提示，不裸 traceback
        print(f"[失败] {getattr(exc, 'user_message', lambda: str(exc))()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
