"""医生修正回流 CLI：`mgd-feedback record / list / summary / export`。

- record：读模型掩膜 / 编辑后掩膜（可选参考掩膜）三张图，一律灰度 →
  resize 到 Settings.target_size（掩膜只能 INTER_NEAREST，禁止 LINEAR 插值）
  → ``>0`` 二值化，然后 record_correction + append_record 落盘到
  ``<home>/db/corrections.jsonl``（``settings.deid_salt`` 设置时 --case
  落盘自动假名化，由 append_record 处理）；掩膜尺寸不符等输入错误走
  record_correction 的既有 ValueError 消息；
- list：浏览最近 N 条修正记录（人读表格 / ``--json``）；
- summary：对持久化文件做与 summarize 同口径的汇总；
- export：按 (added_px + removed_px) 降序导出训练候选清单 JSONL。

退出码：0 = 成功；2 = 输入 / 运行错误（MGDError / ValueError / OSError）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from ...core.config import Settings, get_settings
from ...core.feedback import (
    append_record,
    corrections_path,
    export_from_file,
    load_records,
    record_correction,
    summarize_file,
)
from ...core.imaging.preprocess import load_gray, resize_gray
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging


def _reconfigure_stdio() -> None:
    """管道/重定向输出统一 UTF-8（M-2）：GBK 控制台代码页下打印中文不再崩溃。

    pytest 捕获缓冲等非 TextIOWrapper 流没有 reconfigure，静默保持原样。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _load_mask(path: Path, settings: Settings) -> np.ndarray:
    """读一张掩膜图：灰度 → resize 到 target_size（一律 INTER_NEAREST）→ ``>0`` 二值化。"""
    gray = load_gray(path)
    if gray.shape[:2] != settings.target_size:
        gray = resize_gray(gray, settings.target_size, cv2.INTER_NEAREST)
    return (gray > 0).astype("uint8")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"掩膜文件不存在：{path}")


def _jsonable(value):
    """NaN → None，保证 --json 输出是严格 JSON（其余原样）。"""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _cmd_record(args: argparse.Namespace, settings: Settings) -> int:
    model_path = Path(args.model_mask).expanduser()
    edited_path = Path(args.edited_mask).expanduser()
    reference_path = Path(args.reference_mask).expanduser() if args.reference_mask else None
    for path in (model_path, edited_path, *([reference_path] if reference_path else [])):
        _require_file(path)
    model = _load_mask(model_path, settings)
    edited = _load_mask(edited_path, settings)
    reference = _load_mask(reference_path, settings) if reference_path else None
    record = record_correction(
        args.case, model, edited,
        reference_mask=reference,
        reviewer_id=args.reviewer or "unknown",
        note=args.note or "",
    )
    row = append_record(record, settings, model_mask=model, edited_mask=edited)
    if args.json:
        print(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        dice_before = "N/A" if math.isnan(record.dice_before) else f"{record.dice_before:.4f}"
        print(
            f"已记录：{corrections_path(settings)}　病例 {row['case_id']}　"
            f"新增 {record.added_px} px / 删除 {record.removed_px} px　"
            f"方向 {record.direction}　dice_before {dice_before}　"
            f"dice(model↔edit) {record.dice_after_vs_edit:.4f}"
        )
    return 0


def _cmd_list(args: argparse.Namespace, settings: Settings) -> int:
    last = args.last if args.last and args.last > 0 else None
    rows = load_records(settings, last=last)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif not rows:
        print(f"暂无修正记录（{corrections_path(settings)}）")
    else:
        for row in rows:
            note = str(row.get("note", ""))
            if len(note) > 24:
                note = note[:24] + "…"
            print(
                f"{row.get('ts', '—')}　{row.get('case_id', '—')}　"
                f"+{_jsonable(row.get('added_px', 0))} / −{_jsonable(row.get('removed_px', 0))}　"
                f"{row.get('direction', '—')}　{row.get('reviewer_id', '—')}　{note}"
            )
        print(f"共 {len(rows)} 条（文件：{corrections_path(settings)}）")
    return 0


def _cmd_summary(args: argparse.Namespace, settings: Settings) -> int:
    summary = summarize_file(settings)
    if args.json:
        payload = {
            key: (
                {name: _jsonable(value) for name, value in item.items()}
                if isinstance(item, dict)
                else _jsonable(item)
            )
            for key, item in summary.items()
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"记录数：{summary['n']}　净增病例：{summary['net_add_cases']}")
        print("方向分布：" + "、".join(
            f"{name}={count}" for name, count in summary["direction_counts"].items()
        ))
        for key in ("mean_added_px", "mean_removed_px", "mean_dice_before",
                    "mean_dice_after_vs_edit"):
            value = summary[key]
            print(f"  {key:24s} {'N/A' if value != value else f'{value:.4f}'}")
    return 0


def _cmd_export(args: argparse.Namespace, settings: Settings) -> int:
    out_dir = (
        Path(args.out).expanduser() if args.out
        else settings.output_dir / "correction_candidates"
    )
    # RBAC 接线由治理代理完成
    path = export_from_file(settings, out_dir, max_items=max(1, int(args.max)))
    count = len(path.read_text(encoding="utf-8").splitlines())
    if args.json:
        print(json.dumps({"path": str(path), "count": count}, ensure_ascii=False))
    else:
        print(f"已导出 {count} 条候选：{path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = argparse.ArgumentParser(
        prog="mgd-feedback",
        description="医生修正回流（记录 / 列表 / 汇总 / 导出训练候选；数据留存于 db/corrections.jsonl）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser(
        "record", help="记录一条修正（模型掩膜 + 编辑后掩膜 [+ 参考掩膜]）")
    record_parser.add_argument("--case", required=True,
                               help="病例标识（deid_salt 设置时落盘自动假名化）")
    record_parser.add_argument("--model-mask", required=True, help="模型输出掩膜 PNG")
    record_parser.add_argument("--edited-mask", required=True, help="医生编辑后的掩膜 PNG")
    record_parser.add_argument("--reference-mask", default=None,
                               help="可选参考掩膜 PNG（提供时计算 dice_before）")
    record_parser.add_argument("--reviewer", default="unknown", help="评审人 ID（默认 unknown）")
    record_parser.add_argument("--note", default="", help="备注")
    record_parser.add_argument("--json", action="store_true")

    list_parser = subparsers.add_parser("list", help="浏览修正记录")
    list_parser.add_argument("--last", type=int, default=20,
                             help="只显示最后 N 条（默认 20；0 / 负数 = 全部）")
    list_parser.add_argument("--json", action="store_true")

    summary_parser = subparsers.add_parser("summary", help="汇总统计（与 summarize 同口径）")
    summary_parser.add_argument("--json", action="store_true")

    export_parser = subparsers.add_parser("export", help="导出训练候选清单（按修正量降序）")
    export_parser.add_argument("--out", default=None,
                               help="输出目录（默认 outputs/correction_candidates）")
    export_parser.add_argument("--max", type=int, default=200, help="最多导出条数（默认 200）")
    export_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(log_file=settings.log_file)
    handlers = {
        "record": _cmd_record,
        "list": _cmd_list,
        "summary": _cmd_summary,
        "export": _cmd_export,
    }
    try:
        return handlers[args.command](args, settings)
    except (MGDError, ValueError, OSError) as exc:
        print(f"[失败] {getattr(exc, 'user_message', lambda: str(exc))()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
