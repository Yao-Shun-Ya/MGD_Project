"""数据集 CLI：`mgd-data check|version|backup|splits`。"""
from __future__ import annotations

import argparse
import json
import sys

from ...core.config import get_settings
from ...core.data.pairing import scan_dataset
from ...core.data.splits import SplitSpec, get_splits, stems_signature
from ...core.data.versioning import (
    backup_dataset,
    integrity_report,
    integrity_rows,
    record_version,
)
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="腺而易见数据集管理")
    parser.add_argument("action", choices=["check", "version", "backup", "splits"])
    parser.add_argument("--source", default="本地采集")
    parser.add_argument("--device", default="未记录", help="采集设备")
    parser.add_argument("--hash-mode", default="fast", choices=["fast", "full"])
    parser.add_argument("--target", default=None, help="备份目标目录")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    configure_logging(log_file=settings.log_file)

    try:
        if args.action == "check":
            report = integrity_report(settings)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                for row in integrity_rows(report):
                    print(f"{row[0]:12s} {str(row[1]):>8s}   {row[2]}")
                print(f"\n划分：{report['split']}")
                if not report["ok"]:
                    print("\n缺失清单（最多 50 条）：")
                    for key in ("missing_gland_masks", "missing_eyelid_masks", "duplicate_stems"):
                        for stem in report[key]:
                            print(f"  {key}: {stem}")
            return 0 if report["ok"] else 1

        if args.action == "version":
            payload = record_version(settings, source=args.source, device=args.device,
                                     hash_mode=args.hash_mode)
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json
                  else f"已写入 {settings.data_dir / 'dataset_version.json'}\n"
                       f"哈希摘要：{payload['hash']['digest'][:16]}…　配对：{payload['usable_pairs']}")
            return 0

        if args.action == "backup":
            target = backup_dataset(settings, args.target)
            print(f"已备份到 {target}")
            return 0

        index = scan_dataset(settings)
        spec = SplitSpec()
        split = get_splits(index.pairs, spec, settings)
        signature = stems_signature([pair.stem for pair in index.pairs])
        print(f"train={len(split.train)} val={len(split.val)} test={len(split.test)} "
              f"signature={signature[:16]}…")
        return 0
    except (MGDError, ValueError, OSError) as exc:
        print(f"[失败] {getattr(exc, 'user_message', lambda: str(exc))()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
