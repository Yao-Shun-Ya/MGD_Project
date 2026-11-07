"""流水线 CLI：`mgd-pipeline`。

示例：
    mgd-pipeline --stages train,eval,report --folds 0,1 --profile academic --vram auto
    mgd-pipeline --stages train --configs a.yaml,b.yaml --retries 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...core.config import get_settings
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging
from ...core.training.config_file import build_train_config
from ...core.training.presets import apply_tier, detect_vram_gb, profile_overrides
from ...core.workflow.pipeline import STAGES, run_pipeline


def _parse_folds(text: str) -> list[int]:
    folds: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    return folds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="腺而易见学术训练流水线")
    parser.add_argument("--stages", default="train,eval,report",
                        help=f"逗号分隔，可选：{','.join(STAGES)}")
    parser.add_argument("--configs", default="", help="逗号分隔的多个 YAML（多配置对比）")
    parser.add_argument("--folds", default="0", help="例如 0,1-4（kfold 生效）")
    parser.add_argument("--profile", default="academic")
    parser.add_argument("--vram", default="auto")
    parser.add_argument("--set", action="append", default=[], metavar="A.B=C")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--state", default=None, help="流水线状态文件（用于断点续跑）")
    parser.add_argument("--no-resume", action="store_true", help="忽略 state，从头跑")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--data-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    configure_logging(log_file=settings.log_file)

    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    config_paths = [Path(item.strip()).expanduser() for item in args.configs.split(",") if item.strip()]
    if not config_paths:
        config_paths = [None]

    configs = []
    try:
        for path in config_paths:
            config, meta = build_train_config(
                path,
                overrides=profile_overrides(args.profile),
                cli_overrides=list(args.set),
            )
            config, tier_info = apply_tier(config, args.vram, detect_vram_gb())
            for fold in _parse_folds(args.folds):
                import dataclasses

                folded = dataclasses.replace(
                    config,
                    fold=int(fold),
                    run_name=f"{config.run_name or config.model_name}_f{fold}",
                )
                label = f"{path.stem if path else 'default'}_f{fold}"
                configs.append((label, folded, {**meta, "fold": fold, "tier": tier_info}))
    except (MGDError, ValueError) as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 3

    def emit(event: dict) -> None:
        if args.json:
            print(json.dumps(event, ensure_ascii=False, default=str))
        elif event.get("event") == "stage":
            print(f"[{event['stage']:9s}] {event['label']:16s} {event['status']}"
                  + (f" → {event.get('run_dir', '')}" if event.get("run_dir") else "")
                  + (f"（失败：{event.get('error')}）" if event.get("error") else ""))
        elif event.get("event") == "retry":
            print(f"  重试 {event['stage']}/{event['label']} 第 {event['attempt']} 次：{event['error']}")

    try:
        summary = run_pipeline(
            stages, configs, settings, retries=int(args.retries),
            state_path=args.state, resume=not args.no_resume, on_event=emit,
        )
    except MGDError as exc:
        print(f"[失败] {exc.user_message()}", file=sys.stderr)
        return 1

    failed = [item for item in summary["results"] if item.get("status") == "failed"]
    print(f"\n流水线完成：{len(summary['results'])} 个条目，失败 {len(failed)}。")
    print(f"状态文件：{summary['state_path']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
