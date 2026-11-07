"""训练 CLI（三层入口的底层与中层）。

用法：
    mgd-train --profile quick --vram auto
    mgd-train --config cfg.yaml --set model.name=UNet --set train.epochs=10
    mgd-train --print-schema

优先级：默认 YAML < 用户 YAML < 预设 < --set / 快捷参数 < 显存档（安全边界）。
显存档会覆盖分块/batch/AMP 等安全相关字段；若确实要突破，请改档位而不是改分块。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from ...core.config import get_settings
from ...core.data.pairing import scan_dataset
from ...core.runtime.errors import ConfigError, DataIntegrityError, MGDError
from ...core.runtime.logging import configure_logging
from ...core.training.config_file import DEFAULT_TRAIN_CONFIG, build_train_config
from ...core.training.presets import (
    apply_tier,
    detect_vram_gb,
    profile_names,
    profile_overrides,
    tier_table,
    TRAIN_PROFILES,
)
from ...core.training.runner import run_training
from ...core.training.specs import SCHEMA, SPECS, SPEC_BY_NAME, coerce_value, parse_set_expression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="腺而易见 MGD 训练（YAML + 预设 + 显存档，三层共用同一 schema）",
        epilog="示例：mgd-train --profile quick --vram auto；mgd-train --set train.epochs=5",
    )
    parser.add_argument("--config", default=None, help=f"YAML 配置（默认 {DEFAULT_TRAIN_CONFIG.name}）")
    parser.add_argument("--profile", default=None, choices=profile_names(),
                        help="预设：academic（学术档） / quick（快速试跑） / lowvram（低显存）")
    parser.add_argument("--vram", default="auto",
                        help="显存档位：auto / 6 / 8 / 12 / 16 / 24 / 32 / 40")
    parser.add_argument("--set", action="append", default=[], metavar="A.B=C",
                        help="按 schema 覆盖任意参数，可重复")
    parser.add_argument("--resume", default=None, help="断点续训：run 目录 / checkpoints / .mgd.ckpt")
    parser.add_argument("--data-dir", default=None, help="数据集根目录")
    parser.add_argument("--model", default=None, help="模型（覆盖 model.name）")
    parser.add_argument("--epochs", type=int, default=None, help="总轮数（覆盖 train.epochs）")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="以 JSON Lines 输出事件")
    parser.add_argument("--dry-run", action="store_true", help="只打印生效配置，不训练")
    parser.add_argument("--print-schema", action="store_true", help="打印参数 schema 后退出")
    parser.add_argument("--print-tiers", action="store_true", help="打印显存档位表后退出")
    parser.add_argument("--no-tier", action="store_true", help="不套用显存档（仅调试用）")
    return parser


def _print_schema() -> None:
    print(f"{'参数':32s} {'类型':6s} {'默认':12s} 说明")
    for spec in SPECS:
        default = "" if spec.default is None else str(spec.default)
        flag = "（高级）" if spec.advanced else ""
        print(f"{spec.name:32s} {spec.kind:6s} {default:12s} {spec.label}{flag} {spec.help}")


def _print_tiers() -> None:
    rows = tier_table()
    headers = list(rows[0].keys())
    print(" | ".join(headers))
    for row in rows:
        print(" | ".join(str(row[key]) for key in headers))


def _apply_convenience(config, args):
    for section, key, value in (
        ("model", "name", args.model),
        ("train", "epochs", args.epochs),
        ("train", "batch_size", args.batch_size),
    ):
        if value is None:
            continue
        spec = SPEC_BY_NAME[f"{section}.{key}"]
        setattr(config, spec.field, coerce_value(spec, value))
    if args.resume:
        config = replace(config, resume_from=str(args.resume))
    return config


def _reapply_explicit(config, expressions: list[str]):
    """--set 在档位之上再落一次，保证用户显式值不被档位吞掉（档位只管安全字段）。"""
    for expression in expressions:
        section, key, raw = parse_set_expression(expression)
        spec = SPEC_BY_NAME[f"{section}.{key}"]
        if spec.field == "_output_dir":
            continue
        setattr(config, spec.field, coerce_value(spec, raw))
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_schema:
        _print_schema()
        return 0
    if args.print_tiers:
        _print_tiers()
        return 0

    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    configure_logging(log_file=settings.log_file)

    try:
        index = scan_dataset(settings)
    except DataIntegrityError as exc:
        print(f"[数据集错误] {exc.user_message()}", file=sys.stderr)
        return 2

    try:
        overrides = profile_overrides(args.profile)
        config, meta = build_train_config(
            Path(args.config).expanduser() if args.config else None,
            overrides=overrides,
            cli_overrides=list(args.set),
        )
        if not args.no_tier:
            config, tier_info = apply_tier(config, args.vram, detect_vram_gb())
        else:
            tier_info = {"tier": "off", "reason": "显存档已关闭（--no-tier）"}
        config = _apply_convenience(config, args)
        config = _reapply_explicit(config, list(args.set))
        config.validate()
    except (ConfigError, ValueError) as exc:
        hint = getattr(exc, "hint", None)
        print(f"[配置错误] {exc}" + (f"\n提示：{hint}" if hint else ""), file=sys.stderr)
        return 3

    print(f"样本数：{len(index)}　模型：{config.model_name}　总轮数：{config.epochs}")
    print(
        f"显存档：{tier_info.get('tier')}（{tier_info.get('status', '自定义')}）"
        f"　分块 {tier_info.get('tile', f'{config.tile_height}×{config.tile_height * 2}')}"
        f"　batch {config.batch_size}×累积 {config.grad_accum_steps}"
    )
    print(f"档位依据：{tier_info.get('reason') or tier_info.get('basis', '')}")
    if tier_info.get("status") == "估算":
        print("  ! 该档位为估算值，尚未在本机实测；首轮训练请关注显存峰值。")

    if args.dry_run:
        payload = asdict(config)
        payload.pop("run_dir", None)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    def emit(event: dict) -> None:
        if args.json:
            print(json.dumps(event, ensure_ascii=False, default=str))
            return
        if event["event"] == "start":
            print(f"设备：{event['device']}　训练 {event['train_samples']} / 验证 {event['val_samples']}")
            for warning in event["warnings"]:
                print(f"  ! {warning}")
        elif event["event"] == "epoch":
            print(
                f"epoch {event['epoch']:03d}/{event['total_epochs']} "
                f"loss={event['loss']:.4f} train_dice={event['dice']:.4f} "
                f"val_dice={event['val_dice']:.4f} lr={event['lr']:.2e} "
                f"{event.get('samples_per_second', 0):.2f} 样本/秒"
            )
        elif event["event"] == "done":
            result = event["result"]
            print(
                f"\n完成：{result.epochs_run} 轮，最佳 {result.best_metric:.4f} @ epoch {result.best_epoch}"
            )
            print(f"评估口径：{result.scope_label}")
            vram = "N/A" if result.peak_vram_gb is None else f"{result.peak_vram_gb:.3f} GB"
            print(f"吞吐：{result.samples_per_second:.2f} 样本/秒　峰值显存：{vram}")
            print(f"最佳权重：{result.best_weights}")
            if result.last_checkpoint:
                print(f"检查点：{result.last_checkpoint}")
            if result.run_dir:
                print(f"run 目录：{result.run_dir}")
            for warning in result.warnings:
                print(f"  ! {warning}")

    try:
        result, context = run_training(config, meta, settings, on_event=emit)
    except MGDError as exc:
        print(f"[失败] {exc.user_message()}", file=sys.stderr)
        return 1
    if not args.json:
        print(f"run_id：{context.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
