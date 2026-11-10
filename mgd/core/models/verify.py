"""权重口径验证工具。

用法：
    python -m mgd.core.models.verify path/to/MGD_UNet_v1.0.0.pth --arch UNet

不带 --reference-report 时只做结构校验并写入 verified=false 的 manifest（状态仍为 pending）；
只有提供外部验证报告（输入尺寸 640x1280、归一化口径、参考样本 Dice）时才会写入 verified=true。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ... import __version__
from ..config import Settings, get_settings
from ..governance.hooks import audit_event
from ..runtime.errors import MGDError
from .manifest import (
    NORMALIZATION_TAG,
    build_manifest,
    sha256_file,
    validate_weight,
    write_manifest,
)
from .loader import _build_matching_model, _load_state_dict, arch_from_state_dict, load_any
from .registry import normalize_model_name


def _structure_check(weights: Path, arch: str) -> None:
    """结构校验：按权重的实际输出通道数构建模型（双通道契约权重的默认路径）。"""
    state = _load_state_dict(weights)
    model = _build_matching_model(arch, state)
    del model


def _read_reference(path: Path, settings: Settings, min_dice: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_size = list(settings.target_size)
    if list(payload.get("input_size", [])) != expected_size:
        raise MGDError(f"参考报告输入尺寸不符：{payload.get('input_size')} != {expected_size}")
    if payload.get("normalization") != NORMALIZATION_TAG:
        raise MGDError(f"参考报告归一化口径不符：{payload.get('normalization')}")
    dice = float(payload.get("dice", -1))
    if dice < min_dice:
        raise MGDError(f"参考报告 Dice={dice:.4f} 低于阈值 {min_dice}")
    if not payload.get("evidence"):
        raise MGDError("参考报告缺少 evidence 字段，无法追溯验证来源")
    return payload


def _evaluate_reference(
    weights: Path,
    arch: str,
    scope: str,
    settings: Settings,
    options=None,
) -> dict:
    """在真实数据集的持久化划分上实跑评估，生成可追溯的参考报告。"""
    from ..data.pairing import scan_dataset
    from ..data.splits import SplitSpec, get_splits, stems_signature
    from ..evaluation.evaluator import evaluate_pairs
    from ..inference.service import InferenceOptions, ModelService

    index = scan_dataset(settings)
    split = get_splits(index.pairs, SplitSpec(), settings)
    pairs = {"test": split.test, "val": split.val, "train": split.train}.get(scope, split.test)
    if not pairs:
        raise MGDError(f"{scope} 集为空，无法生成参考报告")
    service = ModelService(settings, allow_unverified=True)
    report = evaluate_pairs(
        pairs, arch=arch, weights_path=weights, settings=settings,
        service=service, scope=scope,
        options=options or InferenceOptions(tile_infer=True, tile_size=384, tta="关闭"),
    )
    return {
        "input_size": list(settings.target_size),
        "normalization": NORMALIZATION_TAG,
        "dice": float(report.summary["dice"]),
        "iou": float(report.summary["iou"]),
        "hd95": float(report.summary["hd95"]),
        "assd": float(report.summary["assd"]),
        "samples": report.samples,
        "scope": report.scope,
        "scope_label": report.scope_label,
        "split_signature": stems_signature([item.stem for item in index.pairs]),
        "evidence": {
            "command": "python -m mgd.core.models.verify --evaluate",
            "dataset": str(settings.data_dir),
            "weights": str(weights),
            "sha256": sha256_file(weights),
            "device": report.device,
            "created_at": report.created_at,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验权重并写入 manifest")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--arch", default=None,
                        help="MGD-Former / UNet / AttentionUNet / WRCANet / MGD-PriorNet（留空则从结构推断）")
    parser.add_argument("--reference-report", type=Path, default=None)
    parser.add_argument("--evaluate", action="store_true",
                        help="在真实数据集的持久化划分上实跑评估并生成参考报告")
    parser.add_argument("--scope", default="test", choices=["test", "val", "train"])
    parser.add_argument("--report-out", type=Path, default=None, help="参考报告输出路径")
    parser.add_argument("--min-dice", type=float, default=0.85)
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.weights.is_file():
        print(f"权重不存在：{args.weights}", file=sys.stderr)
        return 1

    state = load_any(args.weights)
    guessed = arch_from_state_dict(state)
    arch = normalize_model_name(args.arch or guessed or "MGD-Former")
    if args.arch and guessed and guessed != arch:
        print(f"结构提示为 {guessed}，与 --arch {arch} 不一致", file=sys.stderr)
        return 1

    try:
        _structure_check(args.weights, arch)
    except MGDError as exc:
        print(f"[结构校验失败] {exc.user_message()}", file=sys.stderr)
        return 1

    verified = False
    reference = None
    if args.evaluate:
        reference = _evaluate_reference(args.weights, arch, args.scope, settings)
        out = args.report_out or (args.weights.parent / f"{args.weights.stem}.reference_report.json")
        out.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"评估：{reference['scope_label']} n={reference['samples']} "
              f"Dice={reference['dice']:.4f} IoU={reference['iou']:.4f} HD95={reference['hd95']:.2f}")
        print(f"参考报告：{out}")
        verified = bool(reference["dice"] >= float(args.min_dice))
        if not verified:
            print(f"[拒绝转正] Test Dice={reference['dice']:.4f} < {args.min_dice}", file=sys.stderr)
    elif args.reference_report is not None:
        reference = _read_reference(args.reference_report, settings, args.min_dice)
        verified = True

    manifest = build_manifest(
        args.weights,
        arch=arch,
        model_version=__version__,
        verified=verified,
        source="verify-tool",
        settings=settings,
        trained_on=reference,
    )
    write_manifest(manifest, args.weights)
    status = validate_weight(args.weights, arch, settings=settings)

    # 权重治理是硬治理：manifest 落盘后必留审计痕（strict=True —— 审计失败比
    # 业务失败更严重，无法留痕的转正/拒绝必须让整个 verify 失败）
    audit_event(
        settings,
        "weight_verify",
        actor="local",
        payload={
            "weights": args.weights.name,
            "sha256": str(manifest.sha256)[:12],
            "verified": bool(status.status == "verified"),
        },
        strict=True,
    )

    print(f"sha256 : {sha256_file(args.weights)}")
    print(f"arch   : {arch}")
    print(f"status : {status.status} —— {status.reason}")
    return 0 if status.status == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
