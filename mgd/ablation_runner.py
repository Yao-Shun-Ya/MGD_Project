"""一键消融 CLI：在持久化 Test 集上比较所有已训练模型并产出论文图表。

用法：
    python -m mgd.ablation_runner
    python mgd/ablation_runner.py --models UNet AttentionUNet WRCANet MGD-PriorNet

本脚本不训练模型，只用已有 best 权重；缺失权重会明确跳过。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mgd.core.config import get_settings
    from mgd.core.data.pairing import scan_dataset
    from mgd.core.data.splits import SplitSpec, get_splits
    from mgd.core.evaluation.ablation import AblationConfig, run_ablation
    from mgd.core.runtime.errors import MGDError
    from mgd.core.runtime.logging import configure_logging
else:
    from .core.config import get_settings
    from .core.data.pairing import scan_dataset
    from .core.data.splits import SplitSpec, get_splits
    from .core.evaluation.ablation import AblationConfig, run_ablation
    from .core.runtime.errors import MGDError
    from .core.runtime.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MGD 消融实验")
    parser.add_argument("--models", nargs="*", default=None, help="参与模型，默认全部")
    parser.add_argument("--reference", default="MGD-PriorNet", help="Wilcoxon 参考模型")
    parser.add_argument("--fold", type=int, default=None, help="使用 K 折第几折的 Test，默认 holdout")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(log_file=settings.log_file)
    try:
        index = scan_dataset(settings)
        spec = SplitSpec(mode="kfold", fold=args.fold) if args.fold is not None else SplitSpec()
        split = get_splits(index.pairs, spec, settings)
        if not split.test:
            print("[错误] Test 集为空，无法消融", file=sys.stderr)
            return 2
        report = run_ablation(
            split.test,
            AblationConfig(
                models=args.models or None,
                reference_model=args.reference,
                output_dir=Path(args.output_dir) if args.output_dir else None,
            ),
            settings,
        )
    except MGDError as exc:
        print(f"[失败] {exc.user_message()}", file=sys.stderr)
        return 1

    print(f"Test 样本：{len(split.test)}　模型：{', '.join(report.reports)}")
    for name, item in report.reports.items():
        summary = item.summary
        print(
            f"  {name:14s} Dice={summary['dice']:.4f} IoU={summary['iou']:.4f} "
            f"HD95={summary['hd95']:.2f} clDice={summary['cldice']:.4f} "
            f"MeiboscoreAcc={summary['meiboscore_accuracy']:.3f}"
        )
    print(f"\nLaTeX：{report.latex_table}")
    print(f"CSV：{report.summary_csv}")
    for key, path in report.plots.items():
        print(f"图 {key}：{path}")
    for warning in report.warnings:
        print(f"! {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
