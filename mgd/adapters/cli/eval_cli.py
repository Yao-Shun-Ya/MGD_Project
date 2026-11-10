"""评估 CLI：`mgd-eval`。

必须在独立划分上跑；输出 reference_report.json（带 split_signature）、逐样本 CSV 与 LaTeX 表。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ...core.config import get_settings
from ...core.conformal.pipeline import (
    attach_calibration_to_report,
    fit_metric_calibration,
    save_metric_calibration,
)
from ...core.data.pairing import scan_dataset
from ...core.data.splits import SplitSpec, get_splits, stems_signature
from ...core.evaluation.evaluator import evaluate_pairs, export_csv, export_latex, summary_rows
from ...core.evaluation.stats import reference_report_payload, write_reference_report
from ...core.governance.hooks import audit_event, require_or_audit
from ...core.inference.service import InferenceOptions
from ...core.models.manifest import validate_weight
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging, get_logger

logger = get_logger("cli.eval")


def print_conformal_lines(calibration: dict) -> None:
    """人类可读的两行共形结论；口径 skipped 时如实打印原因，不冒充数字。"""
    try:
        alpha = float(calibration.get("alpha", 0.10))
    except (TypeError, ValueError):
        alpha = 0.10
    coverage_pct = int(round((1.0 - alpha) * 100))
    dice_block = calibration.get("dice_coverage") or {}
    rloss_block = calibration.get("rloss_interval") or {}

    bound = dice_block.get("bound")
    if bound is None:
        print(f"Dice 覆盖下界（1-α={coverage_pct}%）：不可用（{dice_block.get('skipped_reason') or '无有效样本'}）")
    else:
        print(f"Dice 覆盖下界（1-α={coverage_pct}%）：≥{float(bound):.2f}")

    q = rloss_block.get("q")
    if q is None:
        print(f"R_loss 预测区间：不可用（{rloss_block.get('skipped_reason') or '无有效样本'}）")
    else:
        print(f"R_loss 预测区间：±{float(q):.2f} pp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="腺而易见 MGD 评估（独立测试集优先）")
    parser.add_argument("--model", default="MGD-Former")
    parser.add_argument("--weights", default=None, help="权重路径；留空 → 自动解析 model_dir")
    parser.add_argument("--scope", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--split-mode", default="holdout", choices=["holdout", "kfold"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tta", default="关闭")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--conformal-alpha", type=float, default=0.10,
                        help="共形校准失效率 α（对应 eval.conformal_alpha）；0.10 → 覆盖率 90%%")
    parser.add_argument("--no-conformal", action="store_true",
                        help="跳过 conformal 校准（不拟合、不落盘，报告不含 conformal 块）")
    parser.add_argument("--strict-weights", action="store_true",
                        help="拒绝未验证（verified=false）权重；默认允许并在报告中标注")
    parser.add_argument("--out", default=None, help="输出目录（默认 $MGD_OUTPUT_DIR/eval_<时间戳>）")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--data-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    configure_logging(log_file=settings.log_file)

    try:
        # RBAC 治理门（评估 = 批量推理）：放在数据扫描等任何实际工作之前；
        # 拒绝时 access_denied 已先行落审计，ConfigError 由本函数既有
        # except MGDError 分支接住（MGDError 子类）→ user_message + 退出码 2。
        require_or_audit(settings, "run_inference", entry="mgd-eval")
        index = scan_dataset(settings)
        spec = SplitSpec(
            mode=args.split_mode,
            n_folds=int(args.n_folds),
            fold=int(args.fold),
        )
        split = get_splits(index.pairs, spec, settings)
    except MGDError as exc:
        print(f"[数据集错误] {exc.user_message()}", file=sys.stderr)
        return 2

    mapping = {"test": split.test, "val": split.val, "train": split.train}
    if args.scope == "all":
        chosen = [("test", split.test)] if split.test else []
        if split.val:
            chosen.append(("val", split.val))
    else:
        chosen = [(args.scope, mapping[args.scope])]
    if not any(pairs for _, pairs in chosen):
        print(f"[失败] {args.scope} 集合为空，无法评估", file=sys.stderr)
        return 2

    options = InferenceOptions(
        tile_infer=True,
        tile_size=int(args.tile_size),
        tile_overlap=float(args.tile_overlap),
        threshold=float(args.threshold),
        tta=str(args.tta),
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out).expanduser().resolve() if args.out else settings.output_dir / f"eval_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = stems_signature([pair.stem for pair in index.pairs])

    exit_code = 0
    for scope, pairs in chosen:
        if not pairs:
            continue
        try:
            report = evaluate_pairs(
                pairs, arch=args.model, weights_path=args.weights, settings=settings,
                options=options, scope=scope, allow_unverified=not args.strict_weights,
            )
        except MGDError as exc:
            print(f"[失败] {exc.user_message()}", file=sys.stderr)
            exit_code = 1
            continue
        status = validate_weight(Path(report.weights), args.model, settings=settings)
        payload = reference_report_payload(
            report,
            split_signature=signature,
            data_signature=signature,
            assignment_signature=getattr(split, "assignment_signature", ""),
            n_bootstrap=int(args.bootstrap),
            extra={
                "split_mode": args.split_mode,
                "fold": int(args.fold),
                "weight_verified": bool(getattr(status, "verified", False)),
                "weight_status": getattr(status, "status", "unknown"),
                "weight_reason": getattr(status, "reason", ""),
            },
        )
        # conformal 校准：拟合 → 持久化 → 挂到报告。校准失败不得让评估失败，
        # 降级为 warning + 报告不含 conformal 块（反其道会让可复现的评估白跑）。
        calibration = None
        if not args.no_conformal:
            try:
                calibration = fit_metric_calibration(report, alpha=float(args.conformal_alpha))
                save_metric_calibration(calibration, settings)
                attach_calibration_to_report(payload, calibration)
            except Exception as exc:  # noqa: BLE001 —— 任何校准问题都只降级不中止
                logger.warning("conformal 校准跳过（不影响评估结果）：%s", exc)
                calibration = None
        target = output_dir / ("reference_report.json" if len(chosen) == 1
                               else f"reference_report_{scope}.json")
        write_reference_report(target, payload)
        export_csv(report, output_dir / f"per_sample_{scope}.csv")
        export_latex(report, output_dir / f"per_sample_{scope}.tex")

        # scope 完成即留痕（非阻塞）：只记 arch / scope / 样本数 / 校准 id 等元数据
        audit_event(
            settings,
            "evaluation",
            actor="local",
            payload={
                "arch": str(args.model),
                "scope": str(scope),
                "samples": int(report.samples),
                "calibration_id": (calibration or {}).get("calibration_id"),
                "output_dir": str(output_dir),
            },
        )

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, default=str))
        else:
            print(f"\n[{scope}] {report.scope_label} · {report.samples} 例 · {report.device}")
            print(f"权重：{Path(report.weights).name}　split_signature：{signature[:12]}…")
            # weight_verified 位于 payload["extra"]（顶层键不存在，旧代码永远误报）
            extra = payload.get("extra", {}) or {}
            if not extra.get("weight_verified", False):
                print(f"[注意] 权重未通过验证（{extra.get('weight_reason') or 'verified=false'}）；"
                      "该数字仅用于工程验证，不可直接写入论文")
            for row in summary_rows(report):
                print(f"  {row[0]:24s} {row[1]}")
            dice = payload["metrics"]["dice"]
            cldice = payload["metrics"]["cldice"]
            print(
                f"Dice 95% CI：{dice['ci_low']:.4f} – {dice['ci_high']:.4f}（bootstrap {args.bootstrap}）"
            )
            if cldice["n"]:
                print(f"clDice 95% CI：{cldice['ci_low']:.4f} – {cldice['ci_high']:.4f}")
            if calibration is not None:
                print_conformal_lines(calibration)
            print(f"产物目录：{output_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
