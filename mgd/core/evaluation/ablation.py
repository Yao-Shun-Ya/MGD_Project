"""一键消融实验（Phase 7）。

在同一持久化 Test 集上评估所有已有 best 权重，输出：
- 论文 LaTeX 表（均值 ± 标准差，最优加粗，Wilcoxon 显著性 *）；
- CSV 汇总；
- 雷达图 / 小提琴图 / Bland–Altman 图。

缺失权重的模型会被明确跳过并写入 warnings，绝不伪造数字。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings, model_best_path
from ..data.pairing import Pair
from ..inference.service import InferenceOptions, ModelService
from ..models.manifest import validate_weight
from ..models.registry import MODEL_NAMES, normalize_model_name
from ..runtime.errors import MGDError
from ..runtime.logging import get_logger
from .evaluator import EvalReport, evaluate_pairs
from .plotting import bland_altman, radar_chart, violin_plot

logger = get_logger("evaluation.ablation")


@dataclass
class AblationConfig:
    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    reference_model: str = "MGD-PriorNet"
    output_dir: Path | None = None
    options: InferenceOptions = field(
        default_factory=lambda: InferenceOptions(tile_infer=True, tile_size=384, tile_overlap=0.25)
    )


@dataclass
class AblationReport:
    reports: dict[str, EvalReport]
    summary_csv: Path
    latex_table: Path
    plots: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""


def _fmt(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def _wilcoxon(reference: list[float], other: list[float]) -> tuple[float, str]:
    pairs = []
    for left, right in zip(reference, other):
        if np.isfinite(left) and np.isfinite(right):
            pairs.append((left, right))
    if len(pairs) < 5:
        return float("nan"), f"n={len(pairs)}<5，样本不足"
    ref = [p[0] for p in pairs]
    oth = [p[1] for p in pairs]
    if all(abs(a - b) < 1e-9 for a, b in zip(ref, oth)):
        return 1.0, "无差异"
    try:
        from scipy.stats import wilcoxon

        statistic = wilcoxon(ref, oth, zero_method="wilcox", alternative="two-sided")
        return float(statistic.pvalue), f"p={float(statistic.pvalue):.4g}"
    except Exception as exc:  # scipy 缺失或全零差
        return float("nan"), f"Wilcoxon 失败：{exc}"


def run_ablation(
    test_pairs: list[Pair],
    config: AblationConfig | None = None,
    settings: Settings | None = None,
) -> AblationReport:
    config = config or AblationConfig()
    settings = settings or get_settings()
    if not test_pairs:
        raise MGDError("Test 集为空，无法消融", hint="请先检查 dataset_splits.json / 数据量")

    output_dir = Path(config.output_dir) if config.output_dir else settings.output_dir / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, EvalReport] = {}
    warnings: list[str] = []
    # 消融按「有 best 权重就评估」的语义执行；pending 权重放行但在 warnings 中如实标注
    service = ModelService(settings, allow_unverified=True)
    for name in (config.models or list(MODEL_NAMES)):
        canonical = normalize_model_name(name)
        weight = model_best_path(canonical, settings)
        if not weight.is_file():
            warnings.append(f"{canonical}: 缺少 best 权重 {weight}，已跳过")
            logger.warning("消融跳过 %s：缺少 %s", canonical, weight)
            continue
        status = validate_weight(weight, canonical, settings=settings)
        if status.status != "verified":
            warnings.append(f"{canonical}: 权重状态 {status.status}（{status.reason}）")
        logger.info("消融评估 %s", canonical)
        reports[canonical] = evaluate_pairs(
            test_pairs, arch=canonical, weights_path=weight, settings=settings,
            service=service, options=config.options, scope="test",
        )

    if not reports:
        raise MGDError("没有可用模型参与消融", hint="\n".join(warnings))

    reference = normalize_model_name(config.reference_model)
    reference_dice = (
        [item.metrics.dice for item in reports[reference].per_sample] if reference in reports else []
    )
    if reference not in reports:
        # 参考模型缺失时 Wilcoxon 列全部失效；显式标注而不是让所有行误标 "reference"
        message = f"参考模型 {reference} 不在本次消融中（权重缺失或未评估），配对检验不可用"
        warnings.append(message)
        logger.warning(message)

    rows = []
    for name, report in reports.items():
        values = {
            "dice": [item.metrics.dice for item in report.per_sample],
            "iou": [item.metrics.iou for item in report.per_sample],
            "hd95": [item.metrics.hd95 for item in report.per_sample],
            "cldice": [item.cldice for item in report.per_sample],
            "r_loss": [item.r_loss_pred for item in report.per_sample],
        }

        def stat(key: str) -> float:
            clean = [v for v in values[key] if np.isfinite(v)]
            return float(np.mean(clean)) if clean else float("nan")

        def std(key: str) -> float:
            clean = [v for v in values[key] if np.isfinite(v)]
            return float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")

        p_value, note = (
            _wilcoxon(reference_dice, values["dice"]) if name != reference and reference_dice
            else (float("nan"), "reference missing" if reference not in reports else "reference")
        )
        rows.append({
            "model": name,
            "dice_mean": stat("dice"), "dice_std": std("dice"),
            "iou_mean": stat("iou"), "iou_std": std("iou"),
            "hd95_mean": stat("hd95"), "hd95_std": std("hd95"),
            "cldice_mean": stat("cldice"), "cldice_std": std("cldice"),
            "r_loss_mean": stat("r_loss"),
            "meiboscore_acc": report.summary.get("meiboscore_accuracy", float("nan")),
            "p_value": p_value,
            "note": note,
            "n": report.samples,
            "weights": report.weights,
        })

    summary_csv = output_dir / "ablation_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["dice_mean"] if np.isfinite(row["dice_mean"]) else -1)
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{消融实验结果（独立 Test 集，均值 $\pm$ 标准差；$^{*}$ 表示 Wilcoxon $p<0.05$）}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Model & Dice & IoU & HD95 & clDice & Meiboscore Acc \\",
        r"\hline",
    ]
    for row in rows:
        cells = []
        for key in ("dice", "iou", "hd95", "cldice"):
            text = f"{_fmt(row[key + '_mean'])} $\\pm$ {_fmt(row[key + '_std'])}"
            if row["model"] == best["model"] and key == "dice":
                text = r"\textbf{" + text + "}"
            cells.append(text)
        acc = _fmt(row["meiboscore_acc"])
        star = ""
        if np.isfinite(row["p_value"]) and row["p_value"] < 0.05:
            star = "$^{*}$"
        # 模型名同样要做下划线转义（与 note 的转义口径一致）
        latex_lines.append(" & ".join([row["model"].replace("_", r"\_") + star, *cells, acc]) + r" \\")
        if row["note"] != "reference":
            latex_lines.append(r"\multicolumn{6}{l}{\footnotesize " + row["note"].replace("_", r"\_") + "} \\")
    latex_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    latex_table = output_dir / "ablation_table.tex"
    latex_table.write_text("\n".join(latex_lines), encoding="utf-8")

    plots: dict[str, Path] = {}
    radar_input = {
        name: {
            "dice": report.summary.get("dice", float("nan")),
            "iou": report.summary.get("iou", float("nan")),
            "cldice": report.summary.get("cldice", float("nan")),
            "nhd95": 1.0 - min(1.0, report.summary.get("hd95", float("nan")) / 300.0)
            if np.isfinite(report.summary.get("hd95", float("nan"))) else float("nan"),
            "meiboscore_accuracy": report.summary.get("meiboscore_accuracy", float("nan")),
        }
        for name, report in reports.items()
    }
    plots["radar"] = radar_chart(radar_input, output_dir / "ablation_radar.png")
    plots["violin_dice"] = violin_plot(
        {name: [item.metrics.dice for item in report.per_sample] for name, report in reports.items()},
        output_dir / "ablation_violin_dice.png", metric="dice",
    )
    if reference in reports:
        ref_report = reports[reference]
        plots["bland_altman"] = bland_altman(
            [item.r_loss_pred for item in ref_report.per_sample],
            [item.r_loss_gt for item in ref_report.per_sample],
            output_dir / "ablation_bland_altman.png",
        )

    return AblationReport(
        reports=reports,
        summary_csv=summary_csv,
        latex_table=latex_table,
        plots=plots,
        warnings=warnings,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
