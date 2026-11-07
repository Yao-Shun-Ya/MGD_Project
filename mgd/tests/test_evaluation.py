"""评估报告：口径标注、CSV/LaTeX 导出。"""
from __future__ import annotations

import numpy as np

from mgd.core.evaluation.evaluator import (
    EvalReport,
    EvalSampleResult,
    export_csv,
    export_latex,
    summary_rows,
)
from mgd.core.imaging.metrics import MetricResult


def _report(scope: str = "holdout") -> EvalReport:
    samples = [
        EvalSampleResult("case_a", MetricResult(dice=0.81, iou=0.68, hd95=4.2, assd=1.1, precision=0.9, recall=0.75)),
        EvalSampleResult("case_b", MetricResult(dice=float("nan"), iou=float("nan"), errors=["标签为空"])),
    ]
    summary = {
        "samples": 2,
        "dice": 0.81,
        "dice_n": 1,
        "iou": 0.68,
        "iou_n": 1,
        "hd95": 4.2,
        "hd95_n": 1,
        "assd": 1.1,
        "assd_n": 1,
        "precision": 0.9,
        "precision_n": 1,
        "recall": 0.75,
        "recall_n": 1,
    }
    return EvalReport(
        arch="UNet",
        weights="/tmp/weights/unet_best.pth",
        scope=scope,
        scope_label="独立测试/验证集" if scope == "holdout" else "训练集自评（不代表泛化性能）",
        samples=2,
        summary=summary,
        per_sample=samples,
        device="CPU",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_summary_rows_always_starts_with_scope():
    report = _report(scope="train_self")
    rows = summary_rows(report)
    assert rows[0] == ["评估口径", "训练集自评（不代表泛化性能）"]
    assert report.is_self_evaluated is True
    labels = {row[0]: row[1] for row in rows}
    assert labels["样本数"] == "2"
    assert "n=1" in labels["Dice"]


def test_holdout_scope_is_not_self_evaluated():
    report = _report(scope="holdout")
    assert report.is_self_evaluated is False
    assert summary_rows(report)[0][1] == "独立测试/验证集"


def test_export_csv_writes_scope_and_rows(tmp_path):
    report = _report()
    target = export_csv(report, tmp_path / "reports" / "eval.csv")
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert "case_a" in text
    assert "独立测试/验证集" in text
    assert "标签为空" in text


def test_export_latex_escapes_and_contains_scope(tmp_path):
    report = _report()
    report.per_sample[0] = EvalSampleResult(
        "case_a", MetricResult(dice=0.81, iou=0.68, hd95=4.2, assd=1.1, precision=0.9, recall=0.75)
    )
    target = export_latex(report, tmp_path / "table.tex")
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in text
    assert "独立测试/验证集" in text
    assert "case\\_a" in text


def test_metric_result_as_dict_keeps_nan():
    result = MetricResult(dice=float("nan"), gt_area=4)
    payload = result.as_dict()
    assert np.isnan(payload["dice"])
    assert payload["gt_area"] == 4
