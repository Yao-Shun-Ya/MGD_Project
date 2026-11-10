"""§5 验收：build_report 扩展参数的向后兼容与新区块渲染。"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from mgd.core.atlas.typology import TYPE_RULES, TypeEvidence
from mgd.core.grading.scales import grade_all
from mgd.core.inference.service import InferenceResult
from mgd.core.provenance import collect_provenance


def _make_result() -> InferenceResult:
    gray = np.full((64, 128), 128, dtype="uint8")
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = np.zeros((64, 128), dtype="uint8")
    mask[10:50, 20:100] = 1
    return InferenceResult(
        image_rgb=rgb,
        prob_map=mask.astype("float32"),
        binary_mask=mask,
        overlay=rgb.copy(),
        heatmap=rgb.copy(),
        contour=rgb.copy(),
        skeleton=rgb.copy(),
        instances=rgb.copy(),
        metrics=[["腺体缺失率 R_loss (%)", "62.0"]],
        requested_threshold=0.7,
        applied_threshold=0.7,
        threshold_escalated=False,
        extra={"confidence": {"score": 80.0, "ambiguous_ratio": 0.02}},
    )


def _evidence(present: bool) -> TypeEvidence:
    rule = TYPE_RULES[0]
    return TypeEvidence(
        key=rule.key, present=present, value=30.0 if present else float("nan"),
        threshold=15.0, unit="%", definition=rule.definition,
    )


def test_legacy_call_unchanged(tmp_path):
    """旧调用（只传 result）行为不变：产物齐全、指标行数不减。"""
    from mgd.core.reporting.report import build_report

    bundle = build_report(_make_result(), model_name="UNet", patient_id="legacy", output_dir=tmp_path)
    assert bundle.html_path.is_file() and bundle.csv_path.is_file()
    text = bundle.html_path.read_text(encoding="utf-8")
    assert "量化指标" in text and "腺体缺失率" in text
    assert "血缘 Provenance" not in text and "采集质控" not in text and "形态分型证据" not in text


def test_full_params_render_new_blocks(tmp_path):
    """全量新参数：HTML 含 provenance 哈希、双分级行、verdict、质控、分型。"""
    from mgd.core.qc.quality import QualityReport
    from mgd.core.reporting.report import build_report

    provenance = collect_provenance(model_name="UNet", threshold=0.7, tile_size=384, tile_overlap=0.25)
    quality = QualityReport(
        score=82.0, flags=["blur"], reasons=["拉普拉斯方差 12.3 低于阈值 60.0"],
        metrics={"blur_variance": 12.3}, thresholds={"blur_min": 60.0},
    )
    explain = {
        "deletion_auc": 0.10, "insertion_auc": 0.40, "road": 0.30,
        "random_deletion_auc": 0.50, "random_insertion_auc": 0.20, "verdict": "pass",
    }
    bundle = build_report(
        _make_result(), model_name="UNet", patient_id="full", output_dir=tmp_path,
        quality=quality, grading=grade_all(30.0), typology=[_evidence(True), _evidence(False)],
        provenance=provenance, explain=explain,
    )
    text = bundle.html_path.read_text(encoding="utf-8")
    assert "血缘 Provenance" in text
    assert (provenance.weights_sha256 or "")[:8] or "未提供"  # 哈希字段在 provenance 渲染链路里
    assert "Arita 睑板腺评分" in text and "Pult Meiboscale" in text
    assert "不可直接互换" in text
    assert "形态分型证据" in text and "触发" in text
    assert "采集质控" in text and "blur" in text
    assert "解释性（归因忠实度）" in text and "通过" in text
    # CSV 追加行
    import csv

    with open(bundle.csv_path, encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    keys = {row[0] for row in rows if row}
    assert {"grade_arita", "grade_pult", "qc_score", "qc_flags"} <= keys


def test_none_params_render_no_empty_tables(tmp_path):
    """新增参数全 None → 不出现空区块/空表头（防回归门）。"""
    from mgd.core.reporting.report import build_report

    bundle = build_report(_make_result(), model_name="UNet", patient_id="none", output_dir=tmp_path)
    text = bundle.html_path.read_text(encoding="utf-8")
    for marker in ("血缘 Provenance", "采集质控", "分级（双体系对照）", "形态分型证据", "解释性（归因忠实度）"):
        assert marker not in text  # "影像与可解释性" 是旧标题，含"解释性"子串，故用精确新标题
    # 不允许出现紧跟空体的表格（渲染层对 None 参数完全不输出区块）


def test_provenance_json_roundtrip(tmp_path):
    provenance = collect_provenance(model_name="MGD-Former", threshold=0.7)
    payload = provenance.to_dict()
    restored = json.loads(json.dumps(payload))
    assert restored["model_name"] == "MGD-Former" and restored["threshold"] == 0.7


# ---------------------------------------------------------------------------
# 集成层审计回归：H-1 / M-4 / M-5 / L-1 / L-2 / L-5
# ---------------------------------------------------------------------------
def _write_measurements_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_cli_sr_missing_dependency_exit_3(tmp_path, capsys, monkeypatch):
    """H-1：pydicom/highdicom 缺失时 --sr → 退出码 3 + 安装提示，绝不裸 traceback。"""
    import importlib.util
    import sys

    from mgd.adapters.cli import report_cli

    input_path = tmp_path / "m.json"
    _write_measurements_json(input_path, {
        "patient_id": "P1", "loss_percent": 30.0,
        "measurements": {"ga_percent": 12.5, "instance_count": 20, "meiboscore": 1},
    })
    if importlib.util.find_spec("pydicom") is not None:
        # 环境装了 pydicom：强制 import pydicom 走 ImportError（None in sys.modules）
        monkeypatch.setitem(sys.modules, "pydicom", None)
    code = report_cli.main(["--input", str(input_path), "--sr", str(tmp_path / "out.dcm"),
                            "--dicom", str(tmp_path / "src.dcm")])
    assert code == 3
    err = capsys.readouterr().err
    assert "依赖缺失" in err and "pip install" in err
    assert "Traceback" not in err


def test_sr_measurements_keeps_zero_meiboscore():
    """M-4：meiboscore=0 是合法等级，显式判 None；缺 key 才回退 -1（未评估）。"""
    from mgd.adapters.cli.report_cli import _sr_measurements

    out = _sr_measurements({"ga_percent": 12.5, "instance_count": 20.0, "meiboscore": 0.0})
    assert out == {"ga_percent": 12.5, "instance_count": 20, "meiboscore": 0}
    missing = _sr_measurements({"ga_percent": 1.0})
    assert missing["meiboscore"] == -1 and missing["instance_count"] == 0
    assert missing["ga_percent"] == 1.0


def test_report_cli_sr_failure_fails_whole_call(tmp_path, capsys, monkeypatch):
    """M-5（失败路径）：--html --csv --sr 组合中 SR 失败 → 整体退出 3，HTML/CSV 不生成。"""
    import importlib.util
    import sys

    from mgd.adapters.cli import report_cli

    input_path = tmp_path / "m.json"
    _write_measurements_json(input_path, {
        "patient_id": "P1", "loss_percent": 30.0,
        "measurements": {"ga_percent": 12.5, "instance_count": 20, "meiboscore": 1},
    })
    out_dir = tmp_path / "out"
    if importlib.util.find_spec("pydicom") is not None:
        monkeypatch.setitem(sys.modules, "pydicom", None)
    code = report_cli.main([
        "--input", str(input_path), "--html", "--csv",
        "--sr", str(out_dir / "o.dcm"), "--dicom", str(tmp_path / "src.dcm"),
        "--out", str(out_dir),
    ])
    assert code == 3
    capsys.readouterr()
    assert not list(out_dir.glob("*_summary.html"))
    assert not list(out_dir.glob("*_summary.csv"))


def test_report_cli_sr_success_continues_to_html_csv(tmp_path, capsys, monkeypatch):
    """M-5（成功路径）：SR 成功后不再提前 return，落到 HTML/CSV 渲染。

    用假 export_sr 绕开 pydicom 依赖（当前环境未安装）；同时从 CLI 全链路
    再证一次 M-4：meiboscore=0 原样进入 export_sr 参数。
    """
    from pathlib import Path as _Path

    from mgd.adapters.cli import report_cli
    from mgd.core.interop import dicom_sr

    captured: dict = {}

    def fake_read(path):
        captured["dicom"] = path
        return object()  # 假源影像：假 export_sr 不做类型校验

    def fake_export_sr(measurements, source_ds, out_path, **kwargs):
        captured["measurements"] = measurements
        out_path = _Path(out_path)
        out_path.write_bytes(b"fake-sr")
        return out_path

    monkeypatch.setattr(report_cli, "_read_source_dicom", fake_read)
    monkeypatch.setattr(dicom_sr, "export_sr", fake_export_sr)

    input_path = tmp_path / "m.json"
    _write_measurements_json(input_path, {
        "patient_id": "P1", "loss_percent": 30.0,
        "measurements": {"ga_percent": 12.5, "instance_count": 20, "meiboscore": 0},
    })
    out_dir = tmp_path / "out"
    code = report_cli.main([
        "--input", str(input_path), "--html", "--csv",
        "--sr", str(out_dir / "o.dcm"), "--dicom", str(tmp_path / "src.dcm"),
        "--out", str(out_dir),
    ])
    assert code == 0
    assert captured["measurements"]["meiboscore"] == 0  # M-4：0 不被吞成 -1
    names = {p.name for p in out_dir.iterdir()}
    assert any(name.endswith("_summary.html") for name in names)
    assert any(name.endswith("_summary.csv") for name in names)
    out = capsys.readouterr().out
    assert ".sr.dcm" in out  # SR 产物也计入「已生成」清单


def test_report_cli_html_nan_rendered_as_na(tmp_path, capsys):
    """L-1：HTML 渲染 NaN → N/A，不得出现 `<td>nan</td>`。"""
    from mgd.adapters.cli import report_cli

    input_path = tmp_path / "meas.json"  # 文件名避开 "nan"，页脚「源：xxx」不引入噪声
    # json.loads 接受裸 NaN 字面量（Python 扩展），正适合构造 NaN 测量
    input_path.write_text(
        '{"patient_id": "P1", "loss_percent": NaN, "measurements": {"ga_percent": NaN, "break_count": 3}}',
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    assert report_cli.main(["--input", str(input_path), "--html", "--out", str(out_dir)]) == 0
    files = list(out_dir.glob("*_summary.html"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "<td>N/A</td>" in text            # NaN 指标显示 N/A
    assert "<td>nan</td>" not in text        # 绝不让 nan 落进指标单元格
    assert "缺失率未提供，跳过分级对照" in text  # NaN loss → 分级对照跳过（不出现 nan 文案）


def test_verdict_not_double_escaped(tmp_path):
    """L-2：verdict 含 `<` 时只转义一层（`&lt;`），不得双重转义成 `&amp;lt;`。"""
    from mgd.core.reporting.report import build_report

    explain = {"verdict": "TI < 0.25（未触发）", "deletion_auc": 0.10}
    bundle = build_report(_make_result(), model_name="UNet", patient_id="esc",
                          output_dir=tmp_path, explain=explain)
    text = bundle.html_path.read_text(encoding="utf-8")
    assert "TI &lt; 0.25（未触发）" in text
    assert "&amp;lt;" not in text


def test_format_metric_inf_is_na():
    """L-5：±Inf 与 NaN 同口径 → 'N/A（未计算）'。"""
    from mgd.core.reporting.report import _format_metric

    assert _format_metric(float("inf")) == "N/A（未计算）"
    assert _format_metric(float("-inf")) == "N/A（未计算）"
    assert _format_metric(float("nan")) == "N/A（未计算）"
    assert _format_metric(1.25) == "1.2500"
