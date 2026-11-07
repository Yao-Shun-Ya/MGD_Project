"""冻结基线 v2 的契约测试：schema / 显存档 / 统计 / 数据集 / 实验导出 / doctor。"""
from __future__ import annotations

import json
from dataclasses import fields

import pytest

from mgd.core.config import get_settings
from mgd.core.data.versioning import backup_dataset, integrity_report, record_version
from mgd.core.evaluation.stats import bootstrap_ci, wilcoxon_pairs
from mgd.core.runtime.errors import ConfigError
from mgd.core.training.engine import TrainConfig
from mgd.core.training.experiments import (
    export_comparison_csv,
    export_comparison_latex,
    run_summary,
)
from mgd.core.training.presets import (
    TIERS,
    apply_tier,
    get_tier,
    recommend_tier,
    tier_table,
)
from mgd.core.training.specs import SCHEMA, SPECS, parse_set_expression, validate_payload


def test_schema_fields_exist_in_train_config():
    known = {item.name for item in fields(TrainConfig)}
    for spec in SPECS:
        if spec.field == "_output_dir":
            continue
        assert spec.field in known, f"schema 指向不存在的 TrainConfig 字段：{spec.name} → {spec.field}"


def test_schema_sections_are_consistent():
    assert set(SCHEMA) == {"run", "data", "model", "train", "eval", "artifacts"}
    assert {"train.epochs", "model.name", "run.vram_tier"} <= {spec.name for spec in SPECS}


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError):
        validate_payload({"train": {"epochz": 10}})
    with pytest.raises(ConfigError):
        parse_set_expression("train.epochz=10")


def test_set_expression_type_and_range():
    section, key, raw = parse_set_expression("train.epochs=3")
    assert (section, key, raw) == ("train", "epochs", "3")
    with pytest.raises(ConfigError):
        validate_payload({"train": {"epochs": 0}})


def test_vram_tiers_cover_six_levels():
    assert set(TIERS) == {"6", "8", "12", "16", "24", "40"}
    for tier in TIERS.values():
        assert tier.status in ("实测", "估算")
        assert tier.basis
        assert tier.tile_height > 0 and tier.batch_size >= 1


def test_apply_tier_sets_safe_fields():
    config = TrainConfig()
    merged, info = apply_tier(config, "6")
    assert merged.tile_height == 256
    assert merged.batch_size == 1
    assert merged.grad_accum_steps == 4
    assert merged.deep_supervision is False
    assert info["status"] == "实测"
    assert info["peak_gb"] == 0.89
    merged40, info40 = apply_tier(config, "40")
    assert merged40.tile_height == 640 and merged40.batch_size == 4
    assert info40["tier"] == "40"


def test_recommend_tier_uses_nominal_capacity():
    assert recommend_tier(8.0)[0] == "8"
    assert recommend_tier(16.0)[0] == "16"
    assert recommend_tier(7.2)[0] == "6"
    assert recommend_tier(None)[0] in TIERS
    tier, reason = get_tier("auto", 24.0)
    assert tier.key == "24" and "降档" in reason


def test_tier_table_has_status_and_basis():
    rows = tier_table()
    assert len(rows) == 6
    assert all(row["状态"] in ("实测", "估算") for row in rows)


def test_bootstrap_and_wilcoxon():
    result = bootstrap_ci([1, 2, 3, 4, 5], n_bootstrap=200)
    assert result["n"] == 5 and result["ci_low"] <= result["mean"] <= result["ci_high"]
    pairs = wilcoxon_pairs([0.5, 0.6, 0.7, 0.8, 0.9, 0.4], [0.4, 0.5, 0.6, 0.7, 0.8, 0.3])
    assert pairs["available"] and 0 <= pairs["p_value"] <= 1


def _make_dataset(tmp_path):
    import dataclasses

    data = tmp_path / "data"
    for name in ("images", "gland_masks", "eyelid_masks"):
        (data / name).mkdir(parents=True)
    for stem in ("case_001", "case_002"):
        for name in ("images", "gland_masks", "eyelid_masks"):
            (data / name / f"{stem}.png").write_bytes(b"x")
    (data / "dataset_splits.json").write_text(
        json.dumps({"train": ["case_001"], "val": [], "test": ["case_002"],
                    "signature": "abc123"}),
        encoding="utf-8",
    )
    # home 也指向临时目录：备份/版本记录等写操作不得污染仓库
    return dataclasses.replace(
        get_settings(data_dir=str(data)),
        home=tmp_path / "home",
        output_dir=tmp_path / "outputs",
    )


def test_integrity_detects_missing_and_duplicates(tmp_path):
    settings = _make_dataset(tmp_path)
    (settings.images_dir / "case_003.png").write_bytes(b"x")
    (settings.images_dir / "case 004.png").write_bytes(b"x")
    report = integrity_report(settings)
    assert report["ok"] is False
    assert "case_003" in report["missing_gland_masks"]
    assert "case 004" in report["missing_gland_masks"]
    assert any("case 004" in item for item in report["naming_anomalies"])


def test_integrity_reads_split_counts_from_entries(tmp_path):
    """format_version=1 的划分文件把清单放在 entries.<mode> 下。"""
    settings = _make_dataset(tmp_path)
    (settings.data_dir / "dataset_splits.json").write_text(
        json.dumps({
            "format_version": 1,
            "signature": "sig-1",
            "dataset_size": 2,
            "entries": {
                "holdout": {
                    "spec": {"mode": "holdout"},
                    "train": ["case_001"],
                    "val": ["case_002"],
                    "test": [],
                }
            },
        }),
        encoding="utf-8",
    )
    report = integrity_report(settings)
    assert report["split"]["counts"] == {"train": 1, "val": 1, "test": 0}
    assert report["split"]["mode"] == "holdout"
    assert report["split"]["signature"] == "sig-1"


def test_record_version_refuses_when_incomplete(tmp_path):
    settings = _make_dataset(tmp_path)
    (settings.images_dir / "case_003.png").write_bytes(b"x")
    with pytest.raises(ValueError):
        record_version(settings)
    assert not (settings.data_dir / "dataset_version.json").exists()


def test_record_version_and_backup_do_not_overwrite(tmp_path):
    settings = _make_dataset(tmp_path)
    payload = record_version(settings, source="测试采集", device="IRT-01")
    assert payload["usable_pairs"] == 2
    assert payload["hash"]["digest"]
    target = backup_dataset(settings)
    assert (target / "images" / "case_001.png").is_file()
    second = backup_dataset(settings)
    assert second != target
    assert (target / "backup_manifest.json").is_file()


def test_run_summary_and_exports(tmp_path):
    import dataclasses

    settings = dataclasses.replace(_make_dataset(tmp_path), output_dir=tmp_path / "outputs")
    run = settings.output_dir / "runs" / "20260101T000000Z_MGD-Former_abc_123456"
    run.mkdir(parents=True)
    (run / "run_result.json").write_text(json.dumps({
        "arch": "MGD-Former", "epochs_run": 2, "best_metric": 0.61, "best_epoch": 2,
        "peak_vram_gb": 3.2, "samples_per_second": 4.5, "wall_seconds": 120.0,
    }), encoding="utf-8")
    (run / "reference_report.json").write_text(json.dumps({
        "valid": True, "split_signature": "abc123", "metrics": {"dice": {"mean": 0.6}},
    }), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "val_dice": 0.5}) + "\n" + json.dumps({"epoch": 2, "val_dice": 0.6}) + "\n",
        encoding="utf-8",
    )
    summary = run_summary(run)
    assert summary["arch"] == "MGD-Former" and summary["verified"] is True
    assert summary["split_signature"] == "abc123"
    csv_path = export_comparison_csv([summary], tmp_path / "cmp.csv")
    tex_path = export_comparison_latex([summary], tmp_path / "cmp.tex")
    assert csv_path.read_text(encoding="utf-8").startswith("run_id,arch")
    assert "\\begin{tabular}" in tex_path.read_text(encoding="utf-8")


def test_pipeline_report_scans_run_dir_when_state_empty(tmp_path):
    """只跑 --stages report 时 state 无 run，报告必须回退扫描 output_dir/runs。"""
    import dataclasses
    from pathlib import Path

    from mgd.core.workflow.pipeline import _write_report

    settings = dataclasses.replace(_make_dataset(tmp_path), output_dir=tmp_path / "outputs")
    good = settings.output_dir / "runs" / "20260101T000000Z_MGD-Former_abc_123456"
    good.mkdir(parents=True)
    (good / "run_result.json").write_text(
        json.dumps({"arch": "MGD-Former", "best_metric": 0.61}), encoding="utf-8")
    (good / "reference_report.json").write_text(
        json.dumps({"valid": True, "split_signature": "abc123"}), encoding="utf-8")
    bare = settings.output_dir / "runs" / "20260102T000000Z_MGD-Former_def_654321"
    bare.mkdir(parents=True)
    (bare / "run_result.json").write_text(
        json.dumps({"arch": "MGD-Former", "best_metric": 0.4}), encoding="utf-8")
    aborted = settings.output_dir / "runs" / "20260103T000000Z_aborted"
    aborted.mkdir(parents=True)

    target = _write_report({"entries": {}, "runs": []}, settings)
    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    assert [item["run_id"] for item in payload["runs"]] == [good.name, bare.name]
    assert payload["runs"][0]["split_signature"] == "abc123"
    assert payload["runs"][0]["best_metric"] == 0.61
    assert payload["missing_split_signature"] == [bare.name]


def test_cli_entry_modules_import():
    """入口模块必须在导入阶段就暴露问题（历史上有过 PROFILE_NAMES 之类 ImportError）。"""
    import importlib

    for name in (
        "mgd.adapters.cli.train_cli",
        "mgd.adapters.cli.eval_cli",
        "mgd.adapters.cli.pipeline_cli",
        "mgd.adapters.cli.data_cli",
        "mgd.core.runtime.doctor",
        "mgd.core.models.verify",
    ):
        importlib.import_module(name)
