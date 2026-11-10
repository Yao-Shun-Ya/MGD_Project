"""2026-11 深度审计修复的回归测试。

覆盖：
- transforms 可 pickle（num_workers>0 在 Windows spawn 下可用的前提）；
- 患者级分组划分（同一患者不跨 split）与 assignment_signature；
- 配置 coerce 对 NaN / 非整浮点的拒绝；
- SplitResult.__len__ 返回 int；
- reference_report 携带 assignment_signature；
- interactive 菜单构造的 --set 表达式全部合法。
"""
from __future__ import annotations

import pickle

import pytest

from mgd.core.data.pairing import Pair
from mgd.core.data.splits import (
    SplitSpec,
    assignment_signature,
    get_splits,
    load_or_create_splits,
    patient_key,
    stems_signature,
)
from mgd.core.data.transforms import build_eval_transforms, build_train_transforms
from mgd.core.evaluation.stats import reference_report_payload
from mgd.core.runtime.errors import ConfigError, DataIntegrityError
from mgd.core.training.specs import coerce_value, parse_set_expression, SPEC_BY_NAME


# ---------------- transforms pickle ----------------

def test_train_transforms_are_picklable(settings):
    transform = build_train_transforms(tile_mode=True, tile_height=64, augmentation=True, settings=settings)
    restored = pickle.loads(pickle.dumps(transform))
    assert restored is not None


def test_eval_transforms_are_picklable(settings):
    transform = build_eval_transforms(settings=settings)
    restored = pickle.loads(pickle.dumps(transform))
    assert restored is not None


# ---------------- 患者级分组 ----------------

def _pairs(tmp_path, stems) -> list[Pair]:
    return [
        Pair(stem=stem, image=tmp_path / f"{stem}.png",
             gland_mask=tmp_path / f"{stem}.png", eyelid_mask=tmp_path / f"{stem}.png")
        for stem in stems
    ]


def test_patient_key_extracts_numeric_id():
    assert patient_key("DMI_OD_LOWER_REFLECTED_IR_10209332") == "10209332"
    assert patient_key("DMI_OS_LOWER_REFLECTED_IR_14933989_001") == "14933989"
    assert patient_key("case_000") == "case_000"  # 无 ≥6 位数字 → 每个 stem 独立


def test_same_patient_never_spans_splits(settings, tmp_path):
    # 6 名患者 × 5 张影像 = 30 例；holdout 后同一患者的影像必须整组落在同一 split
    stems = [f"STUDY_{100000 + pid}_{idx:02d}" for pid in range(6) for idx in range(5)]
    split = get_splits(_pairs(tmp_path, stems), SplitSpec(), settings)

    def patients(items):
        return {patient_key(p.stem) for p in items}

    assert not (patients(split.train) & patients(split.val))
    assert not (patients(split.train) & patients(split.test))
    assert not (patients(split.val) & patients(split.test))
    assert any("患者级划分" in item for item in split.warnings)


def test_assignment_signature_formula_and_persistence(settings, tmp_path):
    stems = [f"case_{i:03d}" for i in range(20)]
    pairs = _pairs(tmp_path, stems)
    record = load_or_create_splits(pairs, SplitSpec(), settings)

    expected = assignment_signature(record.train, record.val, record.test)
    assert record.assignment_signature == expected

    # 持久化文件里也要有 assignment_signature
    import json

    from mgd.core.data.splits import splits_path

    payload = json.loads(splits_path(settings).read_text(encoding="utf-8"))
    entry = payload["entries"]["holdout"]
    assert entry["assignment_signature"] == expected

    # 复用路径同样携带
    reused = load_or_create_splits(pairs, SplitSpec(), settings)
    assert reused.assignment_signature == expected


def test_identity_grouping_matches_legacy_assignment(settings, tmp_path):
    # 无患者编号的数据集：分组退化为逐 stem，划分结果必须与旧实现（洗牌后切片）完全一致
    import random

    stems = [f"case_{i:03d}" for i in range(40)]
    split = get_splits(_pairs(tmp_path, stems), SplitSpec(), settings)

    items = sorted(stems)
    random.Random(42).shuffle(items)
    total = len(items)
    val_count = int(round(total * 0.15))
    test_count = int(round(total * 0.15))
    test_start = total - test_count
    val_start = test_start - val_count
    assert [p.stem for p in split.train] == items[:val_start]
    assert [p.stem for p in split.val] == items[val_start:test_start]
    assert [p.stem for p in split.test] == items[test_start:]


def test_spec_mismatch_reuse_warns(settings, tmp_path):
    stems = [f"case_{i:03d}" for i in range(30)]
    pairs = _pairs(tmp_path, stems)
    first = get_splits(pairs, SplitSpec(), settings)
    # 相同文件列表 + 不同 seed：仍复用持久化划分（有意设计），但必须显式告警
    changed = get_splits(pairs, SplitSpec(seed=7), settings)
    assert changed.assignment_signature == first.assignment_signature
    assert any("spec 参数变化被忽略" in item for item in changed.warnings)


def test_assignment_signature_differs_across_assignments():
    a = assignment_signature(["s1"], ["s2"], ["s3"])
    b = assignment_signature(["s2"], ["s1"], ["s3"])
    assert a != b


# ---------------- specs coerce ----------------

def test_coerce_rejects_nan():
    with pytest.raises(ConfigError, match="NaN"):
        coerce_value(SPEC_BY_NAME["train.lr"], "nan")


def test_coerce_rejects_non_integer_float_for_int_spec():
    with pytest.raises(ConfigError):
        coerce_value(SPEC_BY_NAME["train.epochs"], 19.9)
    assert coerce_value(SPEC_BY_NAME["train.epochs"], 20.0) == 20


def test_parse_set_expression_still_accepts_normal_values():
    section, key, raw = parse_set_expression("train.lr=1e-4")
    assert (section, key, raw) == ("train", "lr", "1e-4")


# ---------------- SplitResult.__len__ ----------------

def test_split_result_len_returns_int(settings, tmp_path):
    pairs = _pairs(tmp_path, [f"case_{i:03d}" for i in range(20)])
    split = get_splits(pairs, SplitSpec(), settings)
    assert isinstance(len(split), int)
    assert len(split) == len(split.train) + len(split.val) + len(split.test)
    assert "train=" in split.summary()


# ---------------- reference_report 双签名 ----------------

class _FakeMetrics:
    def __init__(self) -> None:
        self.dice = 0.8
        self.iou = 0.7
        self.hd95 = 10.0
        self.assd = 3.0
        self.precision = 0.8
        self.recall = 0.8
        self.cldice = 0.8
        self.eyelid_dice = 0.9


class _FakeSample:
    def __init__(self) -> None:
        self.metrics = _FakeMetrics()
        self.eyelid_dice = 0.9
        self.cldice = 0.8


class _FakeReport:
    scope = "test"
    scope_label = "独立测试集"
    samples = 2
    arch = "UNet"
    device = "CPU"
    weights = "fake.pth"
    summary = {"dice": 0.8}

    def __init__(self) -> None:
        self.per_sample = [_FakeSample(), _FakeSample()]
        self.errors: list[str] = []
        self.warnings: list[str] = []


def test_reference_report_carries_assignment_signature():
    payload = reference_report_payload(
        _FakeReport(),
        split_signature="a" * 64,
        data_signature="a" * 64,
        assignment_signature="b" * 64,
    )
    assert payload["assignment_signature"] == "b" * 64
    assert payload["valid"] is True


def test_reference_report_without_assignment_signature_is_still_backward_compatible():
    payload = reference_report_payload(
        _FakeReport(),
        split_signature="a" * 64,
    )
    assert payload["assignment_signature"] == ""


# ---------------- interactive 菜单的 --set 合法性 ----------------

def test_interactive_menu_set_expressions_are_schema_valid():
    """菜单构造的表达式必须全部通过 schema 解析（防止再次出现 parser 不认的参数）。"""
    from mgd.adapters.cli import interactive

    expressions = [
        "train.lr=1e-4",
        "data.val_ratio=0.15",
        "data.test_ratio=0.15",
        "data.split_mode=holdout",
        "data.tile_mode=false",
    ]
    for expression in expressions:
        section, key, raw = parse_set_expression(expression)
        coerce_value(SPEC_BY_NAME[f"{section}.{key}"], raw)
    # 静态确认菜单里不再使用 train_cli 不存在的旗标
    import inspect

    source = inspect.getsource(interactive.run_menu)
    for forbidden in ("--lr ", "--val-ratio ", "--test-ratio ", "--split-mode ", "--no-tile"):
        assert forbidden not in source, f"interactive.run_menu 仍使用 train_cli 不存在的旗标 {forbidden!r}"


# ---------------- models / inference / imaging 审计修复 ----------------

def test_structure_check_accepts_dual_channel_weights(tmp_path):
    """双通道契约权重必须能通过 verify 的结构校验（转正链路对默认模型不能断裂）。"""
    import torch

    from mgd.core.models.loader import _load_state_dict
    from mgd.core.models.registry import create_model
    from mgd.core.models.verify import _structure_check

    model = create_model("MGD-PriorNet", in_channels=1, out_channels=2)
    path = tmp_path / "priornet_dual.pth"
    torch.save(model.state_dict(), path)
    del model
    state = _load_state_dict(path)
    assert state, "state_dict 不应为空"
    _structure_check(path, "MGD-PriorNet")  # 不抛异常即通过


def test_normalize_tta_accepts_ascii_forms():
    from mgd.core.inference.service import TTA_8X, normalize_tta

    assert normalize_tta("8x") == "8x"
    assert normalize_tta("2x") == "2x"
    assert normalize_tta("4X") == "4x"
    assert normalize_tta(TTA_8X) == "8x"
    assert normalize_tta("关闭") == "off"
    assert normalize_tta("") == "off"


def test_meiboscore_uses_exact_thirds():
    from mgd.core.imaging.clinical import meiboscore_from_rloss

    assert meiboscore_from_rloss(66.67) == 3   # 旧实现 66.7 边界误判为 2
    assert meiboscore_from_rloss(66.6) == 2
    assert meiboscore_from_rloss(33.33) == 1
    assert meiboscore_from_rloss(33.35) == 2
    assert meiboscore_from_rloss(4.9) == 0
    assert meiboscore_from_rloss(float("nan")) == -1


def test_contour_visualization_honors_threshold():
    import numpy as np

    from mgd.core.imaging.visualize import generate_contour_visualization

    prob = np.zeros((64, 128), dtype=np.float32)
    prob[10:50, 20:100] = 0.6  # 高于 0.5、低于 0.9
    gray = np.full((64, 128), 128, dtype=np.uint8)
    low = generate_contour_visualization(prob, gray, min_area=10, threshold=0.5)
    high = generate_contour_visualization(prob, gray, min_area=10, threshold=0.9)
    assert low.shape == high.shape == (64, 128, 3)
    assert not np.array_equal(low, high), "不同阈值应产生不同轮廓"


def test_corrupted_manifest_becomes_pending_not_crash(tmp_path, settings):
    """手编/缺字段的 manifest 必须落到 pending，而不是让 TypeError 穿透。"""
    import json

    from mgd.core.models.manifest import validate_weight

    weights = tmp_path / "fake.pth"
    weights.write_bytes(b"fake-weights")
    (weights.parent / "fake.pth.manifest.json").write_text(
        json.dumps({"name": "fake.pth", "arch": "UNet"}),  # 缺 sha256 / input_size 等
        encoding="utf-8",
    )
    status = validate_weight(weights, "UNet", settings=settings)
    assert status.status == "pending"


def test_pretrained_remap_includes_stage_norms():
    """HF 的 stage 输出 LayerNorm 必须映射到 encoder.norms.{i}（带 encoder. 前缀）。"""
    import torch

    from mgd.core.models.pretrained import _remap

    state = {
        "segformer.encoder.layer_norm.0.weight": torch.zeros(64),
        "segformer.encoder.layer_norm.0.bias": torch.zeros(64),
        "segformer.encoder.layer_norm.3.bias": torch.zeros(512),
    }
    mapped = _remap(state)
    assert "encoder.norms.0.weight" in mapped
    assert "encoder.norms.0.bias" in mapped
    assert "encoder.norms.3.bias" in mapped


def test_pretrained_default_dir_discovered(tmp_path, monkeypatch):
    """不设 MGD_PRETRAINED_DIR 时，settings 默认目录里的权重必须被发现。"""
    from mgd.core.config import Settings, set_settings
    from mgd.core.models.pretrained import _candidate_files

    monkeypatch.delenv("MGD_PRETRAINED_DIR", raising=False)
    settings = Settings(home=tmp_path, target_height=64, target_width=128)
    set_settings(settings)  # _candidate_files 内部经 get_settings() 单例读取
    target = settings.model_dir / "pretrained" / "ade20k.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"stub")
    files = _candidate_files("ade20k")
    assert target in files
