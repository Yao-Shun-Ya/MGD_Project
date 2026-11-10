"""WP-9 互操作导出测试（离线 / tmp_path / <10s）。

- DICOM 侧测降级路径：用
  ``monkeypatch.setitem(sys.modules, "highdicom", None)`` 的技巧把
  ``import highdicom`` 变成 ImportError —— 无论环境装没装都能测降级路径
  （export_seg / export_sr 抛 InteropDependencyError 且消息含安装命令）。
  依赖已装时的真实构造/写盘/回读由 ``test_interop_roundtrip.py`` 全链路
  覆盖；这里仅保留两条轻量真实路径用例（shape 校验 / TypeError），以
  ``_requires_dicom_stack`` 标记在缺依赖环境自动 skip。
- FHIR 侧测纯 JSON 构造（可序列化、结构、值/单位）与 provenance 集成。

测试级偏差说明：provenance 用与 ``mgd.core.provenance.Provenance`` 字段名
完全一致的鸭子类型（SimpleNamespace）而不是真实 dataclass —— 因为 import
真实类会连带 ``inference.service``（torch/monai/cv2，约 3.6s），把整组测试
顶破 10s 预算；interop 函数对 provenance 一律按属性鸭子读取（getattr），
边界行为一致。字段清单与真实 dataclass 逐项对齐，若 WP-7 改字段名，
``_PROVENANCE_FIELDS`` 需同步。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types

import numpy as np
import pytest

from mgd.core.interop import (
    LOINC_SYSTEM,
    export_seg,
    export_sr,
    to_fhir_diagnostic_report,
    to_fhir_observation,
)
from mgd.core.runtime.errors import InteropDependencyError

INSTALL_CMD = 'pip install "mgd-seg[dicom]"'

# 与 mgd.core.provenance.Provenance 的 dataclass 字段一一对应（见模块 docstring）
_PROVENANCE_FIELDS = (
    "software_version",
    "model_name",
    "weights_sha256",
    "weights_verified",
    "threshold",
    "tile_size",
    "tile_overlap",
    "tta",
    "calibration_id",
    "created_at",
)


def _force_missing_deps(monkeypatch):
    """让 import pydicom / highdicom 必然失败（sys.modules 里塞 None）。"""
    monkeypatch.setitem(sys.modules, "pydicom", None)
    monkeypatch.setitem(sys.modules, "highdicom", None)


def _make_provenance() -> types.SimpleNamespace:
    """构造字段名与真实 Provenance 完全一致的鸭子类型 provenance。"""
    values = {
        "software_version": "9.9.9-test",
        "model_name": "mgd-test-model",
        "weights_sha256": "deadbeef",
        "weights_verified": True,
        "threshold": 0.5,
        "tile_size": 256,
        "tile_overlap": 0.25,
        "tta": "off",
        "calibration_id": None,
        "created_at": "2026-11-10T08:00:00+00:00",
    }
    assert set(values) == set(_PROVENANCE_FIELDS)
    return types.SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# 1) 降级路径：依赖缺失 → InteropDependencyError（消息含安装命令）
# ---------------------------------------------------------------------------


def test_export_seg_missing_dependency(tmp_path, monkeypatch):
    _force_missing_deps(monkeypatch)
    mask = np.zeros((64, 128), dtype=np.uint8)
    source_ds = types.SimpleNamespace(Rows=64, Columns=128)  # 依赖检查先于一切校验
    with pytest.raises(InteropDependencyError, match=re.escape(INSTALL_CMD)):
        export_seg(mask, source_ds, tmp_path / "seg.dcm")


def test_export_sr_missing_dependency(tmp_path, monkeypatch):
    _force_missing_deps(monkeypatch)
    measurements = {"ga_percent": 1.0}  # 依赖检查先于 key 校验
    with pytest.raises(InteropDependencyError, match=re.escape(INSTALL_CMD)):
        export_sr(measurements, types.SimpleNamespace(), tmp_path / "sr.dcm")


# ---------------------------------------------------------------------------
# 2) to_fhir_observation：可序列化、结构、值/单位、NaN 拒绝
# ---------------------------------------------------------------------------


def test_to_fhir_observation_structure_and_json():
    obs = to_fhir_observation({"ga_percent": 12.34}, "Patient/test-1")
    assert obs["resourceType"] == "Observation"
    assert obs["status"] == "final"
    assert obs["subject"] == {"reference": "Patient/test-1"}
    coding = obs["code"]["coding"][0]
    assert coding["system"] == LOINC_SYSTEM
    assert coding["code"] and coding["display"]  # 占位码非空（诚实占位，非真实 LOINC）
    assert obs["valueQuantity"]["value"] == pytest.approx(12.34)
    assert obs["valueQuantity"]["unit"] == "%"
    assert obs["valueQuantity"]["system"] == "http://unitsofmeasure.org"
    # 纯 JSON、可序列化、不写 NaN
    payload = json.dumps(obs, ensure_ascii=False)
    assert "12.34" in payload
    assert "NaN" not in payload


def test_to_fhir_observation_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        to_fhir_observation({"ga_percent": float("nan")}, "Patient/test-1")
    with pytest.raises(ValueError, match="NaN"):
        to_fhir_observation({"ga_percent": float("inf")}, "Patient/test-1")


def test_to_fhir_observation_requires_ga_percent():
    with pytest.raises(ValueError, match="ga_percent"):
        to_fhir_observation({"instance_count": 3}, "Patient/test-1")


# ---------------------------------------------------------------------------
# 3) to_fhir_diagnostic_report：report+observations 两键、数量、缺 key
# ---------------------------------------------------------------------------


def test_to_fhir_diagnostic_report_structure():
    measurements = {"ga_percent": 12.34, "instance_count": 23, "meiboscore": 2}
    bundle = to_fhir_diagnostic_report(measurements, "Patient/test-1")
    assert set(bundle.keys()) == {"report", "observations"}  # 打包约定：不混非标准键
    report = bundle["report"]
    assert report["resourceType"] == "DiagnosticReport"
    assert report["status"] == "final"
    assert report["presentedForm"] == []
    assert "睑板腺缺失率" in report["conclusion"]
    # observations 数量 = 标量 measurement 数（3 个标量 → 3 条）
    assert len(bundle["observations"]) == 3
    assert len(report["result"]) == 3
    for obs, ref in zip(bundle["observations"], report["result"]):
        assert obs["resourceType"] == "Observation"
        assert ref == {"reference": f"Observation/{obs['id']}"}
    # 整包必须可 json.dumps（纯标准库类型）
    payload = json.dumps(bundle, ensure_ascii=False)
    assert "DiagnosticReport" in payload


def test_to_fhir_diagnostic_report_missing_ga_percent():
    with pytest.raises(ValueError, match="ga_percent"):
        to_fhir_diagnostic_report({"instance_count": 3, "meiboscore": 1}, "Patient/test-1")


# ---------------------------------------------------------------------------
# 4) provenance 集成：effectiveDateTime == provenance.created_at + extension
# ---------------------------------------------------------------------------


def test_provenance_flows_into_observation():
    prov = _make_provenance()
    obs = to_fhir_observation({"ga_percent": 8.8}, "Patient/test-1", provenance=prov)
    assert obs["effectiveDateTime"] == prov.created_at
    # provenance 也写入 extension（占位 url，见模块 docstring）
    extension_urls = [ext["url"] for ext in obs.get("extension", [])]
    assert any("provenance" in url for url in extension_urls)


def test_provenance_flows_into_report():
    prov = _make_provenance()
    bundle = to_fhir_diagnostic_report(
        {"ga_percent": 8.8, "instance_count": 7, "meiboscore": 1},
        "Patient/test-1",
        provenance=prov,
    )
    assert bundle["report"]["effectiveDateTime"] == prov.created_at
    for obs in bundle["observations"]:
        assert obs["effectiveDateTime"] == prov.created_at


# ---------------------------------------------------------------------------
# 5) 真实路径（依赖已装时运行；缺依赖环境自动 skip）
# ---------------------------------------------------------------------------

_requires_dicom_stack = pytest.mark.skipif(
    importlib.util.find_spec("pydicom") is None
    or importlib.util.find_spec("highdicom") is None,
    reason="运行时验证待 pydicom/highdicom 安装后进行",
)


@_requires_dicom_stack
def test_export_seg_shape_mismatch_valueerror(tmp_path):
    # 依赖已装的真实路径：source_ds 必须是真 pydicom Dataset（先过
    # TypeError 校验），shape 校验在依赖检查之后真实触发
    import pydicom

    source_ds = pydicom.dataset.Dataset()
    source_ds.Rows = 64
    source_ds.Columns = 128
    mask = np.zeros((32, 64), dtype=np.uint8)  # 与 Rows/Columns 不一致
    with pytest.raises(ValueError, match="不一致"):
        export_seg(mask, source_ds, tmp_path / "seg.dcm")


@_requires_dicom_stack
def test_export_seg_source_ds_typeerror(tmp_path):
    """依赖已装时，非 pydicom Dataset 的 source_ds 在依赖检查之后抛 TypeError。"""
    source_ds = types.SimpleNamespace(Rows=64, Columns=128)
    mask = np.zeros((64, 128), dtype=np.uint8)
    with pytest.raises(TypeError, match="pydicom.dataset.Dataset"):
        export_seg(mask, source_ds, tmp_path / "seg.dcm")
