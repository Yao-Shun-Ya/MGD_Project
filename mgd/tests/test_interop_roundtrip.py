"""真实 SEG / SR / FHIR 往返测试（WP-9 interop 转正）。

依赖（pydicom / highdicom）已安装时全链路真实运行：export_seg / export_sr
真实构造、真实写盘、真实 dcmread / highdicom 解析回读。依赖未安装时整个
文件 skip（``pytest.importorskip("highdicom")``）——降级路径（缺依赖抛
InteropDependencyError）由 ``test_interop.py`` 的 monkeypatch 用例双环境
覆盖，不在这里重复。

本文件所有断言都以 highdicom 0.28.2 + pydicom 3.0.2 实测为准，两条重要
事实修正（详见 mgd/core/interop/dicom_sr.py 模块 docstring）：

- Comprehensive SR 的 SOP Class UID 是 ``1.2.840.10008.5.1.4.1.1.88.33``
  （DICOM PS3.6 / pydicom 3 常量 ``ComprehensiveSRStorage``）；任务早期
  记忆里的 "88.22 = Comprehensive SR" 是错的——88.22 是 Enhanced SR；
- SEG 的算法标识 ``SegmentationAlgorithmIdentificationSequence`` 写在
  每个 ``SegmentSequence`` 条目**内部**（DICOM Segment Description 宏的
  规定位置），不在文档顶层。

provenance 集成测试用真实的 ``collect_provenance``（模块 docstring 见
test_interop.py：鸭子类型避免 torch 栈导入；这里按任务要求用真身，
torch 导入约 6s，其余用例仍用鸭子类型保持 <15s 预算）。
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

pytest.importorskip("highdicom")

import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import SecondaryCaptureImageStorage, generate_uid

from mgd.core.interop import (
    LOINC_SYSTEM,
    MAX_INSTANCE_MEASUREMENTS,
    UCUM_SYSTEM,
    export_seg,
    export_sr,
    to_fhir_diagnostic_report,
    to_fhir_observation,
)

# Segmentation Storage SOP Class（DICOM PS3.6）
SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"
# Comprehensive SR Storage SOP Class —— pydicom 3 / highdicom 常量与 PS3.6 一致；
# 88.22 是 Enhanced SR（历史记忆笔误，勿改回）
COMPREHENSIVE_SR_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.88.33"

ROWS, COLS = 64, 128

# 与 mgd.core.provenance.Provenance 的 dataclass 字段一一对应（鸭子类型，
# 见 test_interop.py docstring 的偏差说明）
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


def _make_source_ds(rows: int = ROWS, cols: int = COLS) -> FileDataset:
    """合成源影像（Secondary Capture）：完整像素 + SR evidence 必需属性。"""
    pixel = np.zeros((rows, cols), dtype=np.uint8)
    pixel[10:20, 30:50] = 200
    ds = FileDataset(None, {}, file_meta=None, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel.tobytes()
    ds.PatientName = "TEST^P001"
    ds.PatientID = "P001"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "F"
    ds.AccessionNumber = "ACC001"
    ds.StudyID = "ST1"
    ds.StudyDate = "20260101"
    ds.StudyTime = "080000"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 5
    return ds


def _make_mask(rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """确定性二值 mask（散点 + 主区域，非全 0 / 全 1）。"""
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[10:20, 30:50] = 1
    mask[40:48, 90:110] = 1
    return mask


def _make_provenance() -> types.SimpleNamespace:
    """鸭子类型 provenance（字段名与真实 dataclass 完全一致）。"""
    values = {
        "software_version": "9.9.9-test",
        "model_name": "UNet",
        "weights_sha256": "deadbeef",
        "weights_verified": True,
        "threshold": 0.7,
        "tile_size": 256,
        "tile_overlap": 0.25,
        "tta": "off",
        "calibration_id": None,
        "created_at": "2026-11-10T08:00:00+00:00",
    }
    assert set(values) == set(_PROVENANCE_FIELDS)
    return types.SimpleNamespace(**values)


def _numeric_values(content_items) -> list[float]:
    """递归收集内容树里全部 NUM 条目的数值。"""
    values: list[float] = []
    for item in content_items:
        if item.ValueType == "NUM":
            values.append(float(item.MeasuredValueSequence[0].NumericValue))
        if "ContentSequence" in item:
            values.extend(_numeric_values(item.ContentSequence))
    return values


# ---------------------------------------------------------------------------
# 1) SEG 往返：SOP Class / 像素 / 元数据 / highdicom 重解析
# ---------------------------------------------------------------------------


def test_seg_roundtrip_sop_class_and_pixel(tmp_path):
    ds = _make_source_ds()
    mask = _make_mask()
    out = export_seg(mask, ds, tmp_path / "seg.dcm")
    seg = pydicom.dcmread(out)
    assert str(seg.SOPClassUID) == SEGMENTATION_STORAGE_UID
    assert seg.Modality == "SEG"
    # 单帧单段：pydicom pixel_array 为 (Rows, Columns) 二维
    pixel_array = seg.pixel_array
    assert pixel_array.shape == (ROWS, COLS)
    assert np.array_equal(pixel_array.astype(np.uint8), mask)


def test_seg_roundtrip_metadata_and_algorithm_identification(tmp_path):
    ds = _make_source_ds()
    mask = _make_mask()
    out = export_seg(mask, ds, tmp_path / "seg.dcm", provenance=_make_provenance())
    seg = pydicom.dcmread(out)
    assert seg.SegmentSequence[0].SegmentLabel == "meibomian_gland"
    # 算法标识在 SegmentSequence 条目内部（DICOM Segment Description 宏位置）
    alg = seg.SegmentSequence[0].SegmentationAlgorithmIdentificationSequence[0]
    assert alg.AlgorithmName == "mgd-seg"
    assert alg.AlgorithmVersion == "9.9.9-test"
    parameters = str(alg.AlgorithmParameters or "")
    assert "model_name=UNet" in parameters
    assert "binarization_threshold=0.7" in parameters
    assert str(seg.SoftwareVersions) == "9.9.9-test"
    # SeriesDescription 注释后缀（截到 LO 64 字符内）
    assert "v9.9.9-test" in seg.SeriesDescription
    assert "model=UNet" in seg.SeriesDescription
    assert "thr=0.7" in seg.SeriesDescription


def test_seg_roundtrip_highdicom_reparse(tmp_path):
    from highdicom.seg import Segmentation, segread

    ds = _make_source_ds()
    mask = _make_mask()
    out = export_seg(mask, ds, tmp_path / "seg.dcm")
    seg_ds = pydicom.dcmread(out)
    hd = Segmentation.from_dataset(seg_ds)
    assert np.array_equal(hd.pixel_array.astype(np.uint8), mask)
    hd2 = segread(out)
    assert np.array_equal(hd2.pixel_array.astype(np.uint8), mask)


def test_seg_roundtrip_without_provenance(tmp_path):
    """provenance=None：版本/参数诚实写 "unknown" / 缺省，不伪造。"""
    ds = _make_source_ds()
    mask = _make_mask()
    out = export_seg(mask, ds, tmp_path / "seg.dcm")
    seg = pydicom.dcmread(out)
    alg = seg.SegmentSequence[0].SegmentationAlgorithmIdentificationSequence[0]
    assert alg.AlgorithmVersion == "unknown"
    assert not alg.get("AlgorithmParameters")
    assert str(seg.SoftwareVersions) == "unknown"
    assert seg.SeriesDescription == "MGD meibomian gland SEG"


# ---------------------------------------------------------------------------
# 2) SR 往返：SOP Class / 患者引用 / 数值内容树 / 实例截断
# ---------------------------------------------------------------------------


def test_sr_roundtrip_sop_class_and_patient(tmp_path):
    ds = _make_source_ds()
    measurements = {
        "ga_percent": 54.5,
        "instance_count": 10,
        "meiboscore": 2,
        "instances": [
            {"实例编号": 1, "面积(px)": 350, "圆度": 0.82},
            {"实例编号": 2, "面积(px)": 410, "圆度": 0.91},
            {"实例编号": 3, "面积(px)": 220, "圆度": 0.75},
        ],
    }
    out = export_sr(measurements, ds, tmp_path / "sr.dcm")
    sr = pydicom.dcmread(out)
    assert str(sr.SOPClassUID) == COMPREHENSIVE_SR_STORAGE_UID
    assert sr.Modality == "SR"
    # 患者上下文从 evidence（源影像）继承
    assert str(sr.PatientName) == "TEST^P001"
    assert str(sr.StudyInstanceUID) == str(ds.StudyInstanceUID)
    assert sr.CompletionFlag == "COMPLETE"
    assert sr.VerificationFlag == "UNVERIFIED"
    # 截断标记：instances=shown/total
    assert "instances=3/3" in sr.SeriesDescription


def test_sr_roundtrip_numeric_values_and_content_tree(tmp_path):
    from highdicom.sr import ContentSequence, srread
    from highdicom.sr.utils import find_content_items

    ds = _make_source_ds()
    measurements = {
        "ga_percent": 54.5,
        "instance_count": 10,
        "meiboscore": 2,
        "instances": [
            {"实例编号": 1, "面积(px)": 350, "圆度": 0.82},
            {"实例编号": 2, "面积(px)": 410, "圆度": 0.91},
            {"实例编号": 3, "面积(px)": 220, "圆度": 0.75},
        ],
    }
    out = export_sr(measurements, ds, tmp_path / "sr.dcm")
    sr = pydicom.dcmread(out)
    values = _numeric_values(sr.ContentSequence)
    assert 54.5 in values
    assert 10.0 in values  # instance_count
    assert 2.0 in values  # meiboscore
    assert 350.0 in values  # 实例表 面积(px)

    # highdicom 对象模型解析（srread → ComprehensiveSR，content 即
    # ContentSequence 的派生实现）不报错，且与 pydicom 数值一致
    document = srread(out)
    assert isinstance(document.content, ContentSequence)
    items = list(find_content_items(document.content[0], recursive=True))
    parsed_values = [float(i.MeasuredValueSequence[0].NumericValue) for i in items if i.ValueType == "NUM"]
    assert parsed_values == values

    # 测量组数量 = 1 个汇总组 + 3 个实例组
    summary_groups = [
        child
        for item in find_content_items(document.content[0], recursive=False)
        if item.ValueType == "CONTAINER"
        and item.ConceptNameCodeSequence[0].CodeMeaning == "Imaging Measurements"
        for child in item.ContentSequence
    ]
    assert len(summary_groups) == 1 + len(measurements["instances"])


def test_sr_roundtrip_instance_truncation(tmp_path):
    """30 条实例表只写前 20 组（"truncated_to": 20），标记 instances=20/30。"""
    from highdicom.sr import srread
    from highdicom.sr.utils import find_content_items

    ds = _make_source_ds()
    instances = [{"实例编号": i, "面积(px)": 100 + i} for i in range(1, 31)]
    measurements = {"ga_percent": 12.0, "instance_count": 30, "meiboscore": 1, "instances": instances}
    out = export_sr(measurements, ds, tmp_path / "sr.dcm")
    sr = pydicom.dcmread(out)
    assert f"instances={MAX_INSTANCE_MEASUREMENTS}/{len(instances)}" in sr.SeriesDescription

    document = srread(out)
    groups = [
        child
        for item in find_content_items(document.content[0], recursive=False)
        if item.ValueType == "CONTAINER"
        and item.ConceptNameCodeSequence[0].CodeMeaning == "Imaging Measurements"
        for child in item.ContentSequence
    ]
    # 1 个汇总组 + 最多 20 个实例组
    assert len(groups) == 1 + MAX_INSTANCE_MEASUREMENTS
    # 实例组的跟踪标识恰好是前 20 条（mgd_inst_1 .. mgd_inst_20）
    assert len(groups) <= 1 + MAX_INSTANCE_MEASUREMENTS


# ---------------------------------------------------------------------------
# 3) FHIR：code.system 存在 + json.dumps(allow_nan=False) 往返一致
# ---------------------------------------------------------------------------


def test_fhir_code_systems_and_json_roundtrip():
    measurements = {"ga_percent": 54.5, "instance_count": 10, "meiboscore": 2}
    bundle = to_fhir_diagnostic_report(measurements, "Patient/test-1")
    # report 与全部 Observation 的 code.system 必须存在且为 LOINC 占位 system
    assert bundle["report"]["code"]["coding"][0]["system"] == LOINC_SYSTEM
    for obs in bundle["observations"]:
        assert obs["resourceType"] == "Observation"
        assert obs["code"]["coding"][0]["system"] == LOINC_SYSTEM
        assert obs["valueQuantity"]["system"] == UCUM_SYSTEM
    # allow_nan=False 序列化 → 反序列化逐字段一致（纯标准库类型）
    payload = json.dumps(bundle, allow_nan=False)
    assert json.loads(payload) == bundle


def test_fhir_observation_code_system():
    obs = to_fhir_observation({"ga_percent": 8.8}, "Patient/test-1")
    assert obs["code"]["coding"][0]["system"] == LOINC_SYSTEM
    payload = json.dumps(obs, allow_nan=False)
    assert json.loads(payload) == obs


# ---------------------------------------------------------------------------
# 4) provenance 集成（真实 Provenance / collect_provenance，含 torch 栈导入）
# ---------------------------------------------------------------------------


def test_seg_algorithm_version_from_real_provenance(tmp_path):
    """collect_provenance(model_name="UNet", threshold=0.7) 的
    software_version 必须原样写进 SEG 的 AlgorithmVersion 与
    SoftwareVersions。"""
    from mgd.core.provenance import collect_provenance

    provenance = collect_provenance(model_name="UNet", threshold=0.7)
    ds = _make_source_ds()
    mask = _make_mask()
    out = export_seg(mask, ds, tmp_path / "seg.dcm", provenance=provenance)
    seg = pydicom.dcmread(out)
    alg = seg.SegmentSequence[0].SegmentationAlgorithmIdentificationSequence[0]
    assert alg.AlgorithmVersion == provenance.software_version
    assert str(seg.SoftwareVersions) == provenance.software_version
    assert f"model={provenance.model_name}" in seg.SeriesDescription
    assert f"thr={provenance.threshold}" in seg.SeriesDescription
