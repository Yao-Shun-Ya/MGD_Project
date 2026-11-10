"""DICOM Segmentation Object（SEG）导出（WP-9）。

pydicom / highdicom 是可选依赖（pyproject 的 ``mgd-seg[dicom]`` extra）：
本模块在函数体内延迟导入，未安装时抛
:class:`~mgd.core.runtime.errors.InteropDependencyError`（消息含安装命令
``pip install "mgd-seg[dicom]"``），绝不静默吞掉导入失败，
也绝不在模块顶层 import 这两个库。

provenance → SEG 元数据映射（provenance 非 None 时；否则 AlgorithmVersion 与
SoftwareVersions 取 "unknown"，SeriesDescription 不带后缀）：

- ``provenance.software_version``
  → ``SegmentationAlgorithmIdentificationSequence.AlgorithmVersion``（经
  :class:`highdicom.content.AlgorithmIdentificationSequence` 的 ``version``）
  与文档级 ``SoftwareVersions``；
- ``provenance.model_name``
  → AlgorithmIdentification 的 ``parameters["model_name"]``（0.28.2 中
  parameters 是 ``dict[str, str]``，不再有编号的 Parameter 条目）；
- ``provenance.threshold``
  → AlgorithmIdentification 的 ``parameters["binarization_threshold"]``
  （BINARY SEG 的像素就是按该阈值二值化的结果，必须可追溯）；
- 以上三项同时拼接进 SeriesDescription 后缀，作为随文件可读的"注释"：
  ``[v<software_version> model=<model_name> thr=<threshold>]``
  （SeriesDescription VR 是 LO、≤64 字符，前缀从简）。

API 按 highdicom 0.28.2 实际公开签名书写并经本机运行验证（pydicom 3.0.2）：
``Segmentation`` / ``SegmentDescription`` /
``SegmentationAlgorithmIdentificationSequence``（即
``highdicom.content.AlgorithmIdentificationSequence``）/ ``Code`` 的构造、
``Segmentation.save_as(out, enforce_file_format=True)`` 全链路有真实往返
测试覆盖（``mgd/tests/test_interop_roundtrip.py``）。依赖缺失路径
（InteropDependencyError）与输入校验路径（ValueError / TypeError）同样
离线可测；任何 API 不匹配都会以显式异常失败，不做静默降级。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runtime.errors import InteropDependencyError

_INSTALL_HINT = 'pip install "mgd-seg[dicom]"'

# 睑板腺组织的分割编码（CID 4009 结构划分：Tissue / Meibomian gland，SRT）
_SEGMENTED_PROPERTY_CATEGORY = ("T-D0050", "SRT", "Tissue")
_SEGMENTED_PROPERTY_TYPE = ("T-D0500", "SRT", "Tissue")

# 算法家族（CID 8142 Algorithm Family：113092 Deep Learning，DCM）
_ALGORITHM_FAMILY = ("113092", "DCM", "Deep Learning")


def export_seg(
    mask: np.ndarray,
    source_ds,
    out_path: str | Path,
    *,
    segment_label: str = "meibomian_gland",
    algorithm_name: str = "mgd-seg",
    provenance=None,
) -> Path:
    """把二值 mask 导出为 DICOM SEG（Binary Segmentation）对象。

    参数：
        mask: 二值 uint8 数组 ``(H, W)``，取值仅 {0, 1}；H/W 必须与
            ``source_ds`` 的 Rows/Columns 一致（不一致抛 ValueError，
            该校验在依赖检查之后执行）。
        source_ds: 源影像的 ``pydicom.dataset.Dataset``（由调用方负责读取）。
            至少含 Rows/Columns；SOPClassUID/SOPInstanceUID 等几何与引用属性
            由 highdicom 用于构建源影像引用。
        out_path: 输出 ``.dcm`` 路径；父目录不存在会自动创建。
        segment_label: ``SegmentDescription.SegmentLabel``。
        algorithm_name: ``SegmentationAlgorithmIdentificationSequence.AlgorithmName``。
        provenance: :class:`mgd.core.provenance.Provenance`；映射规则见模块 docstring。

    返回：
        写入的文件路径（Path）。

    异常：
        InteropDependencyError: pydicom / highdicom 未安装（消息含
            ``pip install "mgd-seg[dicom]"``）；在一切输入校验之前抛出。
        ValueError: mask dtype / 维度 / 取值非法、形状与 source_ds 不一致，
            或 source_ds 缺少 Rows/Columns。
        TypeError: source_ds 不是 pydicom Dataset。

    注意：
        pydicom 3 的 ``save_as(enforce_file_format=True)`` 写出带 preamble 的
        合规 Part 10 文件；pydicom 2.x 走 TypeError 分支回退旧参数名。
    """
    # 1) 延迟导入：依赖缺失 → InteropDependencyError（必须在所有输入校验之前）。
    #    禁止 except: pass：导入失败必须显式向上抛。
    try:
        import pydicom  # noqa: F401  — 用于 isinstance 校验与确认基础 DICOM 库可用
        import highdicom  # noqa: F401  — 确认主包可导入（版本随 extra 安装）
        from highdicom.seg import (
            SegmentAlgorithmTypeValues,
            SegmentDescription,
            Segmentation,
            SegmentationTypeValues,
        )
        from highdicom.content import AlgorithmIdentificationSequence
        from pydicom.sr.coding import Code
    except ImportError as exc:
        raise InteropDependencyError(
            f"DICOM SEG 导出需要 pydicom / highdicom（当前未安装）。请执行：{_INSTALL_HINT}",
            hint=_INSTALL_HINT,
        ) from exc

    # 2) 输入校验（导入检查之后）
    if not isinstance(source_ds, pydicom.dataset.Dataset):
        raise TypeError(
            "source_ds 必须是 pydicom.dataset.Dataset（源影像读取由调用方负责）"
        )
    array = np.asarray(mask)
    if array.dtype != np.uint8:
        raise ValueError(f"mask 必须是 uint8 类型，实际为 {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"mask 必须是二维 (H, W)，实际为 {array.ndim} 维")
    if not np.isin(np.unique(array), (0, 1)).all():
        raise ValueError("mask 必须是二值（取值仅 0 / 1）")
    rows = getattr(source_ds, "Rows", None)
    columns = getattr(source_ds, "Columns", None)
    if rows is None or columns is None:
        raise ValueError("source_ds 缺少 Rows / Columns，无法核对像素形状")
    expected_shape = (int(rows), int(columns))
    if array.shape != expected_shape:
        raise ValueError(
            f"mask 形状 {array.shape} 与 source_ds 像素形状 {expected_shape} 不一致"
        )

    # 3) 算法标识：AlgorithmIdentificationSequence(name, family, version,
    #    parameters)——0.28.2 的 parameters 是 dict[str, str]（见模块 docstring
    #    映射）；provenance 为 None 时 version / parameters 取缺省值。
    algorithm_version = "unknown"
    parameters: dict[str, str] = {}
    if provenance is not None:
        algorithm_version = str(
            getattr(provenance, "software_version", None) or "unknown"
        )
        model_name = getattr(provenance, "model_name", None)
        if model_name:
            parameters["model_name"] = str(model_name)
        threshold = getattr(provenance, "threshold", None)
        if threshold is not None:
            parameters["binarization_threshold"] = str(float(threshold))
    algorithm_identification = AlgorithmIdentificationSequence(
        name=str(algorithm_name),
        family=Code(*_ALGORITHM_FAMILY),
        version=algorithm_version,
        parameters=parameters or None,
    )

    description = SegmentDescription(
        segment_number=1,
        segment_label=str(segment_label),
        segmented_property_category=Code(*_SEGMENTED_PROPERTY_CATEGORY),
        segmented_property_type=Code(*_SEGMENTED_PROPERTY_TYPE),
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
    )

    # 4) SeriesDescription 后缀 = 随文件可读的 provenance 注释（映射见模块
    #    docstring）；VR LO ≤64 字符，前缀从简。
    series_description = "MGD meibomian gland SEG"
    if provenance is not None:
        suffix = (
            f" [v{getattr(provenance, 'software_version', 'unknown')}"
            f" model={getattr(provenance, 'model_name', 'unknown')}"
            f" thr={getattr(provenance, 'threshold', 'unknown')}]"
        )
        series_description = (series_description + suffix)[:64]

    from pydicom.uid import generate_uid

    # 5) Segmentation 构造：0.28.2 要求 instance_number / manufacturer /
    #    manufacturer_model_name / software_versions / device_serial_number
    #    显式传入；software_versions 与 AlgorithmIdentification.version 同源
    #    （provenance.software_version），设备序列号诚实写 "unknown" 不伪造。
    seg = Segmentation(
        source_images=[source_ds],
        pixel_array=array,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=[description],
        series_instance_uid=generate_uid(),
        series_number=int(getattr(source_ds, "SeriesNumber", 0) or 0) + 1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="mgd-seg",
        manufacturer_model_name="mgd-seg",
        software_versions=algorithm_version,
        device_serial_number="unknown",
        series_description=series_description,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # pydicom >= 3：enforce_file_format=True 确保写出带 preamble 的合规 Part 10 文件
        seg.save_as(out_path, enforce_file_format=True)
    except TypeError:  # pragma: no cover - pydicom 2.x 使用旧参数名
        seg.save_as(out_path, write_like_original=False)
    return out_path
