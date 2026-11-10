"""DICOM Structured Report（SR，TID 1500 Measurement Report）导出（WP-9）。

pydicom / highdicom 是可选依赖（pyproject 的 ``mgd-seg[dicom]`` extra）：
本模块在函数体内延迟导入，未安装时抛
:class:`~mgd.core.runtime.errors.InteropDependencyError`（消息含安装命令
``pip install "mgd-seg[dicom]"``），绝不静默吞掉导入失败，
也绝不在模块顶层 import 这两个库。

文档结构（TID 1500 风格，highdicom 0.28.2 API）：

- 报告根容器由 ``highdicom.sr.MeasurementReport``（无 ``template`` 参数，
  title 缺省即 TID 1500 Measurement Report 根码，根容器自带
  ``template_id=1500``）构建；
- 测量组用 ``MeasurementsAndQualitativeEvaluations``（0.28.2 已移除
  ``ImageMeasurementGroup``），其 ``source_images`` 引用源影像
  （``SourceImageForMeasurementGroup``）；
- 测量条目 ``Measurement(name=..., value=..., unit=..., tracking_identifier=...)``
  —— 0.28.2 中 name 是必填的编码（本地私有方案 ``"MGD"``，与 FHIR 模块的
  "工程占位" 策略一致：上线前由临床信息学审核后替换正式 LOINC/SNOMED 码），
  value 是裸 float、unit 是编码（``NumericMeasurementValue`` / ``Unit`` /
  ``MeasuringUnits`` 类已在 0.28.2 移除）；UCUM 单位（``%`` / ``1`` /
  ``[pix]`` / ``deg``）；
- 一个"汇总"测量组：腺体缺失率（tracking id ``ga_percent``）、腺体计数
  （``instance_count``）、Meiboscore（``meiboscore``）+ 其他数值标量；
- 每实例表（``measurements["instances"]``，即 instance_df 转出的
  dict 记录列表）逐条转成独立的测量组，**最多只写前
  MAX_INSTANCE_MEASUREMENTS（=20）条**（"truncated_to": 20）：实例组在
  "Imaging Measurements" 容器里的数量上限即体现截断，超出部分不写入；
  截断还反映在 SeriesDescription 的 ``instances=<shown>/<total>`` 后缀里，
  便于人工核对；
- 文档封套用 ``highdicom.sr.ComprehensiveSR``（0.28.2 正路 API）：
  ``evidence=[source_ds]`` 自动继承患者/检查上下文，SOP Class 为
  Comprehensive SR Storage（pydicom 3 / DICOM PS3.6：
  ``1.2.840.10008.5.1.4.1.1.88.33``；注意 88.22 是 Enhanced SR，
  旧代码记忆中的 "88.22 = Comprehensive SR" 是错的）；
  ``is_complete=True`` → CompletionFlag=COMPLETE，未验证 →
  VerificationFlag=UNVERIFIED。

provenance → SR 元数据映射（provenance 非 None 时）：

- ``provenance.software_version`` → 文档 ``SoftwareVersions``（缺省 "unknown"）；
- ``provenance.model_name`` → SeriesDescription 后缀 ``model=<...>``；
- ``provenance.threshold`` → SeriesDescription 后缀 ``thr=<...>``；
- 观察者固定为 Person ``MGD^Analysis^Pipeline``（Person Name VR 三段式；
  自动化流水线以人为登记，设备身份走文档级
  Manufacturer/ManufacturerModelName/SoftwareVersions）。

全部构造经本机运行验证（highdicom 0.28.2 + pydicom 3.0.2），真实往返测试
见 ``mgd/tests/test_interop_roundtrip.py``。依赖缺失路径
（InteropDependencyError）与输入校验路径（ValueError / TypeError）同样
离线可测；任何 API 不匹配都会以显式异常失败，不做静默降级。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from ..runtime.errors import InteropDependencyError

_INSTALL_HINT = 'pip install "mgd-seg[dicom]"'

# measurements 必含的三个汇总标量（缺失任一 → ValueError，消息列出缺失 key）
_REQUIRED_KEYS = ("ga_percent", "instance_count", "meiboscore")

# 每实例测量组的数量上限（"truncated_to": 20，见模块 docstring）
MAX_INSTANCE_MEASUREMENTS = 20

# Comprehensive SR Storage SOP Class（DICOM PS3.6 / pydicom 3 常量；
# 88.22 是 Enhanced SR，勿混）
_COMPREHENSIVE_SR_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.88.33"

# 测量名编码用本地私有方案 "MGD"（工程占位，见模块 docstring），CodeValue
# 为 SH（≤16 字符）——identifier 全部经 _TRACK_BY_KEY / _slug 保证 ≤16。
_MEASUREMENT_SCHEME = "MGD"

# 三个汇总标量的可读含义（CodeMeaning）
_MEASUREMENT_MEANINGS = {
    "ga_percent": "Meibomian gland area loss percent",
    "instance_count": "Meibomian gland count",
    "meiboscore": "Meibomian gland meiboscore",
}

# 实例表中文列名 → SR tracking identifier（SH，≤16 字节 ASCII）与
# UCUM 单位（值 + 可读含义）。未列出的键用 _slug() 做 ASCII 兜底，
# 单位按无量纲 "1" 处理。
_TRACK_BY_KEY = {
    "实例编号": ("instance_index", "1", "unity"),
    "面积(px)": ("area_px", "[pix]", "pixel"),
    "周长(px)": ("perimeter_px", "[pix]", "pixel"),
    "长轴(°)": ("major_axis_deg", "deg", "degree"),
    "短轴(°)": ("minor_axis_deg", "deg", "degree"),
    "短长比": ("aspect_ratio", "1", "unity"),
    "圆度": ("roundness", "1", "unity"),
}

# ComprehensiveSR(evidence=...) 直接访问的源影像属性（缺失即 AttributeError，
# 这里提前用 ValueError 挡住并给出可读消息）
_REQUIRED_SOURCE_ATTRS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPClassUID",
    "SOPInstanceUID",
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "StudyTime",
)


def _slug(key: str) -> str:
    """把任意列名压成 SH 兼容（≤16 字节 ASCII）的标识兜底。"""
    cleaned = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_"
        for ch in str(key)
    ).strip("_")
    return (cleaned[:16] or "measurement")


def _finite(value, name: str) -> float:
    """转 float 并拒绝 NaN / Inf（SR 数值测量不允许写非有限数）。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值，实际为 {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 不允许为 NaN / Inf")
    return number


def export_sr(
    measurements: dict,
    source_ds,
    out_path: str | Path,
    *,
    provenance=None,
) -> Path:
    """把测量结果导出为 DICOM SR（TID 1500 Measurement Report）。

    参数：
        measurements: 汇总与实例级测量::

            {
                "ga_percent": float,      # 腺体缺失率（必填）
                "instance_count": int,    # 腺体计数（必填）
                "meiboscore": int,        # Meiboscore（必填）
                "instances": [            # 每实例表（可选，截断到前 20 条，
                    {"实例编号": ..., "面积(px)": ..., ...},   # 见 "truncated_to": 20）
                ],
            }

            缺 ``ga_percent`` / ``instance_count`` / ``meiboscore`` 任一 →
            ValueError（消息列出缺失 key）；``instances`` 缺省视为空表。
            其他数值标量字段也会作为额外 Measurement 写入汇总组。
        source_ds: 源影像 ``pydicom.dataset.Dataset``（调用方负责读取）；
            SR 必须引用源影像并挂在其 Study 下，且 ComprehensiveSR 会从
            evidence 继承患者/检查上下文，故须含 _REQUIRED_SOURCE_ATTRS
            列出的属性（空值即可，不能缺元素）。
        out_path: 输出 ``.dcm`` 路径；父目录不存在会自动创建。
        provenance: :class:`mgd.core.provenance.Provenance`；映射见模块 docstring。

    返回：
        写入的文件路径（Path）。

    异常：
        InteropDependencyError: pydicom / highdicom 未安装（消息含
            ``pip install "mgd-seg[dicom]"``）；在一切输入校验之前抛出。
        ValueError: measurements 缺必需 key、值非有限数、instances 结构非法，
            或 source_ds 缺少引用/上下文必需属性。
        TypeError: source_ds 不是 pydicom Dataset。

    注意：
        pydicom 3 的 ``save_as(enforce_file_format=True)`` 写出带 preamble 的
        合规 Part 10 文件；pydicom 2.x 走 TypeError 分支回退旧参数名。
    """
    # 1) 延迟导入：依赖缺失 → InteropDependencyError（必须在所有输入校验之前）。
    #    禁止 except: pass：导入失败必须显式向上抛。
    try:
        import pydicom
        from pydicom.dataset import Dataset
        import highdicom  # noqa: F401  — 确认主包可导入
        from highdicom.sr import (
            ComprehensiveSR,
            Measurement,
            MeasurementReport,
            MeasurementsAndQualitativeEvaluations,
            ObservationContext,
            ObserverContext,
            PersonObserverIdentifyingAttributes,
            TrackingIdentifier,
        )
        from highdicom.sr.content import SourceImageForMeasurementGroup
        from pydicom.sr.codedict import codes
        from pydicom.sr.coding import Code
        from pydicom.uid import generate_uid
    except ImportError as exc:
        raise InteropDependencyError(
            f"DICOM SR 导出需要 pydicom / highdicom（当前未安装）。请执行：{_INSTALL_HINT}",
            hint=_INSTALL_HINT,
        ) from exc

    # 2) 输入校验（导入检查之后）
    if not isinstance(source_ds, Dataset):
        raise TypeError(
            "source_ds 必须是 pydicom.dataset.Dataset（源影像读取由调用方负责）"
        )
    if not isinstance(measurements, dict):
        raise ValueError("measurements 必须是 dict")
    missing = [key for key in _REQUIRED_KEYS if key not in measurements]
    if missing:
        raise ValueError(
            f"measurements 缺少必需字段: {missing}（需要 {list(_REQUIRED_KEYS)}）"
        )
    ga_percent = _finite(measurements["ga_percent"], "ga_percent")
    instance_count = _finite(measurements["instance_count"], "instance_count")
    meiboscore = _finite(measurements["meiboscore"], "meiboscore")

    absent = [attr for attr in _REQUIRED_SOURCE_ATTRS if not hasattr(source_ds, attr)]
    if absent:
        raise ValueError(
            f"source_ds 缺少 SR 引用/上下文必需属性: {absent}"
            "（ComprehensiveSR 从 evidence 继承患者与检查信息）"
        )
    sop_class_uid = str(source_ds.SOPClassUID)
    sop_instance_uid = str(source_ds.SOPInstanceUID)

    instances = measurements.get("instances") or []
    if not isinstance(instances, (list, tuple)):
        raise ValueError("measurements['instances'] 必须是 list[dict]")
    for item in instances[:MAX_INSTANCE_MEASUREMENTS]:
        if not isinstance(item, dict):
            raise ValueError("measurements['instances'] 的每条记录必须是 dict")

    source_image_item = SourceImageForMeasurementGroup(
        referenced_sop_class_uid=sop_class_uid,
        referenced_sop_instance_uid=sop_instance_uid,
    )

    def _measurement(identifier: str, value: float, unit: tuple[str, str, str]):
        unit_value, unit_scheme, unit_meaning = unit
        name = Code(
            identifier[:16],
            _MEASUREMENT_SCHEME,
            _MEASUREMENT_MEANINGS.get(identifier, str(identifier)),
        )
        return Measurement(
            name=name,
            value=value,
            unit=Code(unit_value, unit_scheme, unit_meaning),
            tracking_identifier=TrackingIdentifier(
                uid=generate_uid(), identifier=identifier[:16]
            ),
        )

    # 3) 汇总测量组：ga_percent / instance_count / meiboscore（+ 其他数值标量）
    summary_measurements = [
        _measurement("ga_percent", ga_percent, ("%", "UCUM", "percent")),
        _measurement(
            "instance_count", float(int(instance_count)), ("1", "UCUM", "unity")
        ),
        _measurement(
            "meiboscore", float(int(meiboscore)), ("1", "UCUM", "unity")
        ),
    ]
    for key, value in measurements.items():
        if key in _REQUIRED_KEYS or key == "instances":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue  # 仅数值标量生成 Measurement（列表/字典/字符串跳过）
        number = _finite(value, key)
        identifier, unit_value, unit_meaning = _TRACK_BY_KEY.get(
            key, (_slug(key), "1", "unity")
        )
        summary_measurements.append(
            _measurement(identifier, number, (unit_value, "UCUM", unit_meaning))
        )

    groups = [
        MeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(
                uid=generate_uid(), identifier="mgd_summary"
            ),
            measurements=summary_measurements,
            source_images=[source_image_item],
        )
    ]

    # 4) 每实例表：逐实例一个测量组，截断到前 20 条
    #    （"truncated_to": 20 —— 实例组数量上限即体现截断）
    for idx, item in enumerate(instances[:MAX_INSTANCE_MEASUREMENTS], start=1):
        instance_measurements = []
        for key, value in item.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = _finite(value, f"instances[{key}]")
            identifier, unit_value, unit_meaning = _TRACK_BY_KEY.get(
                key, (_slug(key), "1", "unity")
            )
            instance_measurements.append(
                _measurement(identifier, number, (unit_value, "UCUM", unit_meaning))
            )
        if instance_measurements:
            groups.append(
                MeasurementsAndQualitativeEvaluations(
                    tracking_identifier=TrackingIdentifier(
                        uid=generate_uid(), identifier=f"mgd_inst_{idx}"[:16]
                    ),
                    measurements=instance_measurements,
                    source_images=[source_image_item],
                )
            )

    # 5) TID 1500 内容树（根容器缺省 title 即 Measurement Report 根码 +
    #    template_id 1500；观察上下文 + 检查项目 + "Imaging Measurements"）
    observation_context = ObservationContext(
        observer_person_context=ObserverContext(
            observer_type=codes.cid270.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                # Person Name VR 的三段式（family^given^middle），避免
                # pydicom 对单组件人名的合法性告警
                name="MGD^Analysis^Pipeline"
            ),
        )
    )
    report = MeasurementReport(
        observation_context=observation_context,
        procedure_reported=Code("P0-0099E", "SRT", "Imaging procedure"),
        imaging_measurements=groups,
    )

    # 6) 文档封套：ComprehensiveSR 从 evidence[0] 继承患者/检查上下文并
    #    构建 CurrentRequestedProcedureEvidenceSequence；患者属性缺失已
    #    在第 2 步用 ValueError 挡住（空值允许，缺元素不允许）。
    now = datetime.now(timezone.utc)
    # SeriesDescription VR LO ≤64 字符：截断标记 inst=shown/total 压在最前，
    # 尾部再 [:64] 兜底（模型名过长也不至于写出非法 LO）。
    series_description = (
        f"MGD SR ga={ga_percent:.1f} n={int(instance_count)}"
        f" ms={int(meiboscore)}"
        f" instances={min(len(instances), MAX_INSTANCE_MEASUREMENTS)}/{len(instances)}"
    )
    if provenance is not None:
        series_description += (
            f" model={getattr(provenance, 'model_name', 'unknown')}"
            f" thr={getattr(provenance, 'threshold', 'unknown')}"
        )
    series_description = series_description[:64]
    software_version = "unknown"
    if provenance is not None:
        software_version = str(
            getattr(provenance, "software_version", None) or "unknown"
        )

    document = ComprehensiveSR(
        evidence=[source_ds],
        content=report,
        series_instance_uid=generate_uid(),
        series_number=int(getattr(source_ds, "SeriesNumber", 0) or 0) + 1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="mgd-seg",
        manufacturer_model_name="mgd-seg",
        software_versions=software_version,
        series_description=series_description,
        is_complete=True,  # CompletionFlag=COMPLETE；is_final=False → PRELIMINARY
        specific_character_set="ISO_IR 192",  # UTF-8，容纳中文列名
        content_date=now.date(),
        content_time=now.time(),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # pydicom >= 3：enforce_file_format=True 确保写出带 preamble 的合规 Part 10 文件
        document.save_as(out_path, enforce_file_format=True)
    except TypeError:  # pragma: no cover - pydicom 2.x 使用旧参数名
        document.save_as(out_path, write_like_original=False)
    return out_path
