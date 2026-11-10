"""DICOM / FHIR 互操作导出（WP-9）。

- `dicom_seg` / `dicom_sr`：pydicom + highdicom 为可选依赖（pyproject 的
  ``mgd-seg[dicom]`` extra），在函数体内延迟导入；未安装时抛
  :class:`~mgd.core.runtime.errors.InteropDependencyError`（消息含安装命令），
  绝不静默吞掉导入失败。
- `fhir`：纯标准库 JSON 构造，零外部依赖、零网络。
"""
from .dicom_seg import export_seg
from .dicom_sr import MAX_INSTANCE_MEASUREMENTS, export_sr
from .fhir import (
    LOINC_SYSTEM,
    UCUM_SYSTEM,
    to_fhir_diagnostic_report,
    to_fhir_observation,
)

__all__ = [
    "export_seg",
    "export_sr",
    "to_fhir_observation",
    "to_fhir_diagnostic_report",
    "LOINC_SYSTEM",
    "UCUM_SYSTEM",
    "MAX_INSTANCE_MEASUREMENTS",
]
