"""WP-10 医生修正捕获与回流（只做记录与导出，不做训练）。

组成：
- :mod:`.corrections`：:class:`CorrectionRecord` 修正记录结构、
  :func:`record_correction` 捕获单条修正、:func:`summarize` 汇总统计、
  :func:`export_candidates` 导出训练候选清单（JSONL、原子写）；
  以及持久化接线：:func:`append_record` 追加落盘到
  ``settings.home / "db" / "corrections.jsonl"``（deid_salt 设置时 case_id
  自动假名化）、:func:`load_records` 容错读取、:func:`summarize_file` /
  :func:`export_from_file` 在文件上做同口径汇总与导出。

范围约定：本包只负责**记录与导出**——不做任何训练、不做模型更新；
掩膜像素不落盘（行内只带 sha256），掩膜文件持久化与训练回流属后续工作。

口径约定：Dice 等指标在本包内独立实现（与
:mod:`mgd.core.imaging.metrics` 同一口径，但不 import 其私有函数）。
"""
from __future__ import annotations

from .corrections import (
    CorrectionRecord,
    append_record,
    corrections_path,
    export_candidates,
    export_from_file,
    load_records,
    record_correction,
    record_to_row,
    summarize,
    summarize_file,
)

__all__ = [
    "CorrectionRecord",
    "append_record",
    "corrections_path",
    "export_candidates",
    "export_from_file",
    "load_records",
    "record_correction",
    "record_to_row",
    "summarize",
    "summarize_file",
]
