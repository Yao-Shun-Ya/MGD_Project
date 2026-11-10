"""WP-5：split conformal 不确定性量化与风险控制。

- calibrator：:class:`ConformalCalibrator`（split conformal 预测区间）
- risk：:func:`select_threshold_for_fnr` / :func:`select_shrink_for_fp`（分布无关风险控制）
- store：校准参数持久化（原子写 + 版本化 JSON）
- pipeline：评估报告 ↔ metric 校准的桥梁（fit/save/load/attach）
"""
from .calibrator import ConformalCalibrator, conformal_upper_quantile
from .pipeline import (
    FORMAT_VERSION,
    attach_calibration_to_report,
    fit_metric_calibration,
    load_latest_metric_calibration,
    save_metric_calibration,
)
from .risk import select_shrink_for_fp, select_threshold_for_fnr
from .store import (
    SUPPORTED_STORE_VERSION,
    calibrator_from_store,
    calibrator_to_store,
    load_calibration,
    save_calibration,
)

__all__ = [
    "ConformalCalibrator",
    "conformal_upper_quantile",
    "select_threshold_for_fnr",
    "select_shrink_for_fp",
    "SUPPORTED_STORE_VERSION",
    "calibrator_from_store",
    "calibrator_to_store",
    "load_calibration",
    "save_calibration",
    "FORMAT_VERSION",
    "attach_calibration_to_report",
    "fit_metric_calibration",
    "load_latest_metric_calibration",
    "save_metric_calibration",
]
