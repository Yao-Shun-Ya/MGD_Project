"""物理标定与像素-毫米换算（WP-1）。

口径链（优先级从高到低）：
1. DICOM 元数据 ``PixelSpacing: [row, col]``（逐例优先，最权威）；
2. ``Settings.physical_scale_mm_per_px``（全局手工标定）；
3. 均缺失 → ``(None, "未标定：长度类指标为像素单位，不可跨设备比较")``。

**禁止用默认 1.0 冒充标定**：未标定时长度类指标保持 px 单位并如实告知。
"""
from __future__ import annotations

import numpy as np

from ..runtime.logging import get_logger

logger = get_logger("quantification.scale")

#: 未标定时的固定说明文案（UI / 报告直接展示）
UN_CALIBRATED_NOTE = "未标定：长度类指标为像素单位，不可跨设备比较"


def resolve_pixel_scale(
    settings: object | None,
    dicom_meta: dict | None = None,
) -> tuple[float | None, str]:
    """解析当前例的像素物理尺度 (mm_per_px, source)。

    - ``dicom_meta`` 为 dict 且含合法 ``PixelSpacing: [row, col]``（均为正有限数）
      时优先采用；各向异性间距取算术平均 (row + col) / 2 并记 debug 日志；
      source = ``"dicom:PixelSpacing"``。
    - 否则回退 ``settings.physical_scale_mm_per_px``（正有限数才有效）；
      source = ``"settings:physical_scale_mm_per_px"``。
    - 均不可用 → ``(None, UN_CALIBRATED_NOTE)``，不抛异常。

    返回值中的 ``None`` 表示未标定，调用方必须把长度类指标标为 px 口径，
    禁止默认 1.0 冒充。
    """
    if isinstance(dicom_meta, dict):
        spacing = dicom_meta.get("PixelSpacing")
        if spacing is not None:
            try:
                row_mm = float(spacing[0])
                col_mm = float(spacing[1])
            except (TypeError, ValueError, IndexError, KeyError):
                logger.debug("resolve_pixel_scale：PixelSpacing=%r 无法解析，跳过 DICOM 来源", spacing)
            else:
                if np.isfinite(row_mm) and np.isfinite(col_mm) and row_mm > 0 and col_mm > 0:
                    if row_mm != col_mm:
                        logger.debug(
                            "resolve_pixel_scale：PixelSpacing 各向异性 (row=%r, col=%r)，取算术平均",
                            row_mm, col_mm,
                        )
                    return float((row_mm + col_mm) / 2.0), "dicom:PixelSpacing"
                logger.debug("resolve_pixel_scale：PixelSpacing=%r 非正有限数，跳过 DICOM 来源", spacing)

    if settings is not None:
        raw = getattr(settings, "physical_scale_mm_per_px", None)
        if raw is not None:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float("nan")
            if np.isfinite(value) and value > 0:
                return float(value), "settings:physical_scale_mm_per_px"
            logger.debug("resolve_pixel_scale：physical_scale_mm_per_px=%r 非法，跳过", raw)

    return None, UN_CALIBRATED_NOTE


def to_mm(value_px: object, mm_per_px: object) -> float:
    """像素值换算为毫米：mm = value_px × mm_per_px。

    长度类输入应为非负像素值：value_px < 0 → NaN（负长度无物理意义）。

    空输入行为：任一入参为 None / NaN / 非数值、mm_per_px ≤ 0、
    value_px < 0，或乘积溢出为非有限值 → NaN（先记 debug 日志），
    绝不以默认 1.0 冒充标定。不抛异常。
    """
    if value_px is None or mm_per_px is None:
        logger.debug("to_mm：输入含 None（value_px=%r, mm_per_px=%r）→ NaN", value_px, mm_per_px)
        return float("nan")
    try:
        value = float(value_px)
        scale = float(mm_per_px)
    except (TypeError, ValueError):
        logger.debug("to_mm：输入无法转为数值（%r, %r）→ NaN", value_px, mm_per_px)
        return float("nan")
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0:
        logger.debug("to_mm：非有限输入或非法标定（%r, %r）→ NaN", value, scale)
        return float("nan")
    if value < 0.0:
        logger.debug("to_mm：长度类输入应为非负（value_px=%r）→ NaN", value)
        return float("nan")
    product = value * scale
    if not np.isfinite(product):
        logger.debug("to_mm：乘积非有限（%r × %r 溢出）→ NaN", value, scale)
        return float("nan")
    return float(product)
