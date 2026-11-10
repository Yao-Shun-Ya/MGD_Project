"""WP-1 可追溯量化指标层。

组成：
- :mod:`.registry`：指标元数据注册表（公式 / 单位 / 出处可追溯）；
- :mod:`.morphometry`：面积、宽度、弯曲、纹理 / 信号类指标；
- :mod:`.topology`：骨架端点 / 分叉、断裂数、实例切分（cc / watershed）；
- :mod:`.zoning`：眼睑分区密度与左右对称指数；
- :mod:`.scale`：像素-毫米物理标定解析与换算。

统一约定：空输入一律返回 NaN（或空容器）并记 debug 日志，绝不抛异常、
绝不用 0 冒充测量值；骨架 / 曲率复用 ``mgd.core.imaging.clinical`` 既有实现。
"""
from __future__ import annotations

from .morphometry import (
    diameter_deformation_index,
    fractal_dimension,
    gland_area_ratio,
    gland_eyelid_distance,
    inflection_count,
    local_contrast,
    signal_index,
    tortuosity_arc_chord,
    tortuosity_curvature,
    tortuosity_index,
    width_histogram,
    width_profile,
)
from .registry import (
    ALLOWED_SOURCES,
    METRIC_REGISTRY,
    MetricSpec,
    all_specs,
    describe,
    get_spec,
    register,
)
from .scale import UN_CALIBRATED_NOTE, resolve_pixel_scale, to_mm
from .topology import break_count, endpoints_and_branches, split_instances
from .zoning import symmetry_index, zone_edges, zonal_density

__all__ = [
    # registry
    "ALLOWED_SOURCES",
    "METRIC_REGISTRY",
    "MetricSpec",
    "all_specs",
    "describe",
    "get_spec",
    "register",
    # morphometry
    "diameter_deformation_index",
    "fractal_dimension",
    "gland_area_ratio",
    "gland_eyelid_distance",
    "inflection_count",
    "local_contrast",
    "signal_index",
    "tortuosity_arc_chord",
    "tortuosity_curvature",
    "tortuosity_index",
    "width_histogram",
    "width_profile",
    # topology
    "break_count",
    "endpoints_and_branches",
    "split_instances",
    # zoning
    "symmetry_index",
    "zone_edges",
    "zonal_density",
    # scale
    "UN_CALIBRATED_NOTE",
    "resolve_pixel_scale",
    "to_mm",
]
