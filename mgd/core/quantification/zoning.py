"""眼睑分区与左右对称指标（WP-1）。

分区口径：沿眼睑外接框宽度等分为 zones 个竖直条带 Z_k，
ρ_k = 100 × |G∩E∩Z_k| / |E∩Z_k|。
空输入一律返回 NaN（zone_edges 返回空列表）并记 debug 日志，绝不抛异常。
"""
from __future__ import annotations

import numpy as np

from ..runtime.logging import get_logger
from .topology import _as_binary2d

logger = get_logger("quantification.zoning")


def zone_edges(eyelid_mask: np.ndarray | None, zones: int = 3) -> list[tuple[float, float]]:
    """眼睑外接框宽度等分的分区边界 [(x0, x1), ...]，区间为左闭右开 [x0, x1)。

    定义：取眼睑前景列范围 [x_min, x_max]，分区边界为
    np.linspace(x_min, x_max + 1, zones + 1)；末区右端点为 x_max + 1，
    故所有前景列都被覆盖。

    空输入行为：眼睑缺失 / 为空 → 返回空列表（先记 debug 日志）；
    zones < 1 抛 ValueError（调用方编码错误，非数据问题）。
    """
    if int(zones) < 1:
        raise ValueError(f"zones 必须为正整数，收到 {zones!r}")
    e = _as_binary2d(eyelid_mask)
    if e is None or int(e.sum()) == 0:
        logger.debug("zone_edges：眼睑掩膜为空或非法，返回空列表")
        return []
    _, eyelid_xs = np.nonzero(e)
    x0 = float(eyelid_xs.min())
    x1 = float(eyelid_xs.max()) + 1.0
    edges = np.linspace(x0, x1, int(zones) + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(int(zones))]


def zonal_density(
    gland: np.ndarray | None,
    eyelid: np.ndarray | None,
    zones: int = 3,
) -> list[float]:
    """分区腺体密度 [ρ_1, ..., ρ_zones]（%）。

    定义：ρ_k = 100 × |G∩E∩Z_k| / |E∩Z_k|，Z_k 为 :func:`zone_edges`
    的第 k 个分区（竖直条带）。

    空输入行为：腺体或眼睑缺失 / 为空 → 返回 [NaN] × zones；
    某分区眼睑为空（无分母）→ 该区 NaN，其余区正常计算。不抛异常。
    """
    n = int(zones)
    if n < 1:
        raise ValueError(f"zones 必须为正整数，收到 {zones!r}")
    g = _as_binary2d(gland)
    e = _as_binary2d(eyelid, g.shape if g is not None else None)
    if g is None or e is None or int(g.sum()) == 0 or int(e.sum()) == 0:
        logger.debug("zonal_density：腺体或眼睑掩膜为空或非法 → [NaN] × %d", n)
        return [float("nan")] * n

    gland_bool = g > 0
    eyelid_bool = e > 0
    columns = np.arange(e.shape[1], dtype=float)
    densities: list[float] = []
    for x0, x1 in zone_edges(e, n):
        in_zone = (columns >= x0) & (columns < x1)
        eyelid_zone = eyelid_bool & in_zone[np.newaxis, :]
        area_eyelid = int(eyelid_zone.sum())
        if area_eyelid == 0:
            densities.append(float("nan"))
            continue
        gland_zone = gland_bool & in_zone[np.newaxis, :]
        intersection = int(np.logical_and(gland_zone, eyelid_zone).sum())
        densities.append(float(100.0 * intersection / area_eyelid))
    return densities


def symmetry_index(left_value: object, right_value: object) -> float:
    """左右对称指数 SI = |a − b| / mean(a, b) = 2|a − b| / (a + b)。

    对左右眼同名指标计算相对差异；SI = 0 完全对称，越大越不对称。
    **仅适用于非负指标（密度 / 百分比等）**：mean(a, b) ≤ 0 或非有限
    （含两负值、一正一负拉平后仍 ≤ 0、数值溢出）时相对差异无临床意义，
    一律返回 NaN。

    空输入行为：任一侧为 None / NaN，或 mean(a, b) ≤ 0 / 非有限 → NaN
    （先记 debug 日志），不抛异常。
    """
    if left_value is None or right_value is None:
        logger.debug("symmetry_index：输入含 None → NaN")
        return float("nan")
    a = float(left_value)
    b = float(right_value)
    if not (np.isfinite(a) and np.isfinite(b)):
        logger.debug("symmetry_index：输入含非有限值（a=%r, b=%r）→ NaN", a, b)
        return float("nan")
    mean = (a + b) / 2.0
    if not np.isfinite(mean) or mean <= 0.0:
        logger.debug(
            "symmetry_index：mean(a, b) = %r 非正或非有限（本指标仅适用于非负输入），→ NaN",
            mean,
        )
        return float("nan")
    return float(abs(a - b) / mean)
