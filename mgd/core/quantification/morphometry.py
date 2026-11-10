"""形态学与信号量化指标（WP-1）。

全部为纯 numpy / opencv 计算，输入二维 uint8 掩膜 + 可选灰度图。
约定：**空输入一律返回 float("nan")（width_histogram 返回空直方图 dict），
返回前用 logger.debug 记原因；绝不抛异常、绝不用 0 冒充**。

骨架、主路径与曲率复用 ``mgd.core.imaging.clinical`` 的既有实现
（``_skeleton``：腐蚀不再改变即终止的防死循环骨架；``_main_path``：两次 BFS
近似测地直径的有序点序；``_curvature``：平均曲率 + P95 截断口径），本模块
不重复造轮子。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..imaging.clinical import _curvature, _main_path, _skeleton
from ..runtime.logging import get_logger
from .topology import _as_binary2d, _neighbor_counts

logger = get_logger("quantification.morphometry")


# ---------------------------------------------------------------------------
# 内部助手
# ---------------------------------------------------------------------------
def _as_gray2d(gray: object) -> np.ndarray | None:
    """把灰度输入规整为二维 float64 数组；非法输入返回 None 并记 debug 日志。

    与 _as_binary2d 同口径：转换全程捕获 TypeError / ValueError，
    绝不向上抛异常（上游统一走 None → NaN 通道）。
    """
    if gray is None:
        logger.debug("_as_gray2d：灰度图输入为 None")
        return None
    try:
        arr = np.asarray(gray)
    except (TypeError, ValueError) as exc:
        logger.debug("_as_gray2d：灰度图无法转为数组（%s）→ None", exc)
        return None
    if arr.ndim != 2:
        logger.debug("_as_gray2d：灰度图必须二维，收到 shape=%s", arr.shape)
        return None
    try:
        return arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        logger.debug("_as_gray2d：灰度图元素无法转为 float（%s）→ None", exc)
        return None


def _as_path(path: object) -> np.ndarray | None:
    """把点序输入规整为 (N,2) float64 数组（仅取前两列）；非法返回 None。

    与 _as_binary2d 同口径：转换全程捕获 TypeError / ValueError
    （字符串、ragged 序列、dict 等非法输入 → None），绝不抛异常。
    """
    if path is None:
        return None
    try:
        pts = np.asarray(path, dtype=float)
    except (TypeError, ValueError) as exc:
        logger.debug("_as_path：点序输入无法转为数值数组（%s）→ None", exc)
        return None
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] == 0:
        return None
    return pts[:, :2]


def _prune_spur_branches(skeleton: np.ndarray, max_spur_len: int) -> np.ndarray:
    """骨架短刺剪枝（教科书式后处理，不改变骨架算法本身）。

    从每个端点（1 邻居点）沿骨架向内走；若 ≤ max_spur_len 步内遇到分叉点
    （≥3 邻居），则该段为短刺 → 整段删除（保留分叉点）；若先到达另一端点
    （简单曲线 / 主干）则完整保留。clinical._skeleton 对实心矩形等凸形状会在
    四角产生对角刺，剪枝后宽度 / 弯曲度才反映腺体主体而非角部伪影。

    最多迭代 8 轮以处理嵌套短刺；输入骨架为空时原样返回。
    """
    skel = (skeleton > 0).astype(np.uint8)
    height, width = skel.shape
    for _round in range(8):
        if int(skel.sum()) == 0:
            break
        counts = _neighbor_counts(skel)
        endpoints = np.argwhere((skel > 0) & (counts == 1))
        if endpoints.size == 0:
            break
        removed_any = False
        for ey, ex in endpoints:
            if skel[ey, ex] == 0:
                continue  # 本轮已被其他短刺的删除波及
            chain = [(int(ey), int(ex))]
            cur = (int(ey), int(ex))
            prev: tuple[int, int] | None = None
            is_spur = False
            for _step in range(max(1, int(max_spur_len))):
                if int(counts[cur]) >= 3:
                    is_spur = True  # 走到分叉点：整段为短刺
                    break
                nxt: tuple[int, int] | None = None
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cur[0] + dy, cur[1] + dx
                        if (
                            0 <= ny < height and 0 <= nx < width
                            and skel[ny, nx] > 0 and (ny, nx) != prev
                        ):
                            nxt = (ny, nx)
                            break
                    if nxt is not None:
                        break
                if nxt is None:
                    break  # 先到达另一端点：简单曲线 / 主干 → 保留
                prev, cur = cur, nxt
                chain.append(cur)
            if is_spur:
                for py, px in chain[:-1]:  # 分叉点本身保留
                    skel[py, px] = 0
                removed_any = True
        if not removed_any:
            break
    return skel


def _skeleton_path(mask: np.ndarray) -> np.ndarray:
    """去刺骨架的主路径 (N,2) 点序（复用 clinical._skeleton + _main_path）。

    先用 :func:`_prune_spur_branches` 去除角部 / 噪声短刺（剪枝步长上限随骨架
    规模自适应：max(4, min(12, 骨架像素数 // 8))），再取主路径；
    无骨架点返回 (0,2)。
    """
    skeleton = _skeleton(mask)
    if int(skeleton.sum()) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    spur_budget = max(4, min(12, int(skeleton.sum()) // 8))
    pruned = _prune_spur_branches(skeleton, spur_budget)
    if int(pruned.sum()) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(_main_path(pruned), dtype=np.float64)


def _ring_mask(mask: np.ndarray, ring_px: int) -> np.ndarray:
    """腺体掩膜的环形背景带：椭圆核膨胀 ring_px 像素后减去腺体本身。"""
    size = 2 * max(1, int(ring_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask, kernel)
    return ((dilated > 0) & (mask == 0)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 面积类
# ---------------------------------------------------------------------------
def gland_area_ratio(gland: np.ndarray | None, eyelid: np.ndarray | None) -> float:
    """腺体面积比 GA（%）。

    定义：GA = 100 × |G∩E| / |E|，G 为腺体掩膜、E 为眼睑 ROI 掩膜
    （Arita 2008 腺体缺失率的补数口径）。

    空输入行为：眼睑缺失 / 为空 → NaN（无分母）；腺体缺失 / 为空 → NaN
    （「无腺体可评估」与「0% 覆盖」不可混同，不用 0 冒充）。
    两者尺寸不符时眼睑掩膜最近邻缩放到腺体尺寸。
    """
    g = _as_binary2d(gland)
    e = _as_binary2d(eyelid, g.shape if g is not None else None)
    if e is None or int(e.sum()) == 0:
        logger.debug("gland_area_ratio：眼睑 ROI 缺失或为空，GA 无分母 → NaN")
        return float("nan")
    if g is None or int(g.sum()) == 0:
        logger.debug("gland_area_ratio：腺体掩膜缺失或为空，无法评估 → NaN")
        return float("nan")
    intersection = int(np.logical_and(g > 0, e > 0).sum())
    return float(100.0 * intersection / int(e.sum()))


# ---------------------------------------------------------------------------
# 宽度类
# ---------------------------------------------------------------------------
def width_profile(mask: np.ndarray | None) -> np.ndarray | float:
    """骨架点序上的局部宽度序列（px）。

    定义：对掩膜做距离变换 ``cv2.distanceTransform(mask, DIST_L2, 3)``，
    在去刺骨架（复用 clinical._skeleton 提取 + 短刺剪枝去除角部 / 噪声伪影）
    主路径点序的每个点上取距离变换值 × 2，得到沿骨架点序排列的宽度数组。

    空输入行为：掩膜为空 / 非法，或骨架为空（如掩膜铺满全图时腐蚀不变）
    → 返回 float("nan")（先记 debug 日志）；正常情况返回 (N,) float64 数组。
    """
    m = _as_binary2d(mask)
    if m is None or int(m.sum()) == 0:
        logger.debug("width_profile：掩膜为空或非法 → NaN")
        return float("nan")
    path = _skeleton_path(m)
    if path.shape[0] == 0:
        logger.debug("width_profile：骨架为空（如整幅掩膜腐蚀不变）→ NaN")
        return float("nan")
    distance = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    ys = np.clip(np.round(path[:, 0]).astype(int), 0, m.shape[0] - 1)
    xs = np.clip(np.round(path[:, 1]).astype(int), 0, m.shape[1] - 1)
    return (2.0 * distance[ys, xs]).astype(np.float64)


def diameter_deformation_index(widths: object) -> float:
    """管径变形指数 DI = std(w) / mean(w)。

    w 为宽度序列（width_profile 输出）；衡量腺体粗细不均程度。

    空输入行为：有效样本数 < 5，或 mean(w) 非正 / 非有限 → NaN
    （先记 debug 日志），不抛异常。
    """
    arr = np.asarray(widths, dtype=float).ravel() if widths is not None else np.zeros(0)
    if arr.size < 5:
        logger.debug("diameter_deformation_index：有效宽度样本 %d < 5 → NaN", arr.size)
        return float("nan")
    mean = float(np.mean(arr))
    if not np.isfinite(mean) or mean <= 0.0:
        logger.debug("diameter_deformation_index：mean(w)=%r 非法 → NaN", mean)
        return float("nan")
    return float(np.std(arr) / mean)


def width_histogram(widths: object, bins: int = 20) -> dict:
    """宽度直方图。

    定义：对有限宽度样本做 ``np.histogram``，返回
    ``{"edges": [分箱边界...], "counts": [各箱计数...]}``。

    空输入行为：无有限样本（空 / NaN）→ 返回 ``{"edges": [], "counts": []}``
    （先记 debug 日志），不抛异常。
    """
    arr = np.asarray(widths, dtype=float).ravel() if widths is not None else np.zeros(0)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        logger.debug("width_histogram：无有效宽度样本，返回空直方图")
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(arr, bins=max(1, int(bins)))
    return {
        "edges": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
    }


# ---------------------------------------------------------------------------
# 弯曲 / 拐点类（输入均为 (N,2) 点序 path）
# ---------------------------------------------------------------------------
def tortuosity_arc_chord(path: object) -> float:
    """弯曲度 τ = 弧长 / 弦长。

    path 为 (N,2) 点序：L_arc = Σ 相邻点欧氏距离，L_chord = 首末端点直线距离；
    直线 τ = 1。

    空输入行为：点数 < 2、维度非法或弦长为 0（含所有点重合的退化 path）
    → NaN（先记 debug 日志），不抛异常。
    """
    pts = _as_path(path)
    if pts is None or pts.shape[0] < 2:
        logger.debug("tortuosity_arc_chord：path 点数 < 2 或非法 → NaN")
        return float("nan")
    arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if not np.isfinite(arc) or not np.isfinite(chord) or chord <= 1e-9:
        logger.debug("tortuosity_arc_chord：退化 path（arc=%r, chord=%r）→ NaN", arc, chord)
        return float("nan")
    return arc / chord


def tortuosity_index(path: object) -> float:
    """弯曲指数 TI = τ − 1（τ 为 :func:`tortuosity_arc_chord`）。

    TI = 0 表示直线，越大越弯曲。
    空输入行为：τ 为 NaN（含退化 path）→ NaN，不抛异常。
    """
    tau = tortuosity_arc_chord(path)
    if not np.isfinite(tau):
        logger.debug("tortuosity_index：τ 非有限 → NaN")
        return float("nan")
    return float(tau - 1.0)


def tortuosity_curvature(path: object) -> tuple[float, float]:
    """骨架路径曲率 (mean_abs_kappa, p95_abs_kappa)。

    复用 clinical._curvature 实现（端点 15% 截断 + 滑动平滑 + P95 口径，
    单位 1/px）。
    空输入行为：点数不足（< 8）或退化 path → (NaN, NaN)，不抛异常。
    """
    pts = _as_path(path)
    if pts is None or pts.shape[0] < 2:
        logger.debug("tortuosity_curvature：path 点数 < 2 或非法 → (NaN, NaN)")
        return float("nan"), float("nan")
    mean_kappa, p95_kappa = _curvature(pts)
    return float(mean_kappa), float(p95_kappa)


def inflection_count(path: object, min_kappa: float = 0.05) -> float:
    """曲率拐点数。

    定义：对相邻三点 (p[i-1], p[i], p[i+1]) 计算有符号 Menger 曲率
    κ = 2·cross(a, b) / (|a|·|b|·|a+b|)；将曲率序列压缩为连续同号段
    （|κ|≈0 的点跳过、不打断相邻关系），统计符号相反的相邻段之间的转折次数，
    且转折两侧段的最大 |κ| 均需 > min_kappa（单位 1/px，默认 0.05）。
    该口径使平滑曲线在 κ 过零处（|κ| 很小）不会被漏计，同时过滤微幅抖动。

    空输入行为：点数 < 3 → NaN；直线（无显著同号段）→ 0（合法测量值）。
    不抛异常。
    """
    pts = _as_path(path)
    if pts is None or pts.shape[0] < 3:
        logger.debug("inflection_count：path 点数 < 3，无法构成三点曲率 → NaN")
        return float("nan")
    a = pts[1:-1] - pts[:-2]  # 入段向量 (N-2, 2)
    b = pts[2:] - pts[1:-1]   # 出段向量
    c = pts[2:] - pts[:-2]    # 跨段向量
    len_a = np.linalg.norm(a, axis=1)
    len_b = np.linalg.norm(b, axis=1)
    len_c = np.linalg.norm(c, axis=1)
    denom = len_a * len_b * len_c
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    valid = denom > 1e-12
    kappa = np.zeros(denom.shape, dtype=float)
    kappa[valid] = 2.0 * cross[valid] / denom[valid]

    runs: list[tuple[int, float]] = []  # (符号, 段内最大 |κ|)
    for sign, magnitude in zip(np.sign(kappa), np.abs(kappa)):
        if sign == 0:
            continue
        if runs and runs[-1][0] == sign:
            runs[-1] = (int(sign), max(runs[-1][1], float(magnitude)))
        else:
            runs.append((int(sign), float(magnitude)))
    count = sum(
        1
        for (_, k1), (_, k2) in zip(runs, runs[1:])
        if k1 > min_kappa and k2 > min_kappa
    )
    return int(count)


# ---------------------------------------------------------------------------
# 纹理 / 信号类
# ---------------------------------------------------------------------------
def signal_index(gray: np.ndarray | None, mask: np.ndarray | None, ring_px: int = 10) -> float:
    """信号指数 SI = ( mean(I[G]) − mean(I[R]) ) / 255。

    G 为腺体掩膜，R 为环形背景带（椭圆核膨胀 ring_px 像素后减去腺体本身），
    I 为灰度图；SI ∈ [-1, 1]，越大表示腺体相对背景越亮。

    空输入行为：灰度图 / 掩膜为空，或环为空（如掩膜覆盖全图、ring_px 过小）
    → NaN（先记 debug 日志），不抛异常。
    """
    value = _gland_ring_stats(gray, mask, ring_px)
    if value is None:
        return float("nan")
    gland_mean, ring_mean = value
    return float((gland_mean - ring_mean) / 255.0)


def local_contrast(gray: np.ndarray | None, mask: np.ndarray | None, ring_px: int = 10) -> float:
    """局部对比度 LC = mean(I[G]) − mean(I[R])（原始灰度差，不除 255）。

    环带定义同 :func:`signal_index`。
    空输入行为同 :func:`signal_index`：空 / 退化输入 → NaN，不抛异常。
    """
    value = _gland_ring_stats(gray, mask, ring_px)
    if value is None:
        return float("nan")
    gland_mean, ring_mean = value
    return float(gland_mean - ring_mean)


def _gland_ring_stats(
    gray: np.ndarray | None,
    mask: np.ndarray | None,
    ring_px: int,
) -> tuple[float, float] | None:
    """signal_index / local_contrast 共用：返回 (腺体灰度均值, 环带灰度均值)。"""
    image = _as_gray2d(gray)
    m = _as_binary2d(mask, image.shape if image is not None else None)
    if image is None:
        logger.debug("signal/local_contrast：灰度图为空或非法 → NaN")
        return None
    if m is None or int(m.sum()) == 0:
        logger.debug("signal/local_contrast：腺体掩膜为空或非法 → NaN")
        return None
    ring = _ring_mask(m, ring_px)
    if int(ring.sum()) == 0:
        logger.debug("signal/local_contrast：环形背景带为空（掩膜覆盖全图或 ring_px 过小）→ NaN")
        return None
    gland_mean = float(image[m > 0].mean())
    ring_mean = float(image[ring > 0].mean())
    return gland_mean, ring_mean


# ---------------------------------------------------------------------------
# 形状 / 距离类
# ---------------------------------------------------------------------------
def fractal_dimension(mask: np.ndarray | None) -> float:
    """盒计数分形维数 D。

    定义：N(ε) 为边长 ε 盒子覆盖前景的个数，ε 取 2 的幂递减
    （2 ≤ ε ≤ min(H,W)/2）；对 log N(ε) ~ log(1/ε) 做 np.polyfit 一次线性
    拟合，斜率即 D（实心矩形 ≈ 2，细线 ≈ 1）。

    空输入行为：掩膜为空 / 非法，或 ε 序列不足两点（图像过小）→ NaN
    （先记 debug 日志），不抛异常。
    """
    m = _as_binary2d(mask)
    if m is None or int(m.sum()) == 0:
        logger.debug("fractal_dimension：掩膜为空或非法 → NaN")
        return float("nan")
    height, width = m.shape
    eps_max = max(2, min(height, width) // 2)
    sizes: list[int] = []
    eps = 2
    while eps <= eps_max:
        sizes.append(eps)
        eps *= 2
    if len(sizes) < 2:
        logger.debug("fractal_dimension：ε 序列仅 %d 点（图像过小）→ NaN", len(sizes))
        return float("nan")
    counts: list[float] = []
    for step in sizes:
        cropped_h = (height // step) * step
        cropped_w = (width // step) * step
        blocks = m[:cropped_h, :cropped_w].reshape(
            cropped_h // step, step, cropped_w // step, step
        )
        occupied = int(blocks.any(axis=(1, 3)).sum())
        counts.append(float(max(occupied, 1)))
    log_inv_eps = np.log2(1.0 / np.asarray(sizes, dtype=float))
    log_counts = np.log2(np.asarray(counts, dtype=float))
    slope = float(np.polyfit(log_inv_eps, log_counts, 1)[0])
    return slope


def gland_eyelid_distance(mask: np.ndarray | None, eyelid_mask: np.ndarray | None) -> float:
    """腺体骨架端点到眼睑外接框垂直中线的平均横向距离（px）。

    定义：端点为去刺骨架（复用 clinical._skeleton + 短刺剪枝，避免实心形状
    四角对角伪影被误计为端点）中 3×3 邻域（不含自身）恰有 1 个骨架邻居的点；
    垂直中线 x_c = (x_min + x_max) / 2 由眼睑外接框给出；
    距离 = mean(|x_endpoint − x_c|)。

    空输入行为：眼睑缺失 / 为空、腺体为空、骨架为空或无端点
    （如闭合环状骨架）→ NaN（先记 debug 日志），不抛异常。
    """
    m = _as_binary2d(mask)
    e = _as_binary2d(eyelid_mask, m.shape if m is not None else None)
    if m is None or int(m.sum()) == 0:
        logger.debug("gland_eyelid_distance：腺体掩膜为空或非法 → NaN")
        return float("nan")
    if e is None or int(e.sum()) == 0:
        logger.debug("gland_eyelid_distance：眼睑掩膜为空或非法 → NaN")
        return float("nan")
    skeleton = _skeleton(m)
    if int(skeleton.sum()) == 0:
        logger.debug("gland_eyelid_distance：腺体骨架为空 → NaN")
        return float("nan")
    spur_budget = max(4, min(12, int(skeleton.sum()) // 8))
    skeleton = _prune_spur_branches(skeleton, spur_budget)
    if int(skeleton.sum()) == 0:
        logger.debug("gland_eyelid_distance：去刺后骨架为空 → NaN")
        return float("nan")
    counts = _neighbor_counts(skeleton)
    endpoint_ys, endpoint_xs = np.nonzero((skeleton > 0) & (counts == 1))
    if endpoint_xs.size == 0:
        logger.debug("gland_eyelid_distance：骨架无端点（如双列宽骨架）→ NaN")
        return float("nan")
    _, eyelid_xs = np.nonzero(e)
    centerline_x = (float(eyelid_xs.min()) + float(eyelid_xs.max())) / 2.0
    return float(np.mean(np.abs(endpoint_xs.astype(float) - centerline_x)))
