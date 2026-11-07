"""真实临床形态学指标（Phase 5）。

修复旧版：不再用 640×1280 全图作分母；缺失率以眼睑 ROI 为分母，
并输出 Meiboscore 0–3 级、骨架弯曲度 τ、平均/最大曲率 κ、截断率与实例明细。
空掩膜返回 NaN + 明确 warning，绝不崩溃、也不用 0 冒充。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

INSTANCE_COLUMNS = [
    "实例编号", "面积(px)", "长轴(px)", "短轴(px)",
    "弯曲度τ", "平均曲率κ", "最大曲率κ", "相对延伸比", "是否截断",
]

MEIBOSCORE_RISK = {
    0: "正常（未见明显腺体缺失）",
    1: "轻度 MGD（腺体缺失 < 1/3）",
    2: "中度 MGD（腺体缺失 1/3 ~ 2/3）",
    3: "重度腺体萎缩（腺体缺失 > 2/3）",
}


@dataclass
class RlossResult:
    r_loss_percent: float = float("nan")
    meiboscore: int = -1
    risk: str = "无法评估"
    gland_in_roi_px: int = 0
    eyelid_area_px: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ClinicalResult:
    summary: list[list[str]] = field(default_factory=list)
    instance_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=INSTANCE_COLUMNS))
    rloss: RlossResult = field(default_factory=RlossResult)
    warnings: list[str] = field(default_factory=list)


def _binary(mask: np.ndarray | None, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = (np.asarray(mask) > 0).astype(np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"掩膜必须是二维，收到 {arr.shape}")
    if shape is not None and arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr


def meiboscore_from_rloss(r_loss_percent: float) -> int:
    """0–3 级（Arita 2008 四档口径；本项目将 <5% 归为 0 级正常）。"""
    if not np.isfinite(r_loss_percent):
        return -1
    if r_loss_percent <= 5.0:
        return 0
    if r_loss_percent <= 33.3:
        return 1
    if r_loss_percent <= 66.7:
        return 2
    return 3


def r_loss(gland_mask: np.ndarray, eyelid_mask: np.ndarray | None) -> RlossResult:
    """R_loss = (1 - |gland ∩ eyelid| / |eyelid|) × 100%。"""
    gland = _binary(gland_mask)
    eyelid = _binary(eyelid_mask, gland.shape) if eyelid_mask is not None else None
    result = RlossResult()

    if eyelid is None:
        result.warnings.append("缺少眼睑 ROI 掩膜，缺失率无分母，返回 NaN")
        return result
    area = int(eyelid.sum())
    result.eyelid_area_px = area
    if area == 0:
        result.warnings.append("眼睑 ROI 为空掩膜，缺失率无定义（NaN）")
        return result

    gland_in_roi = int(np.logical_and(gland, eyelid).sum())
    result.gland_in_roi_px = gland_in_roi
    result.r_loss_percent = float((1.0 - gland_in_roi / area) * 100.0)
    result.meiboscore = meiboscore_from_rloss(result.r_loss_percent)
    result.risk = MEIBOSCORE_RISK.get(result.meiboscore, "无法评估")
    return result


def _skeleton(binary: np.ndarray) -> np.ndarray:
    """形态学骨架；终止条件是「腐蚀不再改变图像」，避免孤立像素死循环。"""
    img = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        if cv2.countNonZero(cv2.subtract(img, eroded)) == 0:
            break
        img = eroded
    return (skeleton > 0).astype(np.uint8)


def _neighbors(point: tuple[int, int]) -> list[tuple[int, int]]:
    y, x = point
    return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def _bfs(start: tuple[int, int], points: set[tuple[int, int]]) -> tuple[dict, dict, tuple[int, int]]:
    distance = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])
    farthest = start
    while queue:
        current = queue.popleft()
        for neighbor in _neighbors(current):
            if neighbor not in points or neighbor in distance:
                continue
            step = 1.4142 if (neighbor[0] != current[0] and neighbor[1] != current[1]) else 1.0
            distance[neighbor] = distance[current] + step
            parent[neighbor] = current
            if distance[neighbor] > distance[farthest]:
                farthest = neighbor
            queue.append(neighbor)
    return distance, parent, farthest


def _main_path(skeleton: np.ndarray) -> np.ndarray:
    """骨架主路径：两次 BFS 近似测地直径，返回有序 (N,2) 坐标。"""
    points = {(int(y), int(x)) for y, x in zip(*np.nonzero(skeleton))}
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    start = min(points)
    _, _, endpoint_a = _bfs(start, points)
    _, parent, endpoint_b = _bfs(endpoint_a, points)
    ordered = []
    node: tuple[int, int] | None = endpoint_b
    while node is not None:
        ordered.append(node)
        node = parent.get(node)
    return np.array(ordered[::-1], dtype=np.float32)


def _smooth(points: np.ndarray, window: int = 7) -> np.ndarray:
    if len(points) < 3:
        return points
    if len(points) < window:
        window = max(3, len(points) if len(points) % 2 == 1 else len(points) - 1)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window
    y = np.convolve(np.pad(points[:, 0], pad, mode="edge"), kernel, mode="valid")
    x = np.convolve(np.pad(points[:, 1], pad, mode="edge"), kernel, mode="valid")
    return np.stack([y, x], axis=1)


def _curvature(points: np.ndarray) -> tuple[float, float]:
    """平均曲率与稳健最大曲率（端点 15% 截断 + P95，避免骨架端点数值尖峰）。"""
    if len(points) < 8:
        return float("nan"), float("nan")
    pts = _smooth(points)
    trim = max(2, int(len(pts) * 0.15))
    pts = pts[trim: len(pts) - trim]
    if len(pts) < 4:
        return float("nan"), float("nan")
    x, y = pts[:, 1], pts[:, 0]
    xp, yp = np.gradient(x), np.gradient(y)
    xpp, ypp = np.gradient(xp), np.gradient(yp)
    denominator = np.power(xp * xp + yp * yp, 1.5)
    valid = denominator > 1e-8
    if not np.any(valid):
        return float("nan"), float("nan")
    kappa = np.zeros_like(denominator)
    kappa[valid] = np.abs(xp[valid] * ypp[valid] - xpp[valid] * yp[valid]) / denominator[valid]
    return float(np.mean(kappa)), float(np.percentile(kappa, 95))


def _instance_row(index: int, component: np.ndarray, eyelid_height: int) -> list:
    ys, xs = np.nonzero(component)
    area = int(len(ys))
    height = int(ys.max() - ys.min() + 1) if area else 0
    width = int(xs.max() - xs.min() + 1) if area else 0

    # 主轴 / 短轴：协方差特征值
    if area >= 2:
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        coords -= coords.mean(axis=0, keepdims=True)
        cov = np.cov(coords, rowvar=False) + np.eye(2) * 1e-6
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        major = float(4.0 * np.sqrt(max(eigenvalues[0], 0.0)))
        minor = float(4.0 * np.sqrt(max(eigenvalues[1], 0.0)))
    else:
        major = minor = float("nan")

    skeleton = _skeleton(component)
    path = _main_path(skeleton)
    if len(path) >= 2:
        arc = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        chord = float(np.linalg.norm(path[-1] - path[0]))
        tau = arc / max(chord, 1.0)
    else:
        arc, tau = 0.0, float("nan")
    kappa_mean, kappa_max = _curvature(path)

    relative = float(height / eyelid_height) if eyelid_height else float("nan")
    truncated = bool(np.isfinite(relative) and relative < 0.5)

    return [
        f"#{index}",
        area,
        round(major, 2) if np.isfinite(major) else "N/A",
        round(minor, 2) if np.isfinite(minor) else "N/A",
        round(tau, 3) if np.isfinite(tau) else "N/A",
        round(kappa_mean, 4) if np.isfinite(kappa_mean) else "N/A",
        round(kappa_max, 4) if np.isfinite(kappa_max) else "N/A",
        round(relative, 3) if np.isfinite(relative) else "N/A",
        "是" if truncated else "否",
    ]


def instance_metrics(
    gland_mask: np.ndarray,
    eyelid_mask: np.ndarray | None = None,
    min_area: int = 120,
) -> pd.DataFrame:
    gland = _binary(gland_mask)
    eyelid = _binary(eyelid_mask, gland.shape) if eyelid_mask is not None else None
    if eyelid is not None and int(eyelid.sum()) > 0:
        ys, xs = np.nonzero(eyelid)
        eyelid_height = int(ys.max() - ys.min() + 1)
    else:
        eyelid_height = int(gland.shape[0])

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gland, connectivity=8)
    rows = []
    index = 0
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) < int(max(1, min_area)):
            continue
        index += 1
        rows.append(_instance_row(index, (labels == label).astype(np.uint8), eyelid_height))
    return pd.DataFrame(rows, columns=INSTANCE_COLUMNS)


def truncation_ratio(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return float("nan")
    flagged = (df["是否截断"] == "是").sum()
    return float(flagged / len(df))


def calculate_clinical_metrics(
    gland_mask: np.ndarray,
    eyelid_mask: np.ndarray | None = None,
    min_area: int = 120,
) -> ClinicalResult:
    """汇总临床指标；空掩膜返回 NaN 并给出 warning，n=0 不崩溃。"""
    gland = _binary(gland_mask)
    result = ClinicalResult()
    result.rloss = r_loss(gland, eyelid_mask)
    result.warnings.extend(result.rloss.warnings)

    total_area = int(gland.sum())
    if total_area == 0:
        result.warnings.append("腺体掩膜为空：实例数为 0，形态学指标为 NaN")
        result.instance_df = pd.DataFrame(columns=INSTANCE_COLUMNS)
        result.summary = [
            ["腺体缺失率 R_loss (%)", "N/A"],
            ["Meiboscore 分级", f"{result.rloss.meiboscore}（{result.rloss.risk}）"
             if result.rloss.meiboscore >= 0 else "N/A"],
            ["眼睑 ROI 面积(px)", str(result.rloss.eyelid_area_px)],
            ["腺体在 ROI 内面积(px)", str(result.rloss.gland_in_roi_px)],
            ["检测腺体数量", "0"],
            ["腺体覆盖率(全图)", "0.00%"],
            ["实例级弯曲度 τ 均值", "N/A"],
            ["实例级平均曲率 κ 均值", "N/A"],
            ["截断腺体占比", "N/A"],
            ["量化口径", "像素未标定，比例为相对值；R_loss 以眼睑 ROI 为分母"],
        ]
        return result

    df = instance_metrics(gland, eyelid_mask, min_area=min_area)
    result.instance_df = df

    def _mean(column: str) -> float:
        if df is None or len(df) == 0:
            return float("nan")
        values = pd.to_numeric(df[column], errors="coerce")
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else float("nan")

    tau_mean = _mean("弯曲度τ")
    kappa_mean = _mean("平均曲率κ")
    trunc_ratio = truncation_ratio(df)

    def fmt(value: float) -> str:
        return "N/A" if not np.isfinite(value) else f"{value:.3f}"

    result.summary = [
        ["腺体缺失率 R_loss (%)", "N/A" if not np.isfinite(result.rloss.r_loss_percent)
         else f"{result.rloss.r_loss_percent:.2f}"],
        ["Meiboscore 分级", f"{result.rloss.meiboscore}（{result.rloss.risk}）"
         if result.rloss.meiboscore >= 0 else "N/A"],
        ["眼睑 ROI 面积(px)", str(result.rloss.eyelid_area_px)],
        ["腺体在 ROI 内面积(px)", str(result.rloss.gland_in_roi_px)],
        ["检测腺体数量", str(len(df))],
        ["腺体覆盖率(全图)", f"{total_area / gland.size:.2%}"],
        ["实例级弯曲度 τ 均值", fmt(tau_mean)],
        ["实例级平均曲率 κ 均值", fmt(kappa_mean)],
        ["截断腺体占比", "N/A" if not np.isfinite(trunc_ratio) else f"{trunc_ratio:.2%}"],
        ["量化口径", "像素未标定，比例为相对值；R_loss 以眼睑 ROI 为分母"],
    ]
    return result


def cldice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1.0) -> float:
    """评估用 clDice（用 skimage 骨架，非可微分版本）。"""
    from skimage.morphology import skeletonize

    pred = _binary(pred_mask)
    gt = _binary(gt_mask, pred.shape)
    if int(gt.sum()) == 0:
        return float("nan")
    if int(pred.sum()) == 0:
        return 0.0
    skel_pred = skeletonize(pred > 0)
    skel_gt = skeletonize(gt > 0)
    tprec = float((skel_pred & (gt > 0)).sum() + smooth) / float(skel_pred.sum() + smooth)
    tsens = float((skel_gt & (pred > 0)).sum() + smooth) / float(skel_gt.sum() + smooth)
    if tprec + tsens <= 0:
        return float("nan")
    return float(2.0 * tprec * tsens / (tprec + tsens))
