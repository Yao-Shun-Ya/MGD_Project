"""腺体实例跨访视匹配（WP-8）。

两次随访之间的腺体实例（来自 `imaging.clinical.INSTANCE_COLUMNS` 的实例明细
DataFrame）需要建立对应关系，才能计算逐实例的形态变化（腺体变长/变短、
弯曲度变化、宽度变化）。本模块用「质心距离 + 形状（长轴）差异」构成成本矩阵，
`scipy.optimize.linear_sum_assignment` 求全局最优二分匹配，并按空间距离阈值
截断不可能的配对。

术语约定（docstring 承诺）：弯曲度 τ 的扭曲指数 TI = τ − 1，因此
ΔTI = Δτ —— 输出列 `delta_tau` 直接就是 Δ(弯曲度τ)。
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#: 与 imaging.clinical.INSTANCE_COLUMNS 一致的中文列名（不得另造列名）
COL_ID = "实例编号"
COL_AREA = "面积(px)"
COL_LENGTH = "长轴(px)"
COL_WIDTH = "短轴(px)"
COL_TAU = "弯曲度τ"

#: match_instances 必需列（id 用于对齐，长轴进形状成本，短轴/τ 进差分输出）
REQUIRED_COLUMNS = (COL_ID, COL_LENGTH, COL_WIDTH, COL_TAU)

#: cost 参数 → (质心项权重, 形状项权重)。未知 cost → ValueError。
_COST_WEIGHTS: dict[str, tuple[float, float]] = {
    "centroid+length": (1.0, 0.5),
    "centroid": (1.0, 0.0),
    "length": (0.0, 0.5),
}

#: 不允许配对时在成本矩阵里填的大数（分配后据此过滤）
_FORBIDDEN = 1e6

_EMPTY_COLUMNS: dict[str, pd.Series] = {
    "base_id": pd.Series(dtype=float),
    "follow_id": pd.Series(dtype=float),
    "status": pd.Series(dtype=object),
    "cost": pd.Series(dtype=float),
    "delta_length_px": pd.Series(dtype=float),
    "delta_tau": pd.Series(dtype=float),
    "delta_width_px": pd.Series(dtype=float),
}


def instance_centroids(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    """从二值腺体掩膜提取实例质心。

    输入按二值处理（>0 为前景），`cv2.connectedComponentsWithStats` 得到连通域，
    返回 {label: (cx, cy)}（像素坐标，跳过背景 label 0）。空掩膜返回 {}。
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"掩膜必须是二维，收到 shape={arr.shape}")
    binary = (arr > 0).astype(np.uint8)
    if binary.sum() == 0:
        return {}
    num, _labels, _stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result: dict[int, tuple[float, float]] = {}
    for label in range(1, int(num)):
        cx, cy = centroids[label]
        result[int(label)] = (float(cx), float(cy))
    return result


def _require_columns(df: pd.DataFrame, side: str) -> None:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"{side} DataFrame 缺少必需列「{col}」；"
                f"现有列：{list(df.columns)}（预期为 INSTANCE_COLUMNS 口径）"
            )


def _safe_delta(follow_value, base_value) -> float:
    """随访 − 基线；任一侧非有限（NaN/Inf）→ NaN，绝不冒充 0。"""
    follow_num = float(follow_value)
    base_num = float(base_value)
    if not (np.isfinite(follow_num) and np.isfinite(base_num)):
        return float("nan")
    return follow_num - base_num


def _validated_ids(df: pd.DataFrame, side: str) -> list:
    if COL_ID not in df.columns:
        raise ValueError(f"{side} DataFrame 缺少必需列「{COL_ID}」（实例编号），无法匹配")
    ids: list = []
    for raw in df[COL_ID].tolist():
        if pd.isna(raw):
            raise ValueError(f"{side} DataFrame 的「{COL_ID}」存在缺失值")
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            ids.append(str(raw))
    return ids


def _unmatched_row(base_id, follow_id, status: str) -> dict:
    return {
        "base_id": base_id,
        "follow_id": follow_id,
        "status": status,
        "cost": np.nan,
        "delta_length_px": np.nan,
        "delta_tau": np.nan,
        "delta_width_px": np.nan,
    }


def match_instances(
    base_df,
    follow_df,
    base_centroids=None,
    follow_centroids=None,
    cost="centroid+length",
    max_distance_px: float = 40.0,
) -> pd.DataFrame:
    """把基线与随访两次的腺体实例做全局最优匹配。

    成本（越小越可能配对）：
      - 质心项（提供 centroids 时）：欧氏像素距离 / max_distance_px（归一化到
        [0,1]）；原始像素距离 > max_distance_px 的配对直接在成本矩阵置大数禁用，
        分配后过滤，视为不可能配对。
      - 形状项：0.5 × |长轴差| / max(1, 两帧全体平均长轴)（列「长轴(px)」）。
      - `cost` 参数选择权重："centroid+length"（默认，1.0/0.5）、"centroid"、
        "length"；未知取值 → ValueError。

    centroids 为 None 时退化为纯形状成本（无质心项，也不做距离截断）；
    `cost="centroid"` 却未提供 centroids 时无法构造成本，直接 ValueError。

    输出列：`base_id, follow_id, status, cost, delta_length_px, delta_tau,
    delta_width_px`。
      - status ∈ {"matched", "disappeared", "new"}（基线有随访无 = disappeared，
        随访新增 = new）。
      - delta 仅 matched 行有值：delta_length_px = follow 长轴 − base 长轴
        （TI = τ − 1，故 ΔTI = Δτ，delta_tau 即 Δ(弯曲度τ)）；
        delta_width_px = Δ(短轴(px))。disappeared / new 行的 cost 与各 delta 列为 NaN。
    """
    if cost not in _COST_WEIGHTS:
        known = ", ".join(sorted(_COST_WEIGHTS))
        raise ValueError(f"未知 cost 模式「{cost}」；可用：{known}")
    weight_centroid, weight_shape = _COST_WEIGHTS[cost]

    base_df = pd.DataFrame(base_df)
    follow_df = pd.DataFrame(follow_df)
    _require_columns(base_df, "base")
    _require_columns(follow_df, "follow")

    base_ids = _validated_ids(base_df, "base")
    follow_ids = _validated_ids(follow_df, "follow")
    # NaN 纪律：长度含 NaN/Inf 的配对，形状成本项记 0 贡献（中性），绝不把长度冒充成 0px；
    # delta 输出仍用原始值 → NaN 诚实透传。
    raw_base_len = base_df[COL_LENGTH].to_numpy(dtype=float)
    raw_follow_len = follow_df[COL_LENGTH].to_numpy(dtype=float)

    columns = list(_EMPTY_COLUMNS.keys())
    if len(base_ids) == 0 or len(follow_ids) == 0:
        rows: list[dict] = []
        if base_ids:
            rows.extend(_unmatched_row(bid, np.nan, "disappeared") for bid in base_ids)
        if follow_ids:
            rows.extend(_unmatched_row(np.nan, fid, "new") for fid in follow_ids)
        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

    use_centroid = base_centroids is not None and follow_centroids is not None
    # centroids 缺失时质心项退化为 0；若退化后没有任何有效成本项则拒绝
    # （例如 cost='centroid' 又不给 centroids），避免静默给出任意匹配。
    if (weight_centroid if use_centroid else 0.0) + weight_shape <= 0.0:
        raise ValueError(
            "cost 模式在当前输入下没有任何有效成本项"
            "（cost='centroid' 需要提供 base_centroids / follow_centroids）"
        )

    cost_matrix = np.zeros((len(base_ids), len(follow_ids)), dtype=float)
    allowed = np.ones_like(cost_matrix, dtype=bool)
    concat = np.concatenate([raw_base_len, raw_follow_len])
    finite = concat[np.isfinite(concat)]
    scale = max(1.0, float(np.mean(finite)) if finite.size else 1.0)
    max_dist = float(max_distance_px)

    for i in range(len(base_ids)):
        for j in range(len(follow_ids)):
            base_length = float(raw_base_len[i])
            follow_length = float(raw_follow_len[j])
            if np.isfinite(base_length) and np.isfinite(follow_length):
                shape_cost = 0.5 * abs(follow_length - base_length) / scale
            else:
                shape_cost = 0.0  # 任一侧非有限 → 形状项中性，交给质心项/默认匹配
            if not use_centroid:
                cost_matrix[i, j] = weight_shape * shape_cost
                continue
            bc = base_centroids.get(base_ids[i])
            fc = follow_centroids.get(follow_ids[j])
            if bc is None or fc is None:
                # 缺质心的实例只依赖形状项，不做距离截断
                cost_matrix[i, j] = weight_shape * shape_cost
                continue
            dist = float(np.hypot(float(bc[0]) - float(fc[0]), float(bc[1]) - float(fc[1])))
            if dist > max_dist:
                cost_matrix[i, j] = _FORBIDDEN
                allowed[i, j] = False
            else:
                cost_matrix[i, j] = weight_centroid * (dist / max_dist) + weight_shape * shape_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_base: set = set()
    matched_follow: set = set()
    rows = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if not allowed[i, j] or cost_matrix[i, j] >= _FORBIDDEN:
            continue
        bid, fid = base_ids[i], follow_ids[j]
        rows.append(
            {
                "base_id": bid,
                "follow_id": fid,
                "status": "matched",
                "cost": float(cost_matrix[i, j]),
                "delta_length_px": _safe_delta(raw_follow_len[j], raw_base_len[i]),
                "delta_tau": _safe_delta(follow_df.iloc[j][COL_TAU], base_df.iloc[i][COL_TAU]),
                "delta_width_px": _safe_delta(follow_df.iloc[j][COL_WIDTH], base_df.iloc[i][COL_WIDTH]),
            }
        )
        matched_base.add(bid)
        matched_follow.add(fid)

    for idx, bid in enumerate(base_ids):
        if bid not in matched_base:
            rows.append(_unmatched_row(bid, np.nan, "disappeared"))
    for fid in follow_ids:
        if fid not in matched_follow:
            rows.append(_unmatched_row(np.nan, fid, "new"))

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)
