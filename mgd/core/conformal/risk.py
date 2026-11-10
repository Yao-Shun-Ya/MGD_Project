"""风险控制：用 split conformal 为运行期参数提供分布无关的图级保证。

两个入口：

- :func:`select_threshold_for_fnr`：为「概率图 → 二值掩膜」选阈值 ``t``
  （``prob >= t`` 记为正），使逐图假阴性率 FNR 的共形上分位不超过 epsilon，
  即 ``P(FNR <= epsilon) >= 1 - alpha``；可行集中取最小 t（保召回最保守）。
- :func:`select_shrink_for_fp`：为「置信掩膜」选置信下限 ``t``
  （保留像素 = ``prob >= t``），使逐图假阳率 FP_rate 的共形上分位不超过
  epsilon；可行集中取最小 t（保留最多像素且 FP 有保证）。

统计口径与 calibrator 一致：对逐图指标序列取有限样本修正上分位
``q = sorted(指标)[ceil((n+1)(1-alpha)) - 1]``，``n`` 为有效图数；
``n < 1/alpha`` 的阈值不可评估（无法给出有限样本保证）。

可交换性单元是**图**（不是像素）：假设校准图集与部署图的逐图指标可交换。
GT 无正像素的图对 FNR 无定义、keep 为空的图对 FP_rate 无定义，
一律跳过并计数，绝不冒充 0.0。
"""
from __future__ import annotations

import math

import numpy as np

from ..runtime.logging import get_logger
from .calibrator import conformal_upper_quantile, validate_alpha

logger = get_logger("conformal.risk")

#: 默认阈值网格：0.05, 0.10, ..., 0.95（19 个候选，保留 2 位小数）
DEFAULT_GRID = np.round(np.linspace(0.05, 0.95, 19), 2)


def _validate_params(epsilon: float, alpha: float) -> tuple[float, float]:
    """校验 epsilon ∈ [0, 1] 与 alpha ∈ (0, 0.5]，返回 (float, float)。"""
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"epsilon 必须为数值，当前为 {epsilon!r}") from exc
    if not math.isfinite(epsilon) or not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon 必须在 [0, 1] 区间，当前为 {epsilon!r}")
    return epsilon, validate_alpha(alpha)


def _prepare_grid(grid) -> np.ndarray:
    """grid 规范化：升序去重的一维 float 数组（保证「最小可行 t」语义）。"""
    if grid is None:
        grid = DEFAULT_GRID
    arr = np.asarray(grid, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("grid 不能为空")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("grid 必须全为有限数值")
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        raise ValueError(f"grid 取值必须在 [0, 1] 区间，当前范围 [{arr.min():g}, {arr.max():g}]")
    return np.unique(arr)


def _validate_inputs(prob_maps, gt_masks) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """校验图序列：等长、非空、逐对形状一致、概率图取值 ∈ [0,1]、GT 可二值化。

    GT 二值化口径：``> 0`` 视为正（容忍 {0,1} 与 {0,255} 等常见存储格式）。
    """
    try:
        prob_list = list(prob_maps)
        gt_list = list(gt_masks)
    except TypeError as exc:
        raise ValueError("prob_maps / gt_masks 必须为图序列（可迭代的 array-like）") from exc
    if len(prob_list) != len(gt_list):
        raise ValueError(
            f"prob_maps 与 gt_masks 数量不一致：{len(prob_list)} vs {len(gt_list)}"
        )
    if not prob_list:
        raise ValueError("prob_maps / gt_masks 不能为空")
    probs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    for i, (p, g) in enumerate(zip(prob_list, gt_list)):
        p_arr = np.asarray(p, dtype=float)
        g_arr = np.asarray(g, dtype=float)
        if p_arr.ndim == 0 or g_arr.ndim == 0:
            raise ValueError(f"第 {i} 对输入必须是逐像素图（至少一维），收到标量")
        if p_arr.shape != g_arr.shape:
            raise ValueError(f"第 {i} 对 prob/gt 形状不一致：{p_arr.shape} vs {g_arr.shape}")
        if not bool(np.all(np.isfinite(p_arr))):
            raise ValueError(f"第 {i} 张概率图含 NaN/inf")
        if p_arr.size and (float(p_arr.min()) < 0.0 or float(p_arr.max()) > 1.0):
            raise ValueError(f"第 {i} 张概率图取值必须在 [0, 1] 区间（概率图语义）")
        if not bool(np.all(np.isfinite(g_arr))):
            raise ValueError(f"第 {i} 张 GT 掩膜含 NaN/inf")
        if g_arr.size and float(g_arr.min()) < 0.0:
            raise ValueError(f"第 {i} 张 GT 掩膜含负值（二值化口径：>0 视为正）")
        probs.append(p_arr)
        gts.append(g_arr)
    return probs, gts


def select_threshold_for_fnr(prob_maps, gt_masks, epsilon=0.10, alpha=0.10, grid=None) -> dict:
    """选二值化阈值 ``t``，使逐图 FNR 的共形上分位不超过 ``epsilon``。

    统计口径：

    - 像素判正口径：``prob >= t`` 记为预测正。
    - 逐图 ``FNR_i = FN / (FN + TP)``，其中 FN = GT 正像素被判负
      （``prob < t``），TP = GT 正像素被判正（``prob >= t``）。
    - GT 无正像素的图 FNR 无定义 → 跳过并在 ``skipped_empty_gt`` 计数。
    - 对有效图的 FNR 序列取共形上分位
      ``q = sorted(FNR)[ceil((n+1)(1-alpha)) - 1]``（n 为有效图数）。
      ``n < 1/alpha`` 时该阈值不可评估（有限样本保证不存在）。
    - ``q <= epsilon`` 即可行；**在可行集中取最小 t**——阈值越小保留的
      正预测越多，对召回（漏检）越保守。
    - 保证（图级可交换性下，有限样本精确成立）：
      ``P(逐图 FNR <= epsilon) >= 1 - alpha``。

    :param prob_maps: 概率图序列（每张取值 ∈ [0, 1]）。
    :param gt_masks: GT 掩膜序列（与 prob_maps 等长且逐对形状一致；
      二值化口径 ``> 0`` 视为正，容忍 {0,255} 等存储格式）。
    :param epsilon: 允许的最大图级 FNR，∈ [0, 1]。
    :param alpha: 失效率，∈ (0, 0.5]。
    :param grid: 候选阈值；默认 ``np.round(np.linspace(0.05, 0.95, 19), 2)``。
    :returns: 可行时 ``{"feasible": True, "threshold": t, "achieved_fnr": q,
        "guarantee": "P(FNR<=epsilon)>=1-alpha", "n_images": n,
        "skipped_empty_gt": k}``；无可行时 ``{"feasible": False, "reason": 中文原因,
        "best_q": 所有可评估阈值中最小的 q}``。
    :raises ValueError: 输入为空 / 数量不一致 / 形状不一致 / 取值非法，
        或 epsilon / alpha 非法。
    """
    epsilon, alpha = _validate_params(epsilon, alpha)
    probs, gts = _validate_inputs(prob_maps, gt_masks)
    grid_arr = _prepare_grid(grid)

    pos_masks = [g > 0 for g in gts]
    valid_idx = [i for i, m in enumerate(pos_masks) if bool(m.any())]
    skipped_empty_gt = len(gts) - len(valid_idx)
    n_valid = len(valid_idx)
    min_required = 1.0 / alpha
    if n_valid < min_required:
        logger.debug("select_threshold_for_fnr：有效图数不足（n=%d < %g），拒绝评估", n_valid, min_required)
        return {
            "feasible": False,
            "reason": (
                f"有效图数不足：n={n_valid} 需要 ≥ 1/alpha={min_required:g}"
                f"（跳过无正像素 GT 的图 {skipped_empty_gt} 张），无法给出有限样本共形保证"
            ),
            "best_q": None,
        }

    pos_list = [pos_masks[i] for i in valid_idx]
    prob_list_valid = [probs[i] for i in valid_idx]
    pos_counts = [int(np.count_nonzero(m)) for m in pos_list]

    best: tuple[float, float] | None = None
    quantiles: list[float] = []
    for t in grid_arr:  # grid 升序 → 首个可行即最小可行 t
        fnrs = []
        for pos, p_arr, count in zip(pos_list, prob_list_valid, pos_counts):
            fn = int(np.count_nonzero(pos & (p_arr < t)))
            fnrs.append(fn / count)
        q = conformal_upper_quantile(fnrs, alpha)
        quantiles.append(q)
        if best is None and q <= epsilon:
            best = (float(t), q)
            logger.info(
                "select_threshold_for_fnr：选定阈值 t=%.3f，achieved_fnr=%.4f ≤ epsilon=%.3f",
                t, q, epsilon,
            )
            break

    if best is not None:
        return {
            "feasible": True,
            "threshold": best[0],
            "achieved_fnr": best[1],
            "guarantee": "P(FNR<=epsilon)>=1-alpha",
            "n_images": n_valid,
            "skipped_empty_gt": skipped_empty_gt,
        }
    logger.debug("select_threshold_for_fnr：网格内无可行阈值（best_q=%r）", min(quantiles) if quantiles else None)
    return {
        "feasible": False,
        "reason": "网格内无可行阈值：所有候选阈值的逐图 FNR 共形上分位均大于 epsilon（含全错预测的情形）",
        "best_q": min(quantiles) if quantiles else None,
    }


def select_shrink_for_fp(prob_maps, gt_masks, epsilon=0.05, alpha=0.10) -> dict:
    """选置信下限 ``t`` 生成「置信掩膜」，使逐图 FP_rate 的共形上分位不超过 ``epsilon``。

    「置信掩膜」语义：只保留高置信像素（``prob >= t``）得到的掩膜，
    低置信区域被收缩掉（不参与下游测量 / 展示），用于把「模型不确定的
    区域」显式排除，避免把噪声像素当真实结构。

    统计口径：

    - 保留像素（keep）：``prob >= t``。
    - 逐图 ``FP_rate_i = |keep ∩ ¬GT| / max(1, |keep|)``：保留像素中
      假阳性的占比；``keep`` 为空的图无定义 → 跳过并在所选阈值的
      ``skipped_empty_keep`` 计数。
    - 对有效图的 FP_rate 序列取共形上分位
      ``q = sorted(FP_rate)[ceil((n+1)(1-alpha)) - 1]``（n 为有效图数）；
      ``n < 1/alpha`` 时该阈值不可评估。
    - ``q <= epsilon`` 即可行；**在可行集中取最小 t**——阈值越小保留的
      像素越多，同时假阳率仍有 ``P(FP_rate <= epsilon) >= 1 - alpha``
      的图级有限样本保证。

    :param prob_maps: 概率图序列（每张取值 ∈ [0, 1]）。
    :param gt_masks: GT 掩膜序列（等长且逐对形状一致；``> 0`` 视为正）。
    :param epsilon: 允许的最大图级 FP_rate，∈ [0, 1]。
    :param alpha: 失效率，∈ (0, 0.5]。
    :returns: 可行时 ``{"feasible": True, "threshold": t, "achieved_fp_rate": q,
        "guarantee": "P(FP_rate<=epsilon)>=1-alpha", "n_images": n,
        "skipped_empty_keep": k}``；无可行时 ``{"feasible": False, "reason": 中文原因,
        "best_q": 所有可评估阈值中最小的 q}``。
    :raises ValueError: 输入为空 / 数量不一致 / 形状不一致 / 取值非法，
        或 epsilon / alpha 非法。
    """
    epsilon, alpha = _validate_params(epsilon, alpha)
    probs, gts = _validate_inputs(prob_maps, gt_masks)
    grid_arr = _prepare_grid(None)
    neg_masks = [~(g > 0) for g in gts]

    min_required = 1.0 / alpha
    best: tuple[float, float, int, int] | None = None
    best_q: float | None = None
    for t in grid_arr:  # grid 升序 → 首个可行即最小可行 t
        rates: list[float] = []
        skipped_empty_keep = 0
        for p_arr, neg in zip(probs, neg_masks):
            keep = p_arr >= t
            n_keep = int(np.count_nonzero(keep))
            if n_keep == 0:
                skipped_empty_keep += 1
                continue
            fp = int(np.count_nonzero(keep & neg))
            rates.append(fp / n_keep)
        if len(rates) < min_required:
            continue  # 该 t 不可评估
        q = conformal_upper_quantile(rates, alpha)
        if best_q is None or q < best_q:
            best_q = q
        if best is None and q <= epsilon:
            best = (float(t), q, len(rates), skipped_empty_keep)
            break

    if best is not None:
        logger.info(
            "select_shrink_for_fp：选定置信下限 t=%.3f，achieved_fp_rate=%.4f ≤ epsilon=%.3f",
            best[0], best[1], epsilon,
        )
        return {
            "feasible": True,
            "threshold": best[0],
            "achieved_fp_rate": best[1],
            "guarantee": "P(FP_rate<=epsilon)>=1-alpha",
            "n_images": best[2],
            "skipped_empty_keep": best[3],
        }
    logger.debug("select_shrink_for_fp：网格内无可行置信下限（best_q=%r）", best_q)
    if best_q is not None:
        reason = "网格内无可行置信下限：所有候选 t 的逐图 FP_rate 共形上分位均大于 epsilon"
    else:
        reason = "所有候选 t 的有效图数均不足（keep 为空的图过多），无法给出有限样本共形保证"
    return {"feasible": False, "reason": reason, "best_q": best_q}
