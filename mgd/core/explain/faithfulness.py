"""忠实度证据（faithfulness）：deletion / insertion AUC + 简化 ROAD + 随机对照。

WP-3 验收核心：归因图必须通过「删减/插入」证据检验，而不是把概率伪彩
当热力图交差。faithfulness_report 同时计算固定 seed 的随机归因对照，
只有当给定归因显著优于随机时才判 pass。
"""
from __future__ import annotations

import numpy as np

# numpy 2.0 起 np.trapz 改名 np.trapezoid；此处解析期兼容，行为完全一致
_trapz = getattr(np, "trapz", None) or np.trapezoid

#: verdict 判定阈值（engine-defined，见 faithfulness_report docstring）
_VERDICT_MARGIN = 0.05


def _check_pair(prob_map: np.ndarray, attribution: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """输入校验：2D 同形状 + 全有限 + prob_map 落在 [0,1]（容差 1e-6）。

    prob_map 是概率图：uint8 掩膜（{0,255}）误当概率图传入会被范围检查拒绝，
    避免静默算出无意义的 AUC。
    """
    prob = np.asarray(prob_map, dtype=np.float32)
    attr = np.asarray(attribution, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"prob_map 期望 2D (H,W)，实际形状 {prob.shape}")
    if attr.shape != prob.shape:
        raise ValueError(f"prob_map 与 attribution 形状不一致：{prob.shape} vs {attr.shape}")
    if not bool(np.all(np.isfinite(prob))):
        raise ValueError("prob_map 含 NaN/Inf，无法计算忠实度证据")
    if not bool(np.all(np.isfinite(attr))):
        raise ValueError("attribution 含 NaN/Inf，无法计算忠实度证据")
    if float(prob.min()) < -1e-6 or float(prob.max()) > 1.0 + 1e-6:
        raise ValueError(
            f"prob_map 超出 [0,1] 概率范围（min={float(prob.min()):g}, max={float(prob.max()):g}）："
            "uint8 掩膜误当概率图会被拒绝，请先归一化到 [0,1]"
        )
    return prob, attr


def _curve_auc(prob_map: np.ndarray, attribution: np.ndarray, steps: int, baseline: float, mode: str) -> float:
    """deletion / insertion 共用的曲线 AUC。

    x 轴为已处理像素比例 ∈ [0,1]（含 0 与 1，共 steps+1 个采样点），
    y 为当前概率图的全图均值；AUC = np.trapz(曲线, x) / (x[-1] − x[0])，归一到 [0,1]。
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps 必须 >= 1，实际 {steps}")
    prob, attr = _check_pair(prob_map, attribution)
    flat = prob.ravel().astype(np.float64)
    order = np.argsort(-attr.ravel().astype(np.float64), kind="stable")
    n = flat.size
    fractions = np.linspace(0.0, 1.0, steps + 1)
    curve = np.empty(fractions.size, dtype=np.float64)
    baseline = float(baseline)
    for index, fraction in enumerate(fractions):
        k = int(round(float(fraction) * n))
        if mode == "deletion":
            trial = flat.copy()
            trial[order[:k]] = baseline
        else:  # insertion：从全 baseline 开始逐级恢复
            trial = np.full(n, baseline, dtype=np.float64)
            trial[order[:k]] = flat[order[:k]]
        curve[index] = float(trial.mean())
    x = fractions
    return float(_trapz(curve, x) / (x[-1] - x[0]))


def deletion_auc(prob_map: np.ndarray, attribution: np.ndarray, steps: int = 20, baseline: float = 0.0) -> float:
    """Deletion AUC：按 attribution 从大到小逐级把像素置为 baseline。

    记录每级概率图的全图平均概率曲线（x = 已删除比例，含 0% 与 100% 共
    steps+1 个点），AUC = np.trapz 归一到 [0,1]。越低越好：好的归因会先
    删掉真正支撑预测的像素，使平均概率迅速跌落。

    :raises ValueError: prob_map / attribution 含 NaN/Inf、形状不一致，或
        prob_map 超出 [0,1]（uint8 掩膜误当概率图会被拒绝）。
    """
    return _curve_auc(prob_map, attribution, steps, baseline, mode="deletion")


def insertion_auc(prob_map: np.ndarray, attribution: np.ndarray, steps: int = 20, baseline: float = 0.0) -> float:
    """Insertion AUC：从全 baseline 开始，按 attribution 从大到小逐级恢复原像素值。

    记录每级平均概率曲线（x = 已恢复比例），AUC 同样归一到 [0,1]。
    越高越好：好的归因先恢复最有信息量的像素，曲线迅速上升。

    :raises ValueError: prob_map / attribution 含 NaN/Inf、形状不一致，或
        prob_map 超出 [0,1]（uint8 掩膜误当概率图会被拒绝）。
    """
    return _curve_auc(prob_map, attribution, steps, baseline, mode="insertion")


def road_score(prob_map: np.ndarray, attribution: np.ndarray) -> float:
    """engine-defined 简化版 ROAD。

    本实现为简化代理：road = insertion_auc(attribution) − deletion_auc(attribution)，
    越大越忠实。完整 ROAD（Remove And Debias）含噪声估计与去偏步骤，
    见工程文档（docs/engineering）；该代理仅用于本包快速回归门，不替代论文口径。
    """
    return float(insertion_auc(prob_map, attribution) - deletion_auc(prob_map, attribution))


def faithfulness_report(prob_map: np.ndarray, attribution: np.ndarray, random_seed: int = 42) -> dict:
    """忠实度报告：给定归因 vs 随机归因对照。

    - random attribution：np.random.default_rng(random_seed) 生成的均匀噪声
      （固定 seed，完全可复现），作为「随便糊一个图」的下限对照；
    - verdict = "pass" 当且仅当
        deletion_auc ≤ random_deletion_auc − 0.05 且
        insertion_auc ≥ random_insertion_auc + 0.05，
      否则 "fail"；0.05 阈值为 engine-defined（本包验收口径，区分真实归因
      与随机噪声的最小裕量），非论文推导值；
    - 返回键：deletion_auc / insertion_auc / road / random_deletion_auc /
      random_insertion_auc / verdict。
    """
    prob = np.asarray(prob_map)
    deletion = deletion_auc(prob, attribution)
    insertion = insertion_auc(prob, attribution)

    rng = np.random.default_rng(int(random_seed))
    noise = rng.uniform(0.0, 1.0, size=prob.shape)
    random_deletion = deletion_auc(prob, noise)
    random_insertion = insertion_auc(prob, noise)

    passed = (
        deletion <= random_deletion - _VERDICT_MARGIN
        and insertion >= random_insertion + _VERDICT_MARGIN
    )
    return {
        "deletion_auc": float(deletion),
        "insertion_auc": float(insertion),
        "road": float(insertion - deletion),
        "random_deletion_auc": float(random_deletion),
        "random_insertion_auc": float(random_insertion),
        "verdict": "pass" if passed else "fail",
    }
