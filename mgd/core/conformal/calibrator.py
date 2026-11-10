"""Split conformal 预测区间：不确定性量化的一致性校准器。

方法（split conformal prediction，Vovk et al.）：
1. 把带真值的数据分成训练份与校准份；本模块只消费校准份 ``(predictions, truths)``。
2. 在校准份上计算非一致性得分（nonconformity score）::

       s_i = |pred_i - truth_i|

3. 取有限样本修正分位（finite-sample corrected quantile）::

       q = sorted(s)[ceil((n + 1) * (1 - alpha)) - 1]

   其中 ``alpha`` 为目标失效率（每侧），``n`` 为校准样本数。
4. 对新点 ``point`` 输出预测区间::

       [point - q, point + q]

统计保证（关键假设——可交换性 exchangeability）：校准份得分与测试点得分
可交换（i.i.d. 是其特例）时，有::

    P(truth ∈ [pred - q, pred + q]) >= 1 - alpha

该保证对任意有限 ``n`` 精确成立（不依赖渐近近似）。但对分布漂移不稳健：
若测试数据与校准数据不可交换（例如扫描设备、人群改变），覆盖率保证失效，
必须重新校准。

样本量要求：``n >= 1/alpha`` 时有限样本分位才有意义（否则排序后连一个
「超出分位」的期望样本都凑不齐，分位退化成最大值也没有理论支撑）。
样本不足时本模块直接抛 :class:`ValueError`，禁止静默输出区间。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..runtime.logging import get_logger

logger = get_logger("conformal.calibrator")

#: 浮点安全余量：当 (n+1)(1-alpha) 在实数意义下恰为整数时，浮点乘积可能
#: 略大于该整数，使 ceil 多加 1（大 n 时会真实发生）。减去一个远小于任何
#: 真实小数间隔的余量可消除该偏差，且不影响真实非整数值的分位结果。
_QUANTILE_EPS = 1e-9


def validate_alpha(alpha: float) -> float:
    """校验 ``alpha ∈ (0, 0.5]``，返回 float；非法时抛 :class:`ValueError`。"""
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alpha 必须为数值，当前为 {alpha!r}") from exc
    if not math.isfinite(alpha) or not (0.0 < alpha <= 0.5):
        raise ValueError(f"alpha 必须在 (0, 0.5] 区间，当前为 {alpha!r}")
    return alpha


def _insufficient_sample_message(n: int, min_required: float) -> str:
    """n < 1/alpha 时的统一报错文案（含可执行建议）。"""
    return (
        f"校准样本不足：n={n} 需要 ≥ 1/alpha={min_required:g}。"
        f"建议：增加校准样本至至少 {math.ceil(min_required)} 个，或调大 alpha；"
        "共形覆盖率保证依赖有限样本分位，样本不足时禁止静默输出区间。"
    )


def conformal_upper_quantile(scores, alpha: float) -> float:
    """共形上分位：``q = sorted(scores)[ceil((n+1)(1-alpha)) - 1]``。

    :param scores: 非一致性得分序列（必须全为有限数值）。
    :param alpha: 目标失效率，∈ (0, 0.5]。
    :raises ValueError: alpha 非法、得分含 NaN/inf，或 ``n < 1/alpha``
        （与 :class:`ConformalCalibrator` 同一口径，禁止静默出分位）。
    """
    alpha = validate_alpha(alpha)
    values: list[float] = []
    for raw in scores:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"非一致性得分必须为数值序列，遇到 {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"非一致性得分含 NaN/inf：{value!r}")
        values.append(value)
    n = len(values)
    min_required = 1.0 / alpha
    if n < min_required:
        raise ValueError(_insufficient_sample_message(n, min_required))
    k = math.ceil((n + 1) * (1.0 - alpha) - _QUANTILE_EPS)
    if not 1 <= k <= n:  # 防御：在上述校验下理论上不可达
        raise ValueError(f"分位索引越界：k={k}, n={n}, alpha={alpha}")
    return float(sorted(values)[k - 1])


@dataclass
class ConformalCalibrator:
    """Split conformal 校准器（不确定性区间）。

    用法::

        cal = ConformalCalibrator(alpha=0.10).fit(preds, truths)
        lo, hi = cal.interval(point)   # [point - q, point + q]

    统计口径（实现契约）：

    - **可交换性假设**：校准集与测试集的非一致性得分可交换（i.i.d. 为特例）；
      分布漂移会直接使覆盖率保证失效，需重新校准。
    - **非一致性得分**：``s_i = |pred_i - truth_i|``。
    - **预测区间**：``[point - q, point + q]``，其中有限样本修正分位
      ``q = sorted(s)[ceil((n+1)(1-alpha)) - 1]``。
    - **保证**：在可交换性下 ``P(truth ∈ 区间) ≥ 1 - alpha`` 对任意有限 n 精确成立。
    - **样本量下限**：``n >= 1/alpha``；不足时 :meth:`fit` / :meth:`quantile`
      抛 :class:`ValueError`（消息含「校准样本不足」与建议），绝不静默出区间。

    :meth:`fit` 成功后覆盖旧 scores（可重复拟合）；失败时不修改任何状态。
    """

    alpha: float = 0.10
    scores: list[float] = field(default_factory=list)

    def fit(self, predictions, truths) -> "ConformalCalibrator":
        """用校准集 ``(predictions, truths)`` 拟合非一致性得分，返回自身。

        :raises ValueError: alpha 非法、两序列长度不等或为空、含 NaN/inf、
            或 ``n < 1/alpha``（校准样本不足）。
        """
        alpha = validate_alpha(self.alpha)
        try:
            pred_values = [float(v) for v in predictions]
            truth_values = [float(v) for v in truths]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"predictions / truths 必须为数值序列：{exc}") from exc
        if len(pred_values) != len(truth_values):
            raise ValueError(
                f"predictions 与 truths 长度不一致：{len(pred_values)} vs {len(truth_values)}"
            )
        if not pred_values:
            raise ValueError("predictions / truths 不能为空（至少 1 对样本）")
        for i, (p, t) in enumerate(zip(pred_values, truth_values)):
            if not math.isfinite(p) or not math.isfinite(t):
                raise ValueError(f"第 {i} 对样本含 NaN/inf：pred={p!r}, truth={t!r}")

        scores = [abs(p - t) for p, t in zip(pred_values, truth_values)]
        min_required = 1.0 / alpha
        if len(scores) < min_required:
            raise ValueError(_insufficient_sample_message(len(scores), min_required))

        # 全部校验通过后才落状态：失败的 fit 不污染旧校准
        self.alpha = alpha
        self.scores = scores
        logger.debug("ConformalCalibrator 拟合完成：n=%d, alpha=%g", len(scores), alpha)
        return self

    def quantile(self) -> float:
        """返回有限样本修正分位 ``q``。

        :raises ValueError: 未拟合 / 样本不足（``n < 1/alpha``）/ alpha 非法 /
            scores 含非有限值。
        """
        return conformal_upper_quantile(self.scores, self.alpha)

    def interval(self, point: float) -> tuple[float, float]:
        """返回 ``(point - q, point + q)``。

        :raises ValueError: ``point`` 为 NaN/inf，或分位不可用（同 :meth:`quantile`）。
        """
        try:
            point = float(point)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"point 必须为数值，当前为 {point!r}") from exc
        if not math.isfinite(point):
            raise ValueError(f"point 必须为有限数值，当前为 {point!r}")
        q = self.quantile()
        return (point - q, point + q)

    def to_dict(self) -> dict:
        """序列化为可存储 dict（含 alpha 与 scores，供 store 持久化）。"""
        return {
            "alpha": float(self.alpha),
            "scores": [float(s) for s in self.scores],
            "n": len(self.scores),
        }

    @classmethod
    def from_dict(cls, d) -> "ConformalCalibrator":
        """从 :meth:`to_dict` 的输出还原；额外字段忽略，缺字段/非法值抛 :class:`ValueError`。"""
        if not isinstance(d, dict):
            raise ValueError(f"from_dict 需要 dict，收到 {type(d).__name__}")
        if "alpha" not in d or "scores" not in d:
            raise ValueError("from_dict 缺少 'alpha' 或 'scores' 字段")
        alpha = validate_alpha(d["alpha"])
        raw_scores = d["scores"]
        if not isinstance(raw_scores, (list, tuple)):
            raise ValueError(f"scores 必须为数值列表，收到 {type(raw_scores).__name__}")
        scores: list[float] = []
        for raw in raw_scores:
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"scores 必须为数值列表，遇到 {raw!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"scores 含 NaN/inf：{value!r}")
            scores.append(value)
        return cls(alpha=alpha, scores=scores)
