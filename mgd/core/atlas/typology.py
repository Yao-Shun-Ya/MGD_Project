"""WP-6 形态学分型：8 类 MGD 形态模式与逐条证据输出。

每条规则以 :class:`TypeRule` 声明（key / 中文名 / 判定条件含默认阈值 / 来源），
判定结果以 :class:`TypeEvidence` 返回：必须携带实测值 ``value``、实际比较阈值
``threshold`` 与单位，绝不只给布尔。统一约定（与 quantification 层一致）：
缺失输入 → ``present=False`` 且 ``value=NaN``，绝不用 0 冒充、绝不崩溃。

来源对照（source 词表沿用 ``quantification.registry.ALLOWED_SOURCES``）：
- ``dropout / shortening / thickening / beading / ghost / disruption / confluence``
  为 engine-defined（工程判定口径；shortening 的实例级阈值 0.5 继承
  ``imaging.clinical`` `是否截断`（相对延伸比 < 0.5）既有口径）；
- ``tortuosity`` 为 Pult & Nichols 2016（τ / TI 形态计量口径），但 TI > 0.25 与
  inflection_count ≥ 2 是工程设定阈值（rule.definition 与此处均已注明）。

阈值覆盖：``overrides[rule_key] = threshold``。乘法类规则（dropout / thickening /
confluence）覆盖的是系数，实际比较阈值 = 系数 × 基准值；其余规则覆盖绝对阈值。
shortening 实例级阈值 0.5、拐点阈值 2、峰数阈值 2 固定不可覆盖（继承既有口径）。
未知 rule key / 非数值 / 非有限 / 负值 → ValueError；乘法类系数还必须 > 0。

特征键名复用已注册指标 key：ga_percent / tortuosity_index / inflection_count /
width_mean_px / break_count / diameter_deformation_index（见 quantification.registry）；
非注册表指标沿用既有命名：truncation_ratio（clinical.truncation_ratio）、
width_profile_std / width_peaks / ghost_area_px / min_inter_instance_distance_px /
reference_density_percent / width_median_px（features dict 传入的工程量）。
``classify_instance`` 的单实例 dict 额外兼容 df 行键：`相对延伸比`（或
relative_extension_ratio）/ `是否截断` / `短轴(px)` / `弯曲度τ`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..imaging.clinical import INSTANCE_COLUMNS
from ..runtime.errors import ConfigError

# ---------------------------------------------------------------------------
# 实例表列名（与 clinical.INSTANCE_COLUMNS 严格对齐；错位在导入期即失败）。
# ---------------------------------------------------------------------------
COLUMN_RELATIVE = "相对延伸比"
COLUMN_TRUNCATED = "是否截断"
COLUMN_MINOR = "短轴(px)"
COLUMN_TAU = "弯曲度τ"

for _COLUMN in (COLUMN_RELATIVE, COLUMN_TRUNCATED, COLUMN_MINOR, COLUMN_TAU):
    if _COLUMN not in INSTANCE_COLUMNS:
        raise ConfigError(
            f"atlas 分型依赖的实例列 {_COLUMN!r} 不在 clinical.INSTANCE_COLUMNS "
            f"{INSTANCE_COLUMNS} 中；请同步修改 mgd/core/atlas/typology.py 的列名常量"
        )

# ---------------------------------------------------------------------------
# 默认阈值（overrides 可覆盖；含义见各 TypeRule.definition）。
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict[str, float] = {
    "dropout": 0.5,      # 系数：ga_percent < 系数 × reference_density_percent
    "shortening": 0.2,   # case 级截断占比阈值（实例级相对延伸比阈值固定 0.5）
    "tortuosity": 0.25,  # TI 阈值（拐点数阈值固定 2）
    "thickening": 1.5,   # 系数：宽度 > 系数 × 同例中位宽度
    "beading": 0.3,      # 宽度剖面变异系数阈值（峰数阈值固定 2）
    "ghost": 120.0,      # 低置信连通域最小面积 px（与 instance_metrics min_area 一致）
    "disruption": 1.0,   # break_count 下限
    "confluence": 0.5,   # 系数：实例间最小间距 < 系数 × 平均宽度
}

#: 实例级缩短阈值：相对延伸比 < 0.5（继承 clinical `是否截断` 口径，固定）。
SHORTENING_INSTANCE_RATIO = 0.5
#: 迂曲判定的拐点数下限（工程阈值，固定）。
TORTUOSITY_INFLECTION_MIN = 2
#: 串珠样变的宽度峰数下限（工程阈值，固定）。
BEADING_PEAKS_MIN = 2


@dataclass(frozen=True)
class TypeRule:
    """单条形态学分型规则的可追溯定义。"""

    key: str          # "dropout","shortening","tortuosity","thickening","beading","ghost","disruption","confluence"
    name_zh: str
    definition: str   # 判定条件（含阈值），逐条写清
    source: str       # "Pult & Nichols 2016" 或 "engine-defined"（显式区分）


@dataclass(frozen=True)
class TypeEvidence:
    """单条规则的判定证据：实测值 + 实际阈值，缺失输入 → present=False 且 value=NaN。"""

    key: str
    present: bool
    value: float
    threshold: float
    unit: str
    definition: str


TYPE_RULES: tuple[TypeRule, ...] = (
    TypeRule(
        key="dropout",
        name_zh="腺体缺失",
        definition=(
            "ga_percent < 0.5 × reference_density_percent：GA 腺体面积比低于同眼参考密度的一半"
            "（参考密度由 features['reference_density_percent'] 传入；缺参考 → 不判定 NaN）。"
            "系数 0.5 可用 overrides['dropout'] 覆盖，实际比较阈值 = 系数 × 参考密度"
        ),
        source="engine-defined",
    ),
    TypeRule(
        key="shortening",
        name_zh="腺体缩短",
        definition=(
            "实例相对延伸比 < 0.5 视为截断（沿用 clinical.py `是否截断` 既有口径，该阈值固定）。"
            "case 级判据：截断占比 truncation_ratio ≥ 0.2（默认 0.2，可用 overrides['shortening'] 覆盖）；"
            "instance 级判据：单实例相对延伸比 < 0.5。features 缺 truncation_ratio 时对 instance_df 逐行"
            "评估：present=任一实例触发，value=最显著实例（相对延伸比最小）的取值"
        ),
        source="engine-defined",
    ),
    TypeRule(
        key="tortuosity",
        name_zh="腺体迂曲",
        definition=(
            "tortuosity_index（TI = τ − 1）> 0.25 且 inflection_count ≥ 2。来源说明：TI 的 τ 弧长/弦长"
            "口径为 Pult & Nichols 2016 文献定义，阈值 0.25 与拐点数 2 为工程设定（TI 阈值可用 "
            "overrides['tortuosity'] 覆盖，拐点阈值固定为 2）；inflection_count 缺失时退化为 TI 单项判定"
            "（实例级无逐实例拐点测量），inflection_count 存在但 < 2 时不触发。features 缺 TI 时按 "
            "instance_df 逐行评估（单实例 TI = `弯曲度τ` − 1，value=最显著实例取值）"
        ),
        source="Pult & Nichols 2016",
    ),
    TypeRule(
        key="thickening",
        name_zh="腺体增粗",
        definition=(
            "width_mean_px > 1.5 × 同例中位宽度（中位宽度优先取 features['width_median_px']，"
            "否则取 instance_df 全体 `短轴(px)` 中位数；中位数无效 → 不判定 NaN）。"
            "系数 1.5 可用 overrides['thickening'] 覆盖，实际比较阈值 = 系数 × 中位宽度。"
            "features 缺 width_mean_px 时对 instance_df 逐行评估 `短轴(px)`："
            "present=任一实例触发，value=最显著实例（宽度最大）的取值"
        ),
        source="engine-defined",
    ),
    TypeRule(
        key="beading",
        name_zh="腺体串珠样变",
        definition=(
            "宽度剖面变异系数 CV = width_profile_std / width_mean_px > 0.3 且峰数 width_peaks ≥ 2"
            "（width_peaks 缺失 → 仅按 CV 单项判定；width_profile_std 缺失时回退注册表指标 "
            "diameter_deformation_index，同一公式 std(w)/mean(w)）。"
            "CV 阈值 0.3 可用 overrides['beading'] 覆盖，峰数阈值固定为 2"
        ),
        source="engine-defined",
    ),
    TypeRule(
        key="ghost",
        name_zh="幻影腺体",
        definition=(
            "低置信连通域面积 ghost_area_px ≥ 120 px（min_area 与 quantification/topology "
            "instance_metrics 口径一致；面积由上游概率图计算后经 features['ghost_area_px'] 传入）。"
            "阈值 120 px 可用 overrides['ghost'] 覆盖；缺失 → 不判定 NaN"
        ),
        source="engine-defined",
    ),
    TypeRule(
        key="disruption",
        name_zh="腺体断裂",
        definition=(
            "break_count ≥ 1（断裂口径见指标注册表：Σ max(0, 实例骨架端点数 − 2)）。"
            "阈值 1 可用 overrides['disruption'] 覆盖；缺失 → 不判定 NaN"
        ),
        source="engine-defined",
    ),
    TypeRule(
        key="confluence",
        name_zh="腺体融合",
        definition=(
            "min_inter_instance_distance_px < 0.5 × 平均宽度（平均宽度优先取 instance_df 全体 "
            "`短轴(px)` 均值，缺列时回退 features['width_mean_px']；均无效 → 不判定 NaN）。"
            "系数 0.5 可用 overrides['confluence'] 覆盖，实际比较阈值 = 系数 × 平均宽度"
        ),
        source="engine-defined",
    ),
)


# ---------------------------------------------------------------------------
# 内部工具：数值解析与实例行模型（缺列 / 'N/A' 字符串一律 NaN，绝不抛异常）。
# ---------------------------------------------------------------------------
def _num(value) -> float:
    """任意输入 → 有限 float；None / bool / 非数值 / 非有限（含 'N/A'）→ NaN。"""
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _median(values: list[float]) -> float:
    """有限值中位数；无有限值 → NaN。"""
    finite = [item for item in values if math.isfinite(item)]
    if not finite:
        return float("nan")
    return float(np.median(finite))


def _instance_rows(instance_df: pd.DataFrame | None) -> list[dict]:
    """把 instance_df 展开为逐行数值模型；df 为空或缺列 → 空列表 / NaN，不崩溃。"""
    rows: list[dict] = []
    if instance_df is None or len(instance_df) == 0:
        return rows
    for _, row in instance_df.iterrows():
        rows.append({
            "relative": _num(row.get(COLUMN_RELATIVE)),
            "truncated": str(row.get(COLUMN_TRUNCATED, "")).strip() == "是",
            "minor": _num(row.get(COLUMN_MINOR)),
            "tau": _num(row.get(COLUMN_TAU)),
        })
    return rows


def _evidence(rule: TypeRule, present: bool, value: float, threshold: float, unit: str) -> TypeEvidence:
    return TypeEvidence(
        key=rule.key,
        present=bool(present),
        value=float(value),
        threshold=float(threshold),
        unit=unit,
        definition=rule.definition,
    )


def _not_judged(rule: TypeRule, unit: str, threshold: float = float("nan")) -> TypeEvidence:
    """缺失输入：present=False 且 value=NaN（阈值仍尽量记录，便于解释）。"""
    return _evidence(rule, False, float("nan"), threshold, unit)


# ---------------------------------------------------------------------------
# 各规则判定（features dict 优先；rows 为 instance_df 逐行模型，实例级回退用）。
# ---------------------------------------------------------------------------
def _eval_dropout(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    ga = _num(features.get("ga_percent"))
    reference = _num(features.get("reference_density_percent"))
    limit = threshold * reference
    if not math.isfinite(ga) or not math.isfinite(reference):
        return _not_judged(rule, "%", limit)
    return _evidence(rule, ga < limit, ga, limit, "%")


def _eval_shortening(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    ratio = _num(features.get("truncation_ratio"))
    if math.isfinite(ratio):
        # case 级：截断占比 ≥ 阈值（默认 0.2）。
        return _evidence(rule, ratio >= threshold, ratio, threshold, "无量纲")
    # instance 级：单实例相对延伸比 < 0.5（clinical `是否截断` 口径，阈值固定）。
    relative = features.get("relative_extension_ratio")
    if relative is None:
        relative = features.get(COLUMN_RELATIVE)
    relative = _num(relative)
    if math.isfinite(relative):
        return _evidence(rule, relative < SHORTENING_INSTANCE_RATIO, relative,
                         SHORTENING_INSTANCE_RATIO, "无量纲")
    if str(features.get(COLUMN_TRUNCATED, "")).strip() == "是":
        # 仅有 `是否截断` 旗标、无数值：present=True，value 以 NaN 如实记录。
        return _evidence(rule, True, float("nan"), SHORTENING_INSTANCE_RATIO, "无量纲")
    # case 级回退：对 instance_df 逐行评估，present=任一实例触发，value=最显著实例值。
    triggered = [
        row["relative"] for row in rows
        if math.isfinite(row["relative"]) and row["relative"] < SHORTENING_INSTANCE_RATIO
    ]
    if triggered:
        return _evidence(rule, True, min(triggered), SHORTENING_INSTANCE_RATIO, "无量纲")
    flagged = [row for row in rows if row["truncated"] and not math.isfinite(row["relative"])]
    if flagged:
        return _evidence(rule, True, float("nan"), SHORTENING_INSTANCE_RATIO, "无量纲")
    finite_all = [row["relative"] for row in rows if math.isfinite(row["relative"])]
    value = min(finite_all) if finite_all else float("nan")
    return _evidence(rule, False, value, SHORTENING_INSTANCE_RATIO, "无量纲")


def _eval_tortuosity(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    inflection = _num(features.get("inflection_count"))
    gate_open = (not math.isfinite(inflection)) or inflection >= TORTUOSITY_INFLECTION_MIN
    ti = _num(features.get("tortuosity_index"))
    if not math.isfinite(ti):
        # 单实例兼容：df 行字典只有 `弯曲度τ` → TI = τ − 1。
        tau = _num(features.get(COLUMN_TAU))
        if math.isfinite(tau):
            ti = tau - 1.0
    if math.isfinite(ti):
        return _evidence(rule, ti > threshold and gate_open, ti, threshold, "无量纲")
    # case 级回退：逐行评估单实例 TI = `弯曲度τ` − 1（无逐实例拐点数 → 拐点门仅用 case 级值）。
    tis = [row["tau"] - 1.0 for row in rows if math.isfinite(row["tau"])]
    if tis:
        value = max(tis)
        return _evidence(rule, value > threshold and gate_open, value, threshold, "无量纲")
    return _not_judged(rule, "无量纲", threshold)


def _eval_thickening(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    median = _num(features.get("width_median_px"))
    if not math.isfinite(median):
        median = _median([row["minor"] for row in rows])
    if not math.isfinite(median):
        return _not_judged(rule, "px")
    limit = threshold * median
    width = _num(features.get("width_mean_px"))
    if math.isfinite(width):
        return _evidence(rule, width > limit, width, limit, "px")
    # 单实例兼容：df 行字典只有 `短轴(px)`。
    minor_alias = _num(features.get(COLUMN_MINOR))
    if math.isfinite(minor_alias):
        return _evidence(rule, minor_alias > limit, minor_alias, limit, "px")
    # case 级回退：逐行评估，present=任一实例触发，value=最显著实例（宽度最大）取值。
    minors = [row["minor"] for row in rows if math.isfinite(row["minor"])]
    if minors:
        value = max(minors)
        return _evidence(rule, value > limit, value, limit, "px")
    return _not_judged(rule, "px", limit)


def _eval_beading(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    std = _num(features.get("width_profile_std"))
    mean = _num(features.get("width_mean_px"))
    if math.isfinite(std) and math.isfinite(mean) and mean > 0:
        cv = std / mean
    else:
        # 回退注册表 DI（diameter_deformation_index = std(w)/mean(w)，同一公式）。
        cv = _num(features.get("diameter_deformation_index"))
    if not math.isfinite(cv):
        return _not_judged(rule, "无量纲", threshold)
    peaks = _num(features.get("width_peaks"))
    peaks_ok = (not math.isfinite(peaks)) or peaks >= BEADING_PEAKS_MIN
    return _evidence(rule, cv > threshold and peaks_ok, cv, threshold, "无量纲")


def _eval_ghost(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    area = _num(features.get("ghost_area_px"))
    if not math.isfinite(area):
        return _not_judged(rule, "px", threshold)
    return _evidence(rule, area >= threshold, area, threshold, "px")


def _eval_disruption(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    count = _num(features.get("break_count"))
    if not math.isfinite(count):
        return _not_judged(rule, "个", threshold)
    return _evidence(rule, count >= threshold, count, threshold, "个")


def _eval_confluence(rule: TypeRule, features: dict, rows: list[dict], threshold: float) -> TypeEvidence:
    distance = _num(features.get("min_inter_instance_distance_px"))
    if not math.isfinite(distance):
        return _not_judged(rule, "px")
    minors = [row["minor"] for row in rows if math.isfinite(row["minor"])]
    base = float(np.mean(minors)) if minors else _num(features.get("width_mean_px"))
    if not math.isfinite(base) or base <= 0:
        return _not_judged(rule, "px")
    limit = threshold * base
    return _evidence(rule, distance < limit, distance, limit, "px")


_EVALUATORS = {
    "dropout": _eval_dropout,
    "shortening": _eval_shortening,
    "tortuosity": _eval_tortuosity,
    "thickening": _eval_thickening,
    "beading": _eval_beading,
    "ghost": _eval_ghost,
    "disruption": _eval_disruption,
    "confluence": _eval_confluence,
}


#: 乘法类规则（overrides 覆盖的是系数，实际比较阈值 = 系数 × 基准值）：
#: 系数必须 > 0（为 0 或负会把比较阈值归零 / 变号，判定失去意义）。
_MULTIPLICATIVE_RULES = frozenset({"dropout", "thickening", "confluence"})


def _resolve_thresholds(overrides: dict[str, float] | None) -> dict[str, float]:
    """合并默认阈值与 overrides；非法值 → ValueError。

    校验：未知 rule key、非数值、非有限、负值一律 ValueError；
    乘法类规则（dropout / thickening / confluence 的系数）还要求 > 0。
    """
    thresholds = dict(DEFAULT_THRESHOLDS)
    if overrides is None:
        return thresholds
    if not isinstance(overrides, dict):
        raise ValueError(
            f"overrides 必须是 dict[rule_key, threshold]，收到 {type(overrides).__name__}"
        )
    for key, value in overrides.items():
        if key not in thresholds:
            raise ValueError(f"未知分型规则 key: {key!r}；合法 key：{sorted(thresholds)}")
        try:
            threshold = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"overrides[{key!r}] 阈值必须是数值，收到 {value!r}") from exc
        if not math.isfinite(threshold):
            raise ValueError(f"overrides[{key!r}] 阈值必须是有限数值，收到 {value!r}")
        if threshold < 0.0:
            raise ValueError(f"overrides[{key!r}] 阈值不能为负，收到 {value!r}")
        if key in _MULTIPLICATIVE_RULES and threshold <= 0.0:
            raise ValueError(
                f"overrides[{key!r}] 是乘法类系数（实际比较阈值 = 系数 × 基准值），"
                f"必须 > 0，收到 {value!r}"
            )
        thresholds[key] = threshold
    return thresholds


def classify_instance(
    features: dict,
    overrides: dict[str, float] | None = None,
) -> list[TypeEvidence]:
    """对单个腺体实例的 features dict 逐条评估 8 类分型，返回固定顺序证据。

    缺失输入 → present=False 且 value=NaN；阈值覆盖规则见模块 docstring。
    单实例 dict 兼容 df 行键（`相对延伸比` / `是否截断` / `短轴(px)` / `弯曲度τ`）。
    """
    feats = dict(features) if features else {}
    thresholds = _resolve_thresholds(overrides)
    return [_EVALUATORS[rule.key](rule, feats, [], thresholds[rule.key]) for rule in TYPE_RULES]


def classify_case(
    instance_df: pd.DataFrame | None,
    case_features: dict,
    overrides: dict[str, float] | None = None,
) -> list[TypeEvidence]:
    """对整例（case_features + instance_df）逐条评估 8 类分型，返回固定顺序证据。

    shortening / tortuosity / thickening 的实例版：features dict 优先，缺失时对
    instance_df 逐行评估，返回 case 级证据（present=任一实例触发，
    value=最显著实例值）；df 缺列不崩溃（NaN 证据）。
    """
    feats = dict(case_features) if case_features else {}
    thresholds = _resolve_thresholds(overrides)
    rows = _instance_rows(instance_df)
    return [_EVALUATORS[rule.key](rule, feats, rows, thresholds[rule.key]) for rule in TYPE_RULES]
