"""WP-1 可追溯指标注册表。

每个量化指标都以 :class:`MetricSpec` 声明：中文别名、定义公式、单位、文献出处
（source）、所需输入（requires）与方向性（higher_is_better）。UI / 报告层只允许
通过本注册表取指标元数据，保证「报告里出现的每一个数字都能追到公式与出处」。

source 词表（只允许三类）：
- ``"Arita et al. 2008"``：Arita 等人 2008 年 MG 分级 / 腺体缺失率口径；
- ``"Pult & Nichols 2016"``：Pult & Nichols 2016 年睑板腺形态计量学口径
  （弯曲度、管径变形、信号 / 对比度等形态计量指标族）；
- ``"engine-defined"``：工程自定义（纯计算构造，无直接文献口径），
  包括 width_mean_px、gland_eyelid_distance_px、symmetry_index、zonal_density、
  endpoint_count、branch_count、break_count、instance_count。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..runtime.errors import ConfigError
from ..runtime.logging import get_logger

logger = get_logger("quantification.registry")

#: source 的合法词表
ALLOWED_SOURCES = ("Arita et al. 2008", "Pult & Nichols 2016", "engine-defined")


@dataclass(frozen=True)
class MetricSpec:
    """单个量化指标的可追溯元数据。"""

    key: str
    name_zh: str
    definition: str
    unit: str
    source: str
    requires: tuple[str, ...]
    higher_is_better: bool | None


METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register(spec: MetricSpec) -> None:
    """注册一个指标定义；key 重复时抛 :class:`ConfigError`（禁止静默覆盖定义）。"""
    if not isinstance(spec, MetricSpec):
        raise ConfigError(f"register 只接受 MetricSpec，收到 {type(spec).__name__}")
    key = str(spec.key).strip()
    if not key:
        raise ConfigError("MetricSpec.key 不能为空")
    if spec.source not in ALLOWED_SOURCES:
        raise ConfigError(
            f"指标 {key!r} 的 source={spec.source!r} 不在允许词表 {ALLOWED_SOURCES} 内"
        )
    if key in METRIC_REGISTRY:
        existing = METRIC_REGISTRY[key]
        raise ConfigError(
            f"指标 {key!r} 已有注册定义（name_zh={existing.name_zh!r}, "
            f"source={existing.source!r}）；禁止重复注册覆盖，如需修改请直接改注册表源码"
        )
    METRIC_REGISTRY[key] = spec
    logger.debug("已注册指标 %s（%s, source=%s）", key, spec.name_zh, spec.source)


def get_spec(key: str) -> MetricSpec:
    """按 key 取指标定义；未知 key 抛 :class:`ConfigError` 并附可选 key 列表。"""
    if key not in METRIC_REGISTRY:
        raise ConfigError(f"未知指标 key: {key!r}；当前可选：{sorted(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[key]


def all_specs() -> list[MetricSpec]:
    """全部指标定义，按 key 排序。"""
    return [METRIC_REGISTRY[key] for key in sorted(METRIC_REGISTRY)]


def describe(key: str) -> dict:
    """返回指标 key 的完整元数据 dict（供 UI / 报告直接展示）。"""
    return asdict(get_spec(key))


# ---------------------------------------------------------------------------
# 内置指标定义（16 项）。definition 必须写清公式；source 必须落在词表内。
# ---------------------------------------------------------------------------
register(MetricSpec(
    key="ga_percent",
    name_zh="腺体面积比 GA (%)",
    definition=(
        "GA = 100 × |G∩E| / |E|，G 为腺体掩膜、E 为眼睑 ROI 掩膜；"
        "即眼睑区内腺体保留面积占比（Arita 2008 腺体缺失率 R_loss 的补数口径）"
    ),
    unit="%",
    source="Arita et al. 2008",
    requires=("gland_mask", "eyelid_mask"),
    higher_is_better=True,
))
register(MetricSpec(
    key="diameter_deformation_index",
    name_zh="管径变形指数 DI",
    definition=(
        "DI = std(w) / mean(w)，w 为沿骨架点序的局部宽度序列"
        "（width_profile：2 × 骨架点距离变换值）；衡量腺体粗细不均程度"
    ),
    unit="无量纲",
    source="Pult & Nichols 2016",
    requires=("width_profile",),
    higher_is_better=False,
))
register(MetricSpec(
    key="tortuosity_arc_chord",
    name_zh="弯曲度 τ（弧长/弦长）",
    definition=(
        "τ = L_arc / L_chord，L_arc 为骨架主路径逐段欧氏弧长之和，"
        "L_chord 为首末端点直线距离；直线 τ=1"
    ),
    unit="无量纲",
    source="Pult & Nichols 2016",
    requires=("skeleton_path",),
    higher_is_better=None,
))
register(MetricSpec(
    key="tortuosity_index",
    name_zh="弯曲指数 TI",
    definition="TI = τ − 1，τ 为弧长/弦长弯曲度；TI=0 表示直线，越大越弯曲",
    unit="无量纲",
    source="Pult & Nichols 2016",
    requires=("skeleton_path",),
    higher_is_better=None,
))
register(MetricSpec(
    key="inflection_count",
    name_zh="曲率拐点数",
    definition=(
        "骨架主路径相邻三点有符号曲率 κ（Menger 口径）；统计符号相反的连续同号段"
        "之间的转折次数，且转折两侧段的最大 |κ| 均需 > min_kappa（默认 0.05 px⁻¹）"
    ),
    unit="个",
    source="Pult & Nichols 2016",
    requires=("skeleton_path",),
    higher_is_better=None,
))
register(MetricSpec(
    key="fractal_dimension",
    name_zh="盒计数分形维数",
    definition=(
        "D = 斜率( log N(ε) ~ log(1/ε) )，N(ε) 为边长 ε 的盒子覆盖前景的个数；"
        "ε 取 2 的幂递减（2 ≤ ε ≤ min(H,W)/2），np.polyfit 线性拟合"
    ),
    unit="无量纲",
    source="Pult & Nichols 2016",
    requires=("gland_mask",),
    higher_is_better=None,
))
register(MetricSpec(
    key="signal_index",
    name_zh="信号指数 SI",
    definition=(
        "SI = ( mean(I[G]) − mean(I[R]) ) / 255，R 为腺体掩膜按椭圆核膨胀 ring_px "
        "像素后减去腺体本身得到的环形背景带"
    ),
    unit="无量纲",
    source="Pult & Nichols 2016",
    requires=("gray_image", "gland_mask"),
    higher_is_better=True,
))
register(MetricSpec(
    key="local_contrast",
    name_zh="局部对比度 LC",
    definition=(
        "LC = mean(I[G]) − mean(I[R])，环 R 定义同 signal_index，"
        "但不做 255 归一化（保留原始灰度差）"
    ),
    unit="灰度级",
    source="Pult & Nichols 2016",
    requires=("gray_image", "gland_mask"),
    higher_is_better=True,
))
register(MetricSpec(
    key="width_mean_px",
    name_zh="平均管宽 (px)",
    definition=(
        "mean(w)，w 为 width_profile 输出的骨架点宽度序列"
        "（2 × DIST_L2 距离变换在骨架点上的取值）；未标定时仅 px 口径"
    ),
    unit="px",
    source="engine-defined",
    requires=("gland_mask",),
    higher_is_better=None,
))
register(MetricSpec(
    key="gland_eyelid_distance_px",
    name_zh="腺体-眼睑中线距离 (px)",
    definition=(
        "腺体骨架端点（3×3 邻域内恰有 1 个骨架邻居的点）到眼睑外接框垂直中线 "
        "x_c=(x_min+x_max)/2 的横向距离 |x − x_c| 的均值"
    ),
    unit="px",
    source="engine-defined",
    requires=("gland_mask", "eyelid_mask"),
    higher_is_better=None,
))
register(MetricSpec(
    key="symmetry_index",
    name_zh="左右对称指数",
    definition=(
        "SI = |a − b| / mean(a, b) = 2|a − b| / (a + b)，"
        "对左右眼同名指标计算相对差异；均值 0 无定义（NaN）"
    ),
    unit="无量纲",
    source="engine-defined",
    requires=("left_value", "right_value"),
    higher_is_better=False,
))
register(MetricSpec(
    key="zonal_density",
    name_zh="分区腺体密度",
    definition=(
        "ρ_k = 100 × |G∩E∩Z_k| / |E∩Z_k|，Z_k 为眼睑外接框宽度等分的第 k 个分区；"
        "某分区眼睑为空则该区 NaN"
    ),
    unit="%",
    source="engine-defined",
    requires=("gland_mask", "eyelid_mask"),
    higher_is_better=True,
))
register(MetricSpec(
    key="endpoint_count",
    name_zh="骨架端点数",
    definition="骨架中 3×3 邻域（不含自身）内恰有 1 个骨架像素的点的个数",
    unit="个",
    source="engine-defined",
    requires=("skeleton",),
    higher_is_better=None,
))
register(MetricSpec(
    key="branch_count",
    name_zh="骨架分叉点数",
    definition="骨架中 3×3 邻域（不含自身）内 ≥3 个骨架像素的点的个数",
    unit="个",
    source="engine-defined",
    requires=("skeleton",),
    higher_is_better=None,
))
register(MetricSpec(
    key="break_count",
    name_zh="腺体断裂数",
    definition=(
        "Σ_i max(0, E_i − 2)，E_i 为第 i 个连通实例骨架的端点数；"
        "正常单条腺体端点数=2，每多出一对端点记一次断裂"
    ),
    unit="个",
    source="engine-defined",
    requires=("gland_mask",),
    higher_is_better=False,
))
register(MetricSpec(
    key="instance_count",
    name_zh="腺体实例数",
    definition=(
        "面积 ≥ min_area（默认 120 px）的连通实例个数"
        "（split_instances method='cc' 口径）"
    ),
    unit="个",
    source="engine-defined",
    requires=("gland_mask",),
    higher_is_better=None,
))
