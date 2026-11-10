"""WP-6 相似病例检索特征向量（固定顺序、版本化）。

``FEATURE_NAMES`` 的顺序即向量维度顺序，随 :class:`~.index.AtlasIndex` 一起写入
JSON（``version`` 字段标记格式版本），一经发布不得调序 / 增删——需要扩展时追加到
尾部并升级版本号。``case_vector`` 把 features dict 投影为 float64 向量（缺失键 /
非数值 / 非有限值 → NaN），``vector_to_dict`` 为其逆变换（检索结果回填 features 用）。

与指标注册表 / clinical 的键名对应关系（回填与解释用）：
- ga_percent / instance_count / width_mean_px / tortuosity_index /
  inflection_count / fractal_dimension / signal_index / break_count：
  与 ``quantification.registry`` 同名指标 key 完全一致；
- width_cv：宽度剖面变异系数，公式与注册表 diameter_deformation_index（DI）
  一致（std(w)/mean(w)）；为向量命名简洁单列此 key；
- truncation_ratio：``imaging.clinical.truncation_ratio``（截断实例占比）；
- mean_curvature：实例级平均曲率 κ 均值（instance_df `平均曲率κ` 列均值）；
- min_inter_instance_distance_px：实例间最小间距（engine-defined）。
"""
from __future__ import annotations

import numpy as np

#: 检索特征向量维度顺序（版本化：调序 / 增删必须升 FEATURE_NAMES_VERSION）。
FEATURE_NAMES: tuple[str, ...] = (
    "ga_percent",
    "instance_count",
    "width_mean_px",
    "width_cv",
    "tortuosity_index",
    "inflection_count",
    "fractal_dimension",
    "signal_index",
    "break_count",
    "truncation_ratio",
    "mean_curvature",
    "min_inter_instance_distance_px",
)

#: FEATURE_NAMES 布局版本（写入索引 JSON，load 时与 AtlasIndex 格式版本共同校验）。
FEATURE_NAMES_VERSION = 1


def case_vector(features: dict) -> np.ndarray:
    """features dict → 固定顺序 float64 向量。

    缺失键 / None / bool（含 ``np.bool_``）/ 非数值字符串 / 非有限值（inf、NaN）
    一律 → np.nan，绝不用 0 冒充测量值；``features=None`` 视为全 NaN。
    """
    vector = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float64)
    if not features:
        return vector
    for index, name in enumerate(FEATURE_NAMES):
        if name not in features:
            continue
        value = features[name]
        if value is None or isinstance(value, (bool, np.bool_)):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(numeric):
            vector[index] = numeric
    return vector


def vector_to_dict(vector) -> dict:
    """case_vector 的逆变换：向量 → {feature_name: float}（NaN 原样保留）。

    长度与 FEATURE_NAMES 不一致 → ValueError。
    """
    array = np.asarray(vector, dtype=np.float64).ravel()
    if array.size != len(FEATURE_NAMES):
        raise ValueError(
            f"向量长度 {array.size} 与 FEATURE_NAMES 长度 {len(FEATURE_NAMES)} 不一致"
        )
    return {name: float(array[index]) for index, name in enumerate(FEATURE_NAMES)}
