"""WP-6 形态学分型与相似病例检索层（atlas）。

组成：
- :mod:`.typology`：8 类形态学分型规则（dropout / shortening / tortuosity /
  thickening / beading / ghost / disruption / confluence），逐条返回可追溯证据
  （实测值 value + 实际比较阈值 threshold + 判定条件 definition + 来源 source）；
- :mod:`.features`：相似病例检索特征向量（固定顺序、版本化的 FEATURE_NAMES，
  缺失维度以 NaN 表达，绝不用 0 冒充）；
- :mod:`.index`：AtlasIndex 余弦距离检索索引（原子 JSON 持久化，版本严格校验）。

统一约定：缺失输入 → present=False 且 value=NaN，绝不崩溃、绝不用 0 冒充；
分型与特征键名复用 ``quantification.registry`` 已注册指标 key 与
``imaging.clinical.INSTANCE_COLUMNS`` 列名（导入期校验对齐）。
"""
from __future__ import annotations

from .features import FEATURE_NAMES, FEATURE_NAMES_VERSION, case_vector, vector_to_dict
from .index import INDEX_VERSION, AtlasIndex
from .typology import (
    DEFAULT_THRESHOLDS,
    TYPE_RULES,
    TypeEvidence,
    TypeRule,
    classify_case,
    classify_instance,
)

__all__ = [
    # typology
    "DEFAULT_THRESHOLDS",
    "TYPE_RULES",
    "TypeEvidence",
    "TypeRule",
    "classify_case",
    "classify_instance",
    # features
    "FEATURE_NAMES",
    "FEATURE_NAMES_VERSION",
    "case_vector",
    "vector_to_dict",
    # index
    "INDEX_VERSION",
    "AtlasIndex",
]
