"""WP-6 相似病例检索索引：余弦距离 + 原子 JSON 持久化。

小规模病例库的内存检索索引（不引入向量数据库依赖）：
- :meth:`AtlasIndex.add` 维度严格校验，重复 case_id 覆盖并记 warning；
- :meth:`AtlasIndex.search` 余弦距离升序返回 top-k（零向量 / 含非有限值的向量
  → 距离 1.0，绝不崩溃；并列距离按 case_id 稳定排序）；
- :meth:`AtlasIndex.to_json` 原子写（tmp + ``os.replace``），JSON 含
  ``feature_names`` 与 ``version``；meta 中的 numpy 标量 / 数组 / NaN / ±inf
  一律归一为严格合法 JSON（非有限浮点 → ``null``），序列化失败时清理 tmp
  不留残留；:meth:`AtlasIndex.load` 严格校验版本与向量维度，
  不符 → ValueError。
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np

from ..runtime.logging import get_logger
from .features import FEATURE_NAMES

logger = get_logger("atlas.index")

#: 索引文件格式版本（结构变更时升级；load 严格校验，不符 → ValueError）。
INDEX_VERSION = 1


def _cosine_distance(query: np.ndarray, stored: np.ndarray) -> float:
    """余弦距离 = 1 − cos(query, stored)；零向量 / 非有限值 → 1.0（不崩溃）。"""
    norm_query = float(np.linalg.norm(query))
    norm_stored = float(np.linalg.norm(stored))
    if (
        not math.isfinite(norm_query) or not math.isfinite(norm_stored)
        or norm_query <= 0.0 or norm_stored <= 0.0
    ):
        return 1.0
    cosine = float(np.dot(query, stored) / (norm_query * norm_stored))
    if not math.isfinite(cosine):
        return 1.0
    cosine = min(1.0, max(-1.0, cosine))  # 抑制浮点溢出，距离落在 [0, 2]
    return 1.0 - cosine


def _vector_to_jsonable(array: np.ndarray) -> list:
    """非有限值 → None（保证严格合法 JSON）；load 时还原为 NaN。"""
    return [float(value) if math.isfinite(value) else None for value in array]


def _sanitize_finite(value):
    """递归把非有限浮点（NaN / ±inf）替换为 None，保证严格合法 JSON。

    处理 dict / list / tuple 结构与 Python float（含 ``np.float64`` 这类
    float 子类）；numpy 标量 / ndarray 交给 ``_json_default`` 兜底。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_finite(item) for item in value]
    return value


def _json_default(value):
    """json.dump 兜底序列化：numpy 标量 → 原生 float/int/bool、ndarray → list、
    其余对象 → str（绝不因 meta 含 numpy 类型而崩溃）。"""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _sanitize_finite(value.tolist())
    return str(value)


class AtlasIndex:
    """形态学特征向量的相似病例检索索引（余弦距离，JSON 持久化）。"""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        names = (
            [str(name) for name in feature_names]
            if feature_names is not None
            else [str(name) for name in FEATURE_NAMES]
        )
        if not names:
            raise ValueError("feature_names 不能为空列表")
        if len(set(names)) != len(names):
            raise ValueError(f"feature_names 存在重复项: {names}")
        self._feature_names = names
        self._entries: dict[str, tuple[np.ndarray, dict]] = {}

    @property
    def feature_names(self) -> list[str]:
        """索引特征名（向量维度顺序）。"""
        return list(self._feature_names)

    def add(self, case_id: str, vector: np.ndarray, meta: dict) -> None:
        """加入 / 覆盖一个病例向量；维度不符 → ValueError，重复 case_id 覆盖并 warning。"""
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case_id 必须是非空字符串，收到 {case_id!r}")
        array = np.asarray(vector, dtype=np.float64).ravel()
        if array.size != len(self._feature_names):
            raise ValueError(
                f"向量维度 {array.size} 与索引维度 {len(self._feature_names)} 不一致"
                f"（case_id={case_id!r}，feature_names={self._feature_names}）"
            )
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise ValueError(f"meta 必须是 dict，收到 {type(meta).__name__}")
        if case_id in self._entries:
            logger.warning("病例 %r 已存在于索引中，覆盖旧向量与元数据", case_id)
        # 深拷贝：调用方随后修改 meta（含嵌套结构）不得影响索引内已存数据
        self._entries[case_id] = (array.copy(), copy.deepcopy(meta))

    def search(self, vector: np.ndarray, k: int = 5) -> list[dict]:
        """余弦距离升序返回 top-k：[{"case_id", "distance", "meta"}, ...]。

        零向量 / 含非有限值的向量按距离 1.0 处理；k 截断，k<=0 返回空列表。
        """
        if k is None or int(k) <= 0:
            return []
        query = np.asarray(vector, dtype=np.float64).ravel()
        if query.size != len(self._feature_names):
            raise ValueError(
                f"查询向量维度 {query.size} 与索引维度 {len(self._feature_names)} 不一致"
            )
        results = [
            {
                "case_id": case_id,
                "distance": _cosine_distance(query, stored),
                "meta": dict(meta),
            }
            for case_id, (stored, meta) in self._entries.items()
        ]
        results.sort(key=lambda item: (item["distance"], item["case_id"]))
        return results[: int(k)]

    def to_json(self, path) -> Path:
        """原子写（tmp + os.replace）到 path，返回最终 Path。

        meta / 向量先经 :func:`_sanitize_finite` 把非有限浮点（NaN / ±inf）
        转为 None，序列化再挂 :func:`_json_default` 兜底 numpy 类型，
        产物保证为严格合法 JSON。dump + replace 全程包裹 try/finally：
        任何一步失败都清理 tmp（``missing_ok=True``）后原样抛出，
        绝不在目标目录留下 .tmp 残留。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "feature_names": list(self._feature_names),
            "items": [
                {
                    "case_id": case_id,
                    "vector": _vector_to_jsonable(array),
                    "meta": _sanitize_finite(meta),
                }
                for case_id, (array, meta) in self._entries.items()
            ],
        }
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(
                    payload, handle, ensure_ascii=False, indent=2, default=_json_default
                )
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path) -> "AtlasIndex":
        """从 JSON 读取索引；version 不符或向量维度与索引不符 → ValueError。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"索引文件 version={version!r} 与当前格式 version={INDEX_VERSION} 不符，"
                "请重建索引（重新 to_json）后再加载"
            )
        index = cls(feature_names=data.get("feature_names"))
        for item in data.get("items", []):
            array = np.asarray(
                [float("nan") if value is None else float(value) for value in item["vector"]],
                dtype=np.float64,
            ).ravel()
            if array.size != len(index._feature_names):
                raise ValueError(
                    f"索引文件中病例 {item.get('case_id')!r} 的向量维度 {array.size} "
                    f"与索引维度 {len(index._feature_names)} 不一致，索引文件已损坏"
                )
            index._entries[str(item["case_id"])] = (array, dict(item.get("meta") or {}))
        return index

    def __len__(self) -> int:
        return len(self._entries)
