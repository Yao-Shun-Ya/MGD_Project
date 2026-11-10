"""WP-8 纵向随访子系统：sqlite3 临床存储 + 图像配准 + 腺体实例跨访视匹配。

- store.ClinicStore：随访数据库（患者 / 就诊 / 测量 / 复核），标准库 sqlite3，
  全部写操作走事务，foreign_keys=ON，NaN 统一存 NULL（项目 NaN 纪律）。
- register.register_pair：ECC（MOTION_EUCLIDEAN）配准，为前后对比做前置对齐。
- match.match_instances：把两次访视的腺体实例 DataFrame 按「质心 + 形状」成本做
  最优二分匹配，输出 matched / disappeared / new 及逐实例差分。
"""
from __future__ import annotations

from .match import instance_centroids, match_instances
from .register import register_pair
from .store import SCHEMA_VERSION, ClinicStore

__all__ = [
    "SCHEMA_VERSION",
    "ClinicStore",
    "register_pair",
    "instance_centroids",
    "match_instances",
]
