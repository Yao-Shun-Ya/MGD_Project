"""WP-7 治理（governance）：审计日志 / 脱敏 / RBAC / 双人复核。

- audit:  JSONL 哈希链审计日志（防篡改、可验证），本包共用 JSONL 原语也在此定义；
- deid:   影像 EXIF 清除、DICOM 身份标签清除、假名化；
- rbac:   tech/doctor/chief/admin 四角色权限矩阵（fail-closed）；
- review: 双人复核与仲裁（review + arbitration 事件）。
"""
from .audit import (
    GENESIS,
    AuditLog,
    append_line,
    canonical_json,
    record_hash,
    read_jsonl,
    utc_now_iso,
)
from .deid import DEID_TAGS, pseudonymize, scrub_dicom_tags, scrub_image_exif
from .rbac import ACTIONS, PERMISSIONS, Role, can, require
from .review import REVIEW_DECISIONS, ReviewLog, ReviewRecord

__all__ = [
    # audit
    "AuditLog",
    "GENESIS",
    "append_line",
    "canonical_json",
    "record_hash",
    "read_jsonl",
    "utc_now_iso",
    # deid
    "DEID_TAGS",
    "pseudonymize",
    "scrub_dicom_tags",
    "scrub_image_exif",
    # rbac
    "ACTIONS",
    "PERMISSIONS",
    "Role",
    "can",
    "require",
    # review
    "REVIEW_DECISIONS",
    "ReviewLog",
    "ReviewRecord",
]
