"""RBAC（WP-7）：四角色权限矩阵 + fail-closed 鉴权。

角色层级（低到高）：tech ⊂ doctor ⊂ chief ⊂ admin，高级别继承低级别全部权限。
动作词表固定（11 个）；新增动作必须先入表、再配权，避免散落字符串比较。
"""
from __future__ import annotations

from typing import Literal

from ..runtime.errors import ConfigError

#: 系统角色（固定四种）
Role = Literal["tech", "doctor", "chief", "admin"]

#: 固定动作词表
ACTIONS: tuple[str, ...] = (
    "view_case",
    "run_inference",
    "edit_mask",
    "grade_case",
    "review_case",
    "export_report",
    "manage_atlas",
    "manage_calibration",
    "delete_case",
    "manage_users",
    "manage_audit",
)

_ACTION_SET = frozenset(ACTIONS)

#: 角色 -> 可执行动作集合（显式展开，便于审计比对）
PERMISSIONS: dict[str, set[str]] = {
    "tech": {
        "view_case",
        "run_inference",
        "export_report",
    },
    "doctor": {
        "view_case",
        "run_inference",
        "export_report",
        "edit_mask",
        "grade_case",
        "review_case",
    },
    "chief": {
        "view_case",
        "run_inference",
        "export_report",
        "edit_mask",
        "grade_case",
        "review_case",
        "manage_atlas",
        "manage_calibration",
        "delete_case",
    },
    "admin": {
        "view_case",
        "run_inference",
        "export_report",
        "edit_mask",
        "grade_case",
        "review_case",
        "manage_atlas",
        "manage_calibration",
        "delete_case",
        "manage_users",
        "manage_audit",
    },
}

# 模块级自检：权限矩阵必须严格构成 tech ⊆ doctor ⊆ chief ⊆ admin 的层级链。
assert PERMISSIONS["tech"] <= PERMISSIONS["doctor"] <= PERMISSIONS["chief"] <= PERMISSIONS["admin"]
assert set(PERMISSIONS["admin"]) == _ACTION_SET, "admin 必须覆盖全部动作词表"


def can(role: str, action: str) -> bool:
    """判断角色能否执行动作；未知角色 / 未知动作一律返回 False，不抛异常。

    不抛异常的原因（fail-closed 安全默认）：鉴权位于所有业务请求路径上，
    把「未知」一律当作「拒绝」，可以保证角色拼写错误、配置漏登记、新增动作
    未入表等情况安全地降级为拒绝，而不是靠异常打断业务，更不会意外放行。
    调用方只需按 False 走拒绝分支；确属配置错误应由 manage_users 管理流程发现。
    """
    if role not in PERMISSIONS:
        return False
    if action not in _ACTION_SET:
        return False
    return action in PERMISSIONS[role]


def require(role: str, action: str) -> None:
    """鉴权断言：无权时抛 ConfigError("角色 X 无权执行 Y")。"""
    if not can(role, action):
        raise ConfigError(f"角色 {role} 无权执行 {action}")
