"""治理接线钩子（hooks）：业务路径接入审计 / RBAC 的唯一入口。

把 governance 从「库」变成「生效路径」：inference / training / verify / versioning /
reporting / 各治理 CLI 一律经由本模块写入审计事件或做角色鉴权，不允许直接散落
new AuditLog(...) / rbac.require(...)。

审计文件位置
    ``settings.home / "logs" / "audit.jsonl"``（由 :func:`audit_log_path` 唯一定义，
    测试用 ``Settings(home=tmp_path)`` 即可整体隔离）。

两种失败语义
- **非阻塞**（``strict=False``，默认）：审计写入的任何异常（磁盘满 / 路径不可写 /
  序列化失败……）→ ``logger.error``（带堆栈摘要）后返回 ``None``，业务继续。
  审计失败不得打断临床工作流，但必须可见——运维通过 errors.jsonl / 控制台发现。
- **严格**（``strict=True``）：异常向上抛。用于治理关键事件（访问拒绝、权重转正）：
  审计失败比业务失败更严重——一个无法留痕的「拒绝」等于没有拒绝。

payload 与日志上下文
    ``audit_event`` 自动把当前日志上下文（run_id / case_id / stage / arch / scope，
    来自 ``runtime.logging.bind_context``）合并进 payload：审计条目天然可按
    case_id / run_id 与 mgd_events.jsonl 交叉检索。显式传入的 payload 键优先于上下文。
    payload 只放治理元数据（arch / sha 前缀 / 耗时 / 角色等），**不得携带 PHI 或图像**。

RBAC 接线
    :func:`require_or_audit` = ``rbac.require`` 失败时先写 ``access_denied`` 审计
    （strict=True，先于异常落地）再原样抛出 :class:`~mgd.core.runtime.errors.ConfigError`
    （MGDError 子类）——CLI 既有的 ``except MGDError`` 分支即可接住并映射退出码 2。
    当前 RBAC 覆盖范围：CLI 治理动作（评估 / 报告导出 / 图谱检索）；WebUI 已有口令层
    且为本机单用户，service/UI 层不重复设卡。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime.errors import ConfigError
from ..runtime.logging import get_log_context, get_logger
from .audit import AuditLog
from .rbac import require

logger = get_logger("governance.hooks")

#: 随 payload 自动携带的日志上下文字段（与 runtime.logging._CORE_FIELDS 对齐）
_CONTEXT_FIELDS: tuple[str, ...] = ("run_id", "case_id", "stage", "arch", "scope")

#: 访问拒绝事件的固定 event 名
ACCESS_DENIED_EVENT = "access_denied"


def audit_log_path(settings: Any) -> Path:
    """审计日志唯一位置：``settings.home / "logs" / "audit.jsonl"``。

    全部审计事件共用这一个文件（哈希链跨事件连续）；测试传 ``Settings(home=tmp)``
    即可与生产数据完全隔离。
    """
    return Path(settings.home) / "logs" / "audit.jsonl"


def _context_payload() -> dict[str, str]:
    """当前日志上下文的可序列化快照（None 字段不写入）。"""
    context = get_log_context()
    payload: dict[str, str] = {}
    for name in _CONTEXT_FIELDS:
        value = getattr(context, name)
        if value:
            payload[name] = value
    for key, value in context.extra:
        payload.setdefault(key, value)
    return payload


def audit_event(
    settings: Any,
    event: str,
    actor: str = "system",
    payload: dict | None = None,
    *,
    strict: bool = False,
) -> dict | None:
    """写一条治理审计事件，返回完整记录（含 seq/hash）；失败时按 strict 语义处理。

    - 文件：:func:`audit_log_path`；写入走 :class:`AuditLog`（JSONL 哈希链，
      append + fsync，断电安全）。
    - payload 自动合并当前日志上下文（case_id / run_id / stage / arch / scope），
      显式传入的键优先；NaN/±Inf 由 audit 层清洗为 None，保证文件是严格 JSON。
    - ``strict=False``（默认）：任何异常 → ``logger.error``（含堆栈摘要）→ 返回 None，
      业务不阻断；``strict=True``：异常向上抛（治理关键事件不允许「无声失败」）。
    """
    merged: dict = _context_payload()
    if payload:
        merged.update(payload)
    try:
        return AuditLog(audit_log_path(settings)).append(str(event), str(actor), merged)
    except Exception as exc:
        if strict:
            raise
        logger.error(
            "审计事件 %s 写入失败（非阻塞，业务继续）：%s: %s",
            event, type(exc).__name__, exc,
            exc_info=True,
        )
        return None


def require_or_audit(
    settings: Any,
    action: str,
    *,
    entry: str,
    actor: str = "local",
) -> None:
    """RBAC 鉴权 + 拒绝审计：无权时先落 ``access_denied`` 审计，再抛 ConfigError。

    - 有权：直接返回（无审计噪音）；
    - 无权：``audit_event("access_denied", payload={role, action, entry}, strict=True)``
      **先于异常**落地（哪怕紧接着的抛出被上层吞掉，拒绝本身也已留痕），然后原样
      向上抛 :class:`ConfigError`（MGDError 子类，CLI 既有 except 分支可接住）；
    - 审计写入失败（strict=True）会替代业务异常向上抛：无法留痕的拒绝比拒绝本身
      更严重。
    """
    try:
        require(str(getattr(settings, "role", "")), str(action))
    except ConfigError:
        audit_event(
            settings,
            ACCESS_DENIED_EVENT,
            actor=actor,
            payload={
                "role": str(getattr(settings, "role", "")),
                "action": str(action),
                "entry": str(entry),
            },
            strict=True,
        )
        raise
