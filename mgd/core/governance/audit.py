"""JSONL 哈希链审计日志（WP-7）：防篡改、可验证、断电安全。

设计要点：
- 每条记录结构：{"seq", "ts", "event", "actor", "payload", "prev_hash", "hash"}；
- hash = sha256(prev_hash + canonical_json(去掉 hash 字段的记录))，
  canonical_json = sort_keys=True + ensure_ascii=False（同一条记录在任何机器上算出的哈希一致）；
- 首条记录 prev_hash = "GENESIS"；
- append 使用追加模式 + 显式 flush + fsync，seq 取文件现有最大 seq + 1；
- 文件损坏行（半行 JSON / 非对象 / 缺字段）在读取时跳过并 logger.warning，不让业务读取中断；
- verify() 全链复核：任何一条被改写、删除或乱序，都会在断点 seq 处暴露。

本模块同时承载 governance 包共用的 JSONL 原语（canonical_json / utc_now_iso / read_jsonl），
review.py 复用同一套序列化与容错读取，保证治理类日志格式一致。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..runtime.logging import get_logger

_LOG = get_logger("governance.audit")

#: 首条记录的 prev_hash
GENESIS = "GENESIS"

#: 审计记录必备字段（缺任一字段的行视为损坏行）
REQUIRED_FIELDS: tuple[str, ...] = ("seq", "ts", "event", "actor", "payload", "prev_hash", "hash")


def utc_now_iso() -> str:
    """ISO 8601 UTC 时间串（秒级精度，带 +00:00 时区）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_finite(value: Any) -> Any:
    """递归把 NaN/±Inf 转成 None，保证审计文件是严格 JSON（不留 NaN 字面量）。"""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {key: _sanitize_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_finite(item) for item in value]
    return value


def canonical_json(payload: Any) -> str:
    """跨平台稳定的 JSON 序列化：键排序 + 不转义非 ASCII + 非有限值转 None。"""
    return json.dumps(_sanitize_finite(payload), sort_keys=True, ensure_ascii=False, allow_nan=False)


def record_hash(prev_hash: str, record: dict) -> str:
    """链式哈希：sha256(prev_hash + canonical_json(记录去掉 hash 字段))。"""
    body = {key: value for key, value in record.items() if key != "hash"}
    return hashlib.sha256((prev_hash + canonical_json(body)).encode("utf-8")).hexdigest()


def read_jsonl(
    path: str | Path,
    *,
    what: str = "记录",
    required: tuple[str, ...] = (),
) -> list[dict]:
    """容错读取 JSONL 文件：损坏行跳过并 warning，绝不中断业务读取。

    - 不存在的文件视为空；
    - 半行 JSON / 非法 JSON / 非 JSON 对象 / 缺少 required 字段的行一律跳过并 logger.warning。
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                _LOG.warning("%s 第 %d 行损坏（%s），已跳过", what, lineno, exc)
                continue
            if not isinstance(record, dict):
                _LOG.warning("%s 第 %d 行不是 JSON 对象，已跳过", what, lineno)
                continue
            missing = [field for field in required if field not in record]
            if missing:
                _LOG.warning("%s 第 %d 行缺少字段 %s，已跳过", what, lineno, missing)
                continue
            records.append(record)
    return records


def append_line(path: str | Path, line: str) -> None:
    """向 JSONL 追加一行（含换行）：追加模式 + 显式 flush + fsync。

    若文件末尾缺少换行符（例如上次进程崩溃留下的半行 JSON），先补一个换行，
    避免新记录与损坏残片拼在同一行、进而被读取端一并丢弃。
    """
    file_path = Path(path)
    needs_newline = False
    if file_path.exists() and file_path.stat().st_size > 0:
        with open(file_path, "rb") as probe:
            probe.seek(-1, os.SEEK_END)
            needs_newline = probe.read(1) != b"\n"
    with open(file_path, "a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class AuditLog:
    """追加式 JSONL 审计日志，带 sha256 哈希链。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 写入
    def append(self, event: str, actor: str, payload: dict) -> dict:
        """追加一条审计记录并返回完整记录（含 seq/hash）。

        seq 取文件现有最大 seq + 1；prev_hash 取最后一条可读记录的 hash（空文件为 GENESIS）。
        追加模式写入 + 显式 flush + fsync，保证进程崩溃时记录不丢。
        """
        records = self._read_records()
        seq = max((int(rec["seq"]) for rec in records), default=0) + 1
        prev_hash = records[-1]["hash"] if records else GENESIS
        record = {
            "seq": seq,
            "ts": utc_now_iso(),
            "event": str(event),
            "actor": str(actor),
            "payload": dict(payload),
            "prev_hash": prev_hash,
        }
        record["hash"] = record_hash(prev_hash, record)
        append_line(self.path, canonical_json(record))
        return record

    # ------------------------------------------------------------------ 读取
    def _read_records(self) -> list[dict]:
        """读取全部可解析记录（损坏行由 read_jsonl 跳过并告警）。"""
        return read_jsonl(self.path, what="审计日志", required=REQUIRED_FIELDS)

    def verify(self) -> tuple[bool, str]:
        """全链校验。返回 (True, "ok") 或 (False, "chain broken at seq=N ...")。"""
        records = self._read_records()
        prev_hash = GENESIS
        base_seq: int | None = None
        for index, record in enumerate(records):
            seq = int(record["seq"])
            if base_seq is None:
                base_seq = seq
            if seq != base_seq + index:
                return False, f"chain broken at seq={seq}: seq 不连续（期望 {base_seq + index}）"
            if record["prev_hash"] != prev_hash:
                return False, f"chain broken at seq={seq}: prev_hash 与上一条记录不衔接"
            expected = record_hash(record["prev_hash"], record)
            if record["hash"] != expected:
                return False, f"chain broken at seq={seq}: 记录内容哈希不匹配（疑似被篡改）"
            prev_hash = record["hash"]
        return True, "ok"

    def tail(self, n: int = 50) -> list[dict]:
        """返回最后 n 条可读记录（按文件顺序，最新在后）。"""
        if n <= 0:
            return []
        return self._read_records()[-n:]

    def __len__(self) -> int:
        return len(self._read_records())
