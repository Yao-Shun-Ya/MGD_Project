"""双人复核与仲裁（WP-7）。

流程：两位复核者各自对同一病例给出 agree/disagree 意见；
- 意见一致 -> 直接通过，无需仲裁；
- 意见不一致（且 >= 2 人）-> needs_arbitration 为 True，由更高权限者 arbitrate 给出终审意见。

存储：JSONL，复用 audit.py 的 canonical_json / utc_now_iso / read_jsonl 原语与原子写入风格。
同一 reviewer 对同一病例重复 record 时覆盖旧记录（以最新为准）并 logger.warning。
"""
from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ..runtime.errors import ConfigError
from ..runtime.logging import get_logger
from .audit import append_line, canonical_json, read_jsonl, utc_now_iso

_LOG = get_logger("governance.review")

#: 复核意见的合法取值
REVIEW_DECISIONS: tuple[str, ...] = ("agree", "disagree")

_REQUIRED_ENTRY_FIELDS: tuple[str, ...] = ("event", "case_id")


@dataclass
class ReviewRecord:
    """一条复核意见。"""

    case_id: str
    reviewer_id: str
    decision: str  # "agree" | "disagree"
    comment: str
    ts: str


class ReviewLog:
    """JSONL 复核日志：review 条目 + arbitration 条目。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 读取
    def _read_entries(self) -> list[dict]:
        return read_jsonl(self.path, what="复核日志", required=_REQUIRED_ENTRY_FIELDS)

    @staticmethod
    def _is_review_for(entry: dict, case_id: str, reviewer_id: str) -> bool:
        return (
            entry.get("event") == "review"
            and entry.get("case_id") == case_id
            and entry.get("reviewer_id") == reviewer_id
        )

    # ------------------------------------------------------------------ 写入
    def _atomic_append(self, entry: dict) -> None:
        """原子追加：复用 audit.append_line（整行一次写 + flush + fsync + 换行守卫）。"""
        append_line(self.path, canonical_json(entry))

    def _atomic_rewrite(self, entries: list[dict]) -> None:
        """原子重写（用于覆盖旧记录）：写临时文件后 os.replace，杜绝半写状态。"""
        tmp_path = self.path.with_name(f"{self.path.name}.tmp{os.getpid()}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(canonical_json(entry) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            with suppress(OSError):
                tmp_path.unlink()

    # ------------------------------------------------------------------ API
    def record(
        self,
        case_id: str,
        reviewer_id: str,
        decision: str,
        comment: str = "",
    ) -> ReviewRecord:
        """记录一条复核意见；同一 reviewer 对同一 case 重复记录时覆盖旧记录（以最新为准）。"""
        case_id = str(case_id)
        reviewer_id = str(reviewer_id)
        if decision not in REVIEW_DECISIONS:
            raise ConfigError(f"复核意见必须是 {REVIEW_DECISIONS} 之一，收到：{decision!r}")
        entry = {
            "event": "review",
            "case_id": case_id,
            "reviewer_id": reviewer_id,
            "decision": decision,
            "comment": str(comment),
            "ts": utc_now_iso(),
        }
        entries = self._read_entries()
        superseded = [e for e in entries if self._is_review_for(e, case_id, reviewer_id)]
        if superseded:
            _LOG.warning(
                "复核覆盖：reviewer=%s 对 case=%s 的旧意见 (%s) 被最新意见 (%s) 取代",
                reviewer_id,
                case_id,
                superseded[-1].get("decision"),
                decision,
            )
            self._atomic_rewrite([e for e in entries if not self._is_review_for(e, case_id, reviewer_id)] + [entry])
        else:
            self._atomic_append(entry)
        return ReviewRecord(
            case_id=case_id,
            reviewer_id=reviewer_id,
            decision=decision,
            comment=entry["comment"],
            ts=entry["ts"],
        )

    def decisions(self, case_id: str) -> dict[str, str]:
        """当前各复核者的最新意见：reviewer_id -> decision（文件顺序后者覆盖前者）。"""
        latest: dict[str, str] = {}
        for entry in self._read_entries():
            if entry.get("event") == "review" and entry.get("case_id") == str(case_id):
                latest[str(entry["reviewer_id"])] = str(entry["decision"])
        return latest

    def needs_arbitration(self, case_id: str) -> bool:
        """是否需要仲裁：意见数 >= 2 且存在不一致。"""
        values = set(self.decisions(case_id).values())
        return len(values) >= 2

    def arbitrate(self, case_id: str, arbiter_id: str, final_decision: str, comment: str = "") -> dict:
        """仲裁：仅当 needs_arbitration 为 True 才允许，否则 ConfigError。

        仲裁条目 event="arbitration" 追加进日志，包含终审意见与仲裁人。
        """
        case_id = str(case_id)
        if not self.needs_arbitration(case_id):
            raise ConfigError(f"病例 {case_id} 无需仲裁：复核意见一致或不足两条")
        if final_decision not in REVIEW_DECISIONS:
            raise ConfigError(f"终审意见必须是 {REVIEW_DECISIONS} 之一，收到：{final_decision!r}")
        entry = {
            "event": "arbitration",
            "case_id": case_id,
            "arbiter_id": str(arbiter_id),
            "final_decision": final_decision,
            "comment": str(comment),
            "ts": utc_now_iso(),
        }
        self._atomic_append(entry)
        return entry

    def history(self, case_id: str) -> list[dict]:
        """病例的复核 + 仲裁全部条目，按文件顺序（最早在前）。"""
        case_id = str(case_id)
        return [entry for entry in self._read_entries() if entry.get("case_id") == case_id]
