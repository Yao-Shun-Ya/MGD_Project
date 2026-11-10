"""随访临床数据库（WP-8）。

设计要点：
- 只用标准库 sqlite3，禁止 SQLAlchemy；连接 `check_same_thread=False` +
  `row_factory=sqlite3.Row`，所有写操作走 `with self._transaction():` 事务。
- 连接开启 `PRAGMA foreign_keys=ON`（SQLite 默认关闭，必须每次连接显式开启）。
- 幂等迁移：`schema_version` 表记录版本，`migrate()` 逐版本、每版本一个事务，
  重复调用返回 0。
- NaN 纪律：NaN 测量值统一存 NULL，读回时还原为 `float('nan')`，
  绝不用 0 冒充缺失。
"""
from __future__ import annotations

import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

#: 每个版本的迁移 SQL（版本号 -> 语句列表）。新版本追加新键即可。
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS patient (
            id TEXT PRIMARY KEY,
            pseudonym TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS visit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL REFERENCES patient(id),
            visit_date TEXT NOT NULL,
            eye TEXT NOT NULL DEFAULT 'OS',
            eyelid TEXT NOT NULL DEFAULT 'lower',
            image_path TEXT NOT NULL DEFAULT '',
            note TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS measurement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visit_id INTEGER NOT NULL REFERENCES visit(id),
            key TEXT NOT NULL,
            value REAL,
            unit TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS review (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visit_id INTEGER NOT NULL REFERENCES visit(id),
            reviewer_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            comment TEXT NOT NULL DEFAULT '',
            ts TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_visit_patient ON visit(patient_id)",
        "CREATE INDEX IF NOT EXISTS idx_measurement_visit ON measurement(visit_id)",
        "CREATE INDEX IF NOT EXISTS idx_review_visit ON review(visit_id)",
    ),
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClinicStore:
    """随访 sqlite 存储。用法：`with ClinicStore(path) as store: ...`。"""

    def __init__(self, db_path) -> None:
        self.db_path = Path(db_path)
        # 路径父目录自动创建
        parent = self.db_path.parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        # check_same_thread=False 允许跨线程共享连接，但 sqlite3 的事务状态是连接级的：
        # 两个线程并发 `with self._transaction():` 会互相踩踏事务（实测 OperationalError）。
        # 进程内写互斥交给这把锁；多进程并发由调用方保证（单机工具定位）。
        self._write_lock = threading.Lock()
        self.migrate()

    @contextmanager
    def _transaction(self):
        """写事务统一入口：进程内锁 + sqlite 事务（with self._transaction():）。"""
        with self._write_lock, self._conn:
            yield

    # ---- 生命周期 ----

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ClinicStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 迁移 ----

    def migrate(self) -> int:
        """幂等迁移：返回本次实际执行的迁移步数（无操作返回 0）。

        每个 migration version 单独一个事务并立即写入 schema_version，
        保证中断后可从断点继续。
        """
        with self._transaction():
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                )
                """
            )
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = int(row["v"]) if row is not None and row["v"] is not None else 0

        steps = 0
        for version in sorted(_MIGRATIONS):
            if version <= current:
                continue
            with self._transaction():
                for statement in _MIGRATIONS[version]:
                    self._conn.execute(statement)
                self._conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?)", (version,)
                )
            steps += 1
        return steps

    # ---- 患者与就诊 ----

    def upsert_patient(self, patient_id, pseudonym="", note="") -> None:
        patient_id = str(patient_id)
        if not patient_id.strip():
            raise ValueError("patient_id 不能为空")
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO patient(id, pseudonym, created_at, note)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    pseudonym = excluded.pseudonym,
                    note = excluded.note
                """,
                (patient_id, str(pseudonym), _utcnow_iso(), str(note)),
            )

    def add_visit(self, patient_id, visit_date, eye="OS", eyelid="lower",
                  image_path="", note="") -> int:
        with self._transaction():
            cursor = self._conn.execute(
                """
                INSERT INTO visit(patient_id, visit_date, eye, eyelid, image_path, note)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(patient_id),
                    str(visit_date),
                    str(eye),
                    str(eyelid),
                    str(image_path),
                    str(note),
                ),
            )
        return int(cursor.lastrowid)

    # ---- 测量与复核 ----

    def add_measurement(self, visit_id, key, value, unit="", source="") -> int:
        # NaN 纪律：NaN 统一存 NULL（绝不存 0 冒充）
        value = float(value)
        if math.isnan(value):
            stored: float | None = None
        else:
            stored = value
        with self._transaction():
            cursor = self._conn.execute(
                """
                INSERT INTO measurement(visit_id, key, value, unit, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (int(visit_id), str(key), stored, str(unit), str(source), _utcnow_iso()),
            )
        return int(cursor.lastrowid)

    def add_review(self, visit_id, reviewer_id, decision, comment="", ts=None) -> int:
        ts_value = str(ts) if ts is not None else _utcnow_iso()
        with self._transaction():
            cursor = self._conn.execute(
                """
                INSERT INTO review(visit_id, reviewer_id, decision, comment, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (int(visit_id), str(reviewer_id), str(decision), str(comment), ts_value),
            )
        return int(cursor.lastrowid)

    # ---- 查询 ----

    def list_patients(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, pseudonym, created_at, note FROM patient ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def list_visits(self, patient_id) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT id, patient_id, visit_date, eye, eyelid, image_path, note
            FROM visit
            WHERE patient_id = ?
            ORDER BY visit_date, id
            """,
            (str(patient_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def measurements_for_visit(self, visit_id) -> dict[str, float | None]:
        rows = self._conn.execute(
            "SELECT key, value FROM measurement WHERE visit_id = ? ORDER BY id",
            (int(visit_id),),
        ).fetchall()
        result: dict[str, float | None] = {}
        for row in rows:
            value = row["value"]
            # NaN 纪律：NULL 读回 float('nan')
            result[str(row["key"])] = float("nan") if value is None else float(value)
        return result

    def reviews_for_visit(self, visit_id) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT id, visit_id, reviewer_id, decision, comment, ts
            FROM review
            WHERE visit_id = ?
            ORDER BY id
            """,
            (int(visit_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def timeline(self, patient_id) -> list[dict]:
        """每次就诊一行（按 visit_date, id 排序），内嵌 measurements dict 与 reviews 列表。"""
        result: list[dict] = []
        for visit in self.list_visits(patient_id):
            entry = dict(visit)
            entry["measurements"] = self.measurements_for_visit(visit["id"])
            entry["reviews"] = self.reviews_for_visit(visit["id"])
            result.append(entry)
        return result
