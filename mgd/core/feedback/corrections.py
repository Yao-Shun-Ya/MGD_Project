"""医生修正记录、汇总与训练候选导出。

口径约定（与 :mod:`mgd.core.imaging.metrics` 一致，但独立实现，不 import 其私有函数）：
- Dice = 2×|A∩B| / (|A|+|B|)；双空 → NaN（无定义，不用 0 冒充），单边为空 → 0.0；
- 掩膜二值化：``>0`` 视为前景。

内存约定：:attr:`CorrectionRecord.model_mask` / :attr:`CorrectionRecord.edited_mask`
保存调用方传入数组的**引用**（不深拷贝）——调用方随后原地修改数组会同步反映到
记录中；需要快照请调用方自行 ``.copy()``。掩膜数组不参与 JSON 导出（体积考虑，
见 :func:`export_candidates` docstring），掩膜持久化属后续工作。

持久化（v2.5 接线）：:func:`append_record` 把修正行追加写入
``settings.home / "db" / "corrections.jsonl"``（纯 JSONL、不嵌掩膜像素，
只带掩膜内容的 sha256）；:func:`load_records` / :func:`summarize_file` /
:func:`export_from_file` 在持久化文件上提供与内存侧同口径的读取 / 汇总 /
导出。``settings.deid_salt`` 设置时，落盘 case_id 自动假名化（延迟导入
:func:`mgd.core.governance.deid.pseudonymize`）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..config import Settings
from ..runtime.logging import get_logger

logger = get_logger("feedback.corrections")

DIRECTION_ADD_ONLY = "add_only"
DIRECTION_REMOVE_ONLY = "remove_only"
DIRECTION_MIXED = "mixed"
DIRECTION_NONE = "none"


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice 系数：2×|A∩B| / (|A|+|B|)。

    口径与 :mod:`mgd.core.imaging.metrics` 一致：输入先按 ``>0`` 二值化；
    两者均为空时 Dice 无定义，返回 NaN（不用 0 冒充）；单边为空 → 0.0。
    """
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    sum_a, sum_b = int(a.sum()), int(b.sum())
    if sum_a == 0 and sum_b == 0:
        return float("nan")
    inter = int(np.logical_and(a, b).sum())
    return float(2 * inter / (sum_a + sum_b))


def _direction(added_px: int, removed_px: int) -> str:
    """按像素增删给修正分类：add_only / remove_only / mixed / none。"""
    if added_px > 0 and removed_px == 0:
        return DIRECTION_ADD_ONLY
    if removed_px > 0 and added_px == 0:
        return DIRECTION_REMOVE_ONLY
    if added_px > 0 and removed_px > 0:
        return DIRECTION_MIXED
    return DIRECTION_NONE


def _utc_now_iso() -> str:
    """当前 UTC 时间的 timezone-aware ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _mean_ignore_nan(values: list[float]) -> float:
    """np.nanmean 语义：NaN 忽略；全 NaN（含空表）→ NaN。

    手工过滤而非直接调 np.nanmean，避免 numpy 对全 NaN 空切片抛 RuntimeWarning。
    """
    valid = [v for v in values if not math.isnan(v)]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def _jsonable(value: float) -> float | None:
    """NaN → None（JSON ``null``），其余原样返回，保证输出是严格 JSON。"""
    return None if isinstance(value, float) and math.isnan(value) else value


@dataclass
class CorrectionRecord:
    """单条医生修正记录。

    Attributes:
        case_id: 病例标识。
        model_mask: 模型输出掩膜。**保存引用，不深拷贝**：调用方随后原地修改
            该数组会同步改变本记录所见内容；需要快照请自行 ``.copy()``。
        edited_mask: 医生编辑后的掩膜。内存约定同上。
        dice_before: Dice(model_mask, reference_mask)；无 reference → NaN。
        dice_after_vs_edit: Dice(edited_mask, reference_mask)；无 reference 时
            退化为 Dice(model_mask, edited_mask)——衡量模型与编辑的一致度，
            而非对真值的准确度。
        added_px: |edited & ~model|，医生新增的前景像素数。
        removed_px: |model & ~edited|，医生删除的前景像素数。
        ts: 记录时间，ISO 8601、UTC、timezone-aware。
        reviewer_id: 医生/评审人标识。
        note: 自由文本备注。
    """

    case_id: str
    model_mask: np.ndarray
    edited_mask: np.ndarray
    dice_before: float
    dice_after_vs_edit: float
    added_px: int
    removed_px: int
    ts: str
    reviewer_id: str
    note: str = ""

    @property
    def direction(self) -> str:
        """修正方向分类：add_only / remove_only / mixed / none（见 :func:`_direction`）。"""
        return _direction(self.added_px, self.removed_px)


def record_correction(
    case_id: str,
    model_mask: np.ndarray,
    edited_mask: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    reviewer_id: str = "unknown",
    ts: str | None = None,
    note: str = "",
) -> CorrectionRecord:
    """捕获一条医生修正并计算统计量，返回 :class:`CorrectionRecord`。

    - 掩膜按 ``>0`` 二值化后比较；
    - model_mask / edited_mask 形状不一致 → ValueError；
      reference_mask 提供时同样校验形状；
    - ts 缺省取当前 UTC 时间（timezone-aware ISO 8601）；
    - record.model_mask / edited_mask 保存传入数组的引用（不深拷贝，
      见 :class:`CorrectionRecord` docstring）。
    """
    model = np.asarray(model_mask)
    edited = np.asarray(edited_mask)
    if model.shape != edited.shape:
        raise ValueError(f"掩膜形状不一致：model={model.shape}, edited={edited.shape}")
    if reference_mask is not None:
        reference = np.asarray(reference_mask)
        if reference.shape != model.shape:
            raise ValueError(f"掩膜形状不一致：reference={reference.shape}, model={model.shape}")
        dice_before = _dice(model, reference)
        dice_after = _dice(edited, reference)
    else:
        # 无真值：dice_before 无定义 → NaN；dice_after_vs_edit 退化为模型与编辑的一致度
        dice_before = float("nan")
        dice_after = _dice(model, edited)

    model_bin, edited_bin = model > 0, edited > 0
    added_px = int(np.logical_and(edited_bin, ~model_bin).sum())
    removed_px = int(np.logical_and(model_bin, ~edited_bin).sum())

    record = CorrectionRecord(
        case_id=case_id,
        model_mask=model,
        edited_mask=edited,
        dice_before=dice_before,
        dice_after_vs_edit=dice_after,
        added_px=added_px,
        removed_px=removed_px,
        ts=ts if ts is not None else _utc_now_iso(),
        reviewer_id=reviewer_id,
        note=note,
    )
    logger.debug(
        "record_correction: case=%s added=%d removed=%d direction=%s",
        case_id, added_px, removed_px, record.direction,
    )
    return record


def summarize(records: list[CorrectionRecord]) -> dict:
    """汇总修正记录。

    返回::

        {
            "n": 记录数,
            "mean_added_px": added_px 均值,
            "mean_removed_px": removed_px 均值,
            "mean_dice_before": dice_before 均值（NaN 忽略，np.nanmean 语义）,
            "mean_dice_after_vs_edit": 同上,
            "direction_counts": {"add_only": x, "remove_only": y, "mixed": z, "none": w},
            "net_add_cases": added_px > removed_px 的记录数（净增像素）,
        }

    空 records → 不抛异常：n / direction_counts / net_add_cases 全 0，
    均值类字段为 NaN（与"全 NaN → NaN"的 nanmean 口径一致）。
    """
    direction_counts = {
        DIRECTION_ADD_ONLY: 0,
        DIRECTION_REMOVE_ONLY: 0,
        DIRECTION_MIXED: 0,
        DIRECTION_NONE: 0,
    }
    added, removed, before, after = [], [], [], []
    net_add_cases = 0
    for record in records:
        direction_counts[record.direction] += 1
        added.append(record.added_px)
        removed.append(record.removed_px)
        before.append(record.dice_before)
        after.append(record.dice_after_vs_edit)
        if record.added_px > record.removed_px:
            net_add_cases += 1
    return {
        "n": len(records),
        "mean_added_px": _mean_ignore_nan(added),
        "mean_removed_px": _mean_ignore_nan(removed),
        "mean_dice_before": _mean_ignore_nan(before),
        "mean_dice_after_vs_edit": _mean_ignore_nan(after),
        "direction_counts": direction_counts,
        "net_add_cases": net_add_cases,
    }


def export_candidates(
    records: list[CorrectionRecord],
    out_dir: str | os.PathLike,
    max_items: int = 200,
) -> Path:
    """导出训练候选清单（JSONL）到 ``<out_dir>/correction_candidates.jsonl``。

    - 按 (added_px + removed_px) 降序取前 ``max_items`` 条；同分保持输入顺序
      （稳定排序）；
    - 每行字段：case_id / added_px / removed_px / dice_before /
      dice_after_vs_edit / ts / reviewer_id / note；
    - **不内嵌掩膜数组**（体积考虑）：本清单只携带标量统计，供后续按
      case_id 回查原图与掩膜；掩膜持久化属后续工作；
    - NaN 的 dice 序列化为 ``null``，保证每行是严格 JSON（非 Python 工具可消费）；
    - 原子写：先写同目录临时文件，再 ``os.replace`` 落盘；
    - records 为空 → ValueError。
    """
    if not records:
        raise ValueError("export_candidates: records 为空，无候选可导出")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ranked = sorted(records, key=lambda r: r.added_px + r.removed_px, reverse=True)[:max_items]

    final_path = out_path / "correction_candidates.jsonl"
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path), prefix=".correction_candidates.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for record in ranked:
                row = {
                    "case_id": record.case_id,
                    "added_px": int(record.added_px),
                    "removed_px": int(record.removed_px),
                    "dice_before": _jsonable(record.dice_before),
                    "dice_after_vs_edit": _jsonable(record.dice_after_vs_edit),
                    "ts": record.ts,
                    "reviewer_id": record.reviewer_id,
                    "note": record.note,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    logger.debug("export_candidates: %d 条候选 -> %s", len(ranked), final_path)
    return final_path


# ============================================================================
# 持久化（v2.5 接线）：纯 JSONL 落盘，不嵌掩膜像素，只带掩膜内容 sha256。
# ============================================================================

#: 持久化文件名（完整路径见 :func:`corrections_path`）
CORRECTIONS_FILENAME = "corrections.jsonl"

#: 掩膜 sha256 的流式分块大小（8 MB/块，避免一次性复制大数组）
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _mask_sha256(mask: np.ndarray) -> str:
    """掩膜数组的 sha256：对 ``ndarray.tobytes()`` 的像素字节流式喂入。

    只覆盖像素字节（不含 shape/dtype）：同内容 → 同哈希（稳定），内容变 → 哈希变。
    逐行块喂入（每块 ≤ 8 MB），不在内存里复制整个掩膜。
    """
    arr = np.ascontiguousarray(np.asarray(mask))
    digest = hashlib.sha256()
    if arr.ndim == 0:
        digest.update(arr.tobytes())
        return digest.hexdigest()
    row_bytes = int(arr.dtype.itemsize) * (int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1)
    rows_per_chunk = max(1, _HASH_CHUNK_BYTES // max(1, row_bytes))
    for start in range(0, int(arr.shape[0]), rows_per_chunk):
        digest.update(np.ascontiguousarray(arr[start:start + rows_per_chunk]).tobytes())
    return digest.hexdigest()


def record_to_row(
    record: CorrectionRecord,
    *,
    model_mask_sha256: str | None = None,
    edited_mask_sha256: str | None = None,
) -> dict:
    """CorrectionRecord → 持久化行 dict（不内嵌掩膜数组）。

    与 :func:`export_candidates` 的行字段（case_id / added_px / removed_px /
    dice_before / dice_after_vs_edit / ts / reviewer_id / note）一致，另加
    ``direction`` 与 ``model_mask_sha256`` / ``edited_mask_sha256`` 两个掩膜
    哈希字段（未提供掩膜时为 null）；NaN 的 dice 一律转 None（严格 JSON）。
    掩膜哈希只覆盖像素字节（``ndarray.tobytes()``），用于事后核对掩膜内容、
    防止 model/edited 掩膜被调换；按需重建像素属后续工作。
    """
    return {
        "case_id": str(record.case_id),
        "dice_before": _jsonable(record.dice_before),
        "dice_after_vs_edit": _jsonable(record.dice_after_vs_edit),
        "added_px": int(record.added_px),
        "removed_px": int(record.removed_px),
        "ts": str(record.ts),
        "reviewer_id": str(record.reviewer_id),
        "note": str(record.note),
        "direction": record.direction,
        "model_mask_sha256": model_mask_sha256,
        "edited_mask_sha256": edited_mask_sha256,
    }


def corrections_path(settings: Settings) -> Path:
    """修正持久化文件路径：``settings.home / "db" / "corrections.jsonl"``。

    路径直接由 ``settings.home`` 组装（与 db_path / atlas_index_path 同一
    ``db/`` 目录惯例；不改动 config.py）。
    """
    return Path(settings.home) / "db" / CORRECTIONS_FILENAME


def append_record(
    record: CorrectionRecord,
    settings: Settings,
    *,
    model_mask: np.ndarray | None = None,
    edited_mask: np.ndarray | None = None,
) -> dict:
    """把一条修正追加写入 JSONL 持久化文件，返回实际写入的行 dict。

    - ``model_mask`` / ``edited_mask`` 提供时计算 sha256（对
      ``ndarray.tobytes()`` 按 8MB/块流式喂入，见 :func:`_mask_sha256`）写入行内；
      掩膜像素本身**不落盘**；
    - ``settings.deid_salt`` 非 None 时，行内 ``case_id`` 先经
      :func:`mgd.core.governance.deid.pseudonymize` 假名化（延迟导入，依赖
      PIL 故按需加载）：同一 case_id 在同一盐下伪名稳定、可跨时间关联，
      不同盐得到不同伪名，明文 case_id 不落盘；传入的 record 对象不被修改；
      空 case_id + 盐 会因 pseudonymize 拒绝而抛 ValueError（合规上禁止无盐
      或空 ID 假名化）；
    - 原子行写：文件末尾若缺换行（上次进程崩溃留下的半行 JSON），先补一个
      换行再追加，避免新记录与残片拼在同一行（语义对齐 governance.audit 的
      ``append_line``，此处独立实现、不 import governance 写路径）；显式
      flush + fsync；
    - 每行严格 JSON（NaN→null，``allow_nan=False`` 双保险）。
    """
    model_sha = _mask_sha256(model_mask) if model_mask is not None else None
    edited_sha = _mask_sha256(edited_mask) if edited_mask is not None else None
    row = record_to_row(record, model_mask_sha256=model_sha, edited_mask_sha256=edited_sha)
    if settings.deid_salt:
        from ..governance.deid import pseudonymize  # 延迟导入：deid 依赖 PIL，按需加载

        row["case_id"] = pseudonymize(row["case_id"], settings.deid_salt)
    path = corrections_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, allow_nan=False)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as probe:
            probe.seek(-1, os.SEEK_END)
            needs_newline = probe.read(1) != b"\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    logger.debug("append_record: case=%s -> %s", row["case_id"], path)
    return row


def load_records(settings: Settings, *, last: int | None = None) -> list[dict]:
    """读取 JSONL 修正记录（按文件顺序，最新在后）。

    - 文件不存在 → 空表；
    - 损坏行（非法 JSON / 非 JSON 对象）逐行跳过并 ``logger.warning``（带行号，
      即"跳过计数"），绝不中断业务读取；
    - ``last`` 给定且 > 0 → 只返回最后 ``last`` 条（0 / 负数 / None = 全部）。
    """
    path = corrections_path(settings)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("修正记录 第 %d 行损坏（%s），已跳过", lineno, exc)
                continue
            if not isinstance(row, dict):
                logger.warning("修正记录 第 %d 行不是 JSON 对象，已跳过", lineno)
                continue
            rows.append(row)
    if last is not None and int(last) > 0:
        rows = rows[-int(last):]
    return rows


def _row_int(row: dict, key: str) -> int:
    """行字段 → int（缺失 / 非法值按 0 处理，读取端容错）。"""
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _row_dice(row: dict, key: str) -> float:
    """行字段 → float；null（NaN 落盘形态）或非法值 → NaN（与内存口径一致）。"""
    value = row.get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_file(settings: Settings) -> dict:
    """对持久化文件做 :func:`summarize` 同口径汇总（对行 dict 适配）。

    直接复用 :func:`summarize` 的统计语义（direction_counts / net_add_cases /
    mean_*）：行内缺 ``direction`` 时按 added/removed 重新分类；dice 为 null
    的行在均值中被忽略（np.nanmean 口径）。空文件 / 文件不存在 → 与
    ``summarize([])`` 相同的全零 / NaN 结构，不抛异常。
    """
    adapted = [
        SimpleNamespace(
            direction=str(row.get("direction") or _direction(_row_int(row, "added_px"),
                                                             _row_int(row, "removed_px"))),
            added_px=_row_int(row, "added_px"),
            removed_px=_row_int(row, "removed_px"),
            dice_before=_row_dice(row, "dice_before"),
            dice_after_vs_edit=_row_dice(row, "dice_after_vs_edit"),
        )
        for row in load_records(settings)
    ]
    return summarize(adapted)


def export_from_file(settings: Settings, out_dir: str | os.PathLike, max_items: int = 200) -> Path:
    """从持久化 JSONL 直接导出训练候选清单（复用 :func:`export_candidates`）。

    - 行按 (added_px + removed_px) 降序取前 ``max_items`` 条（稳定排序，同分
      保持文件顺序），输出行格式与 :func:`export_candidates` 完全一致（8 字段
      投影，不含 direction / 掩膜哈希）；
    - 文件不存在 / 无可读记录 → ValueError（与 export_candidates 空表行为一致）；
    - 原子写与临时文件清理由 :func:`export_candidates` 承担。
    """
    rows = load_records(settings)
    if not rows:
        raise ValueError("export_from_file: 修正记录为空，无候选可导出")
    ranked = sorted(
        rows, key=lambda row: _row_int(row, "added_px") + _row_int(row, "removed_px"),
        reverse=True,
    )[:max_items]
    adapted = [
        SimpleNamespace(
            case_id=str(row.get("case_id", "")),
            added_px=_row_int(row, "added_px"),
            removed_px=_row_int(row, "removed_px"),
            dice_before=_row_dice(row, "dice_before"),
            dice_after_vs_edit=_row_dice(row, "dice_after_vs_edit"),
            ts=str(row.get("ts", "")),
            reviewer_id=str(row.get("reviewer_id", "")),
            note=str(row.get("note", "")),
        )
        for row in ranked
    ]
    return export_candidates(adapted, out_dir, max_items=max_items)
