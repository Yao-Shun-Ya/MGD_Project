"""数据集版本化、完整性校验与写前备份。

安全边界：
- 本模块**只读**用户影像/掩膜；唯一的写操作是元数据（dataset_version.json）与备份副本；
- 任何写之前先跑完整性校验；校验失败只报告，不改动原文件；
- 备份目录带时间戳，已存在则追加序号，绝不覆盖。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..governance.hooks import audit_event
from ..runtime.logging import get_logger

logger = get_logger("data.versioning")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALLOWED_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def naming_anomalies(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        stem = path.stem
        if any(ch not in ALLOWED_NAME_CHARS for ch in stem):
            issues.append(f"{path.name}：含非 ASCII / 特殊字符")
        elif " " in stem:
            issues.append(f"{path.name}：含空格")
        elif stem != stem.strip():
            issues.append(f"{path.name}：首尾空白")
    return issues


def integrity_report(settings: Settings) -> dict[str, Any]:
    """一眼看出缺图、缺掩膜、重复、命名异常。"""
    from collections import Counter

    images = _files(settings.images_dir)
    gland = _files(settings.gland_masks_dir)
    eyelid = _files(settings.eyelid_masks_dir)
    image_stems = {path.stem for path in images}
    gland_stems = {path.stem for path in gland}
    eyelid_stems = {path.stem for path in eyelid}

    missing_gland = sorted(image_stems - gland_stems)
    missing_eyelid = sorted(image_stems - eyelid_stems)
    orphan_gland = sorted(gland_stems - image_stems)
    orphan_eyelid = sorted(eyelid_stems - image_stems)
    duplicates = sorted(
        [stem for stem, count in Counter(path.stem for path in images).items() if count > 1]
    )
    anomalies = (
        naming_anomalies(images) + naming_anomalies(gland) + naming_anomalies(eyelid)
    )

    split_path = settings.data_dir / "dataset_splits.json"
    split_info: dict[str, Any] = {"exists": split_path.is_file()}
    if split_path.is_file():
        try:
            payload = json.loads(split_path.read_text(encoding="utf-8"))
            # format_version=1 把各划分存在 entries.<mode> 下；兼容旧版顶层字段
            entries = payload.get("entries") if isinstance(payload.get("entries"), dict) else {}
            entry = entries.get("holdout") or next(
                (item for item in entries.values() if isinstance(item, dict)), {}
            )
            source = entry if (entry and "train" in entry) else payload
            split_info.update({
                "signature": payload.get("signature") or payload.get("split_signature"),
                "mode": (entry.get("spec") or {}).get("mode") if isinstance(entry, dict) else None,
                "counts": {key: len(source.get(key) or []) for key in ("train", "val", "test")},
            })
        except (OSError, json.JSONDecodeError) as exc:
            split_info["error"] = str(exc)

    covered = image_stems & gland_stems
    report = {
        "checked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "directories": {
            "images": str(settings.images_dir),
            "gland_masks": str(settings.gland_masks_dir),
            "eyelid_masks": str(settings.eyelid_masks_dir),
        },
        "counts": {"images": len(images), "gland_masks": len(gland), "eyelid_masks": len(eyelid)},
        "missing_gland_masks": missing_gland[:50],
        "missing_eyelid_masks": missing_eyelid[:50],
        "orphan_gland_masks": orphan_gland[:50],
        "orphan_eyelid_masks": orphan_eyelid[:50],
        "duplicate_stems": duplicates[:50],
        "naming_anomalies": anomalies[:50],
        "usable_pairs": len(covered),
        "split": split_info,
        "truncated": {
            "missing_gland_masks": len(missing_gland) > 50,
            "missing_eyelid_masks": len(missing_eyelid) > 50,
            "orphans": (len(orphan_gland) + len(orphan_eyelid)) > 50,
            "naming_anomalies": len(anomalies) > 50,
        },
    }
    report["ok"] = not any(
        report[key] for key in (
            "missing_gland_masks", "missing_eyelid_masks", "orphan_gland_masks",
            "orphan_eyelid_masks", "duplicate_stems",
        )
    )
    return report


def integrity_rows(report: dict[str, Any]) -> list[list[Any]]:
    counts = report["counts"]
    return [
        ["影像", counts["images"], "—"],
        ["腺体掩膜", counts["gland_masks"], f"缺 {len(report['missing_gland_masks'])}" if report["missing_gland_masks"] else "完整"],
        ["眼睑掩膜", counts["eyelid_masks"], f"缺 {len(report['missing_eyelid_masks'])}" if report["missing_eyelid_masks"] else "完整"],
        ["可用配对", report["usable_pairs"], "OK" if report["ok"] else "需处理"],
        ["孤儿腺体掩膜", len(report["orphan_gland_masks"]), "—"],
        ["孤儿眼睑掩膜", len(report["orphan_eyelid_masks"]), "—"],
        ["重名样本", len(report["duplicate_stems"]), "—"],
        ["命名异常", len(report["naming_anomalies"]), "—"],
    ]


def hash_manifest(settings: Settings, mode: str = "fast") -> dict[str, Any]:
    """快速模式用 size+mtime；完整模式逐文件 sha256（1000 例约数分钟）。"""
    entries: list[dict[str, Any]] = []
    for label, directory in (
        ("images", settings.images_dir),
        ("gland_masks", settings.gland_masks_dir),
        ("eyelid_masks", settings.eyelid_masks_dir),
    ):
        for path in _files(directory):
            record = {"kind": label, "stem": path.stem, "name": path.name, "size": path.stat().st_size}
            if mode == "full":
                record["sha256"] = sha256_file(path)
            else:
                record["mtime_ns"] = path.stat().st_mtime_ns
            entries.append(record)
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {"mode": mode, "count": len(entries), "digest": digest, "entries": entries}


def record_version(
    settings: Settings,
    *,
    source: str = "本地采集",
    device: str = "未记录",
    hash_mode: str = "fast",
) -> dict[str, Any]:
    """写数据集版本记录；写之前必须先通过完整性校验。"""
    report = integrity_report(settings)
    if not report["ok"]:
        raise ValueError(
            "数据集完整性校验未通过，拒绝写入版本记录："
            f"缺腺体掩膜 {len(report['missing_gland_masks'])}、缺眼睑掩膜 "
            f"{len(report['missing_eyelid_masks'])}、孤儿掩膜 "
            f"{len(report['orphan_gland_masks']) + len(report['orphan_eyelid_masks'])}"
        )
    manifest = hash_manifest(settings, mode=hash_mode)
    payload = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": str(source),
        "capture_device": str(device),
        "counts": report["counts"],
        "usable_pairs": report["usable_pairs"],
        "split_signature": report["split"].get("signature"),
        "hash": {"mode": manifest["mode"], "digest": manifest["digest"], "count": manifest["count"]},
        "integrity": {
            "ok": report["ok"],
            "missing_gland_masks": report["missing_gland_masks"],
            "missing_eyelid_masks": report["missing_eyelid_masks"],
            "duplicate_stems": report["duplicate_stems"],
            "naming_anomalies": report["naming_anomalies"],
        },
    }
    target = settings.data_dir / "dataset_version.json"
    # 原子写（与 splits.py 的持久化同一标准），中断不留半截版本记录
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return payload


def backup_dataset(settings: Settings, target_dir: str | Path | None = None) -> Path:
    """备份影像与掩膜到带时间戳目录；只复制，不移动、不覆盖。

    备份成功后写 ``dataset_backup`` 审计事件（非阻塞；payload 只含目标目录与
    文件/配对计数，不含任何文件名之外的内容信息）。
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if target_dir is None:
        target = settings.home / "backups" / f"dataset_backup_{stamp}"
    else:
        target = Path(target_dir)
        if target.exists():
            target = target.with_name(f"{target.name}_{stamp}")
    if target.exists():
        suffix = 1
        while target.with_name(f"{target.name}_{suffix}").exists():
            suffix += 1
        target = target.with_name(f"{target.name}_{suffix}")
    target.mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, Any]] = []
    for label, directory in (
        ("images", settings.images_dir),
        ("gland_masks", settings.gland_masks_dir),
        ("eyelid_masks", settings.eyelid_masks_dir),
    ):
        destination = target / label
        destination.mkdir(parents=True, exist_ok=True)
        for path in _files(directory):
            shutil.copy2(path, destination / path.name)
            manifest.append({
                "kind": label, "name": path.name, "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    meta = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_home": str(settings.home),
        "files": manifest,
        "counts": {"files": len(manifest)},
    }
    (target / "backup_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 治理审计（非阻塞）：备份属于数据治理动作，成功后必须可追溯。
    # 「配对数」按 images∩gland_masks 的 stem 交集计（与 pairing 口径一致）。
    image_stems = {path.stem for path in _files(settings.images_dir)}
    gland_stems = {path.stem for path in _files(settings.gland_masks_dir)}
    audit_event(
        settings,
        "dataset_backup",
        actor="system",
        payload={
            "target": str(target),
            "pairs": len(image_stems & gland_stems),
            "files": len(manifest),
        },
    )
    return target
