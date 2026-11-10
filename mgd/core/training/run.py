"""一次训练 run 的目录与元数据。

目录约定（冻结）：
<output_dir>/runs/<run_id>/
  config.yaml         配置快照（YAML 原值 + 生效值）
  env.json            代码版本 / 依赖版本 / 设备 / git commit
  train.log           训练日志（与 stdout 同内容）
  metrics.jsonl       每轮一行指标
  checkpoints/        断点检查点与 best 权重（含 manifest 草稿）
  reference_report.json  训练结束后的验证口径报告（可选）
  run_result.json     最终结果摘要
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ... import __version__
from ..config import PACKAGE_DIR, Settings
from ..runtime.logging import get_logger

logger = get_logger("training.run")

_REPO_ROOT = PACKAGE_DIR.parent


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    checkpoints_dir: Path

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def env_path(self) -> Path:
        return self.run_dir / "env.json"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "train.log"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def result_path(self) -> Path:
        return self.run_dir / "run_result.json"

    @property
    def reference_path(self) -> Path:
        return self.run_dir / "reference_report.json"


def _slug(value: str, limit: int = 24) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return (cleaned or "run")[:limit]


def git_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if output.returncode == 0:
            return output.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def config_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def create_run(
    settings: Settings,
    arch: str,
    *,
    name: str | None = None,
    output_dir: str | Path | None = None,
    fingerprint: str | None = None,
) -> RunContext:
    root = Path(output_dir).expanduser().resolve() if output_dir else settings.output_dir / "runs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit() or "nogit"
    parts = [stamp, _slug(name) if name else _slug(arch), commit, (fingerprint or "000000")[:6]]
    base_id = "_".join(parts)
    run_id = base_id
    run_dir = root / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_id = f"{base_id}_{suffix}"
        run_dir = root / run_id
    run_dir.mkdir(parents=True)
    return RunContext(run_id=run_id, run_dir=run_dir, checkpoints_dir=run_dir / "checkpoints")


def write_env_snapshot(path: str | Path, settings: Settings, device_text: str | None = None) -> Path:
    packages = ("torch", "numpy", "opencv-python", "monai", "scikit-image", "scipy", "pandas", "PyYAML", "gradio")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            from importlib.metadata import version

            versions[package] = version(package)
        except Exception:
            versions[package] = "未安装"
    payload = {
        "mgd_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(),
        "device": device_text,
        "packages": versions,
        "env": {
            key: os.environ.get(key)
            for key in ("MGD_HOME", "MGD_DATA_DIR", "MGD_MODEL_DIR", "MGD_OUTPUT_DIR", "MGD_DEVICE")
            if os.environ.get(key)
        },
        "paths": {
            "home": str(settings.home),
            "data_dir": str(settings.data_dir),
            "model_dir": str(settings.model_dir),
            "output_dir": str(settings.output_dir),
        },
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def append_metric(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_result(path: str | Path, result: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return target


def latest_checkpoint(checkpoints_dir: str | Path) -> Path | None:
    directory = Path(checkpoints_dir)
    if not directory.is_dir():
        return None
    candidates = sorted(
        (item for item in directory.iterdir() if item.suffix == ".ckpt"),
        key=lambda item: item.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def resolve_resume(value: str | Path | None, checkpoints_dir: Path) -> Path | None:
    """resume_from 支持目录、run 目录或具体 ckpt；目录形式自动取最新检查点。"""
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        candidate = latest_checkpoint(path) or latest_checkpoint(path / "checkpoints")
        if candidate is None:
            raise FileNotFoundError(f"目录中没有检查点：{path}")
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"检查点不存在：{path}")
    return path


def publish_best(best_path: Path, target: Path) -> None:
    """把 run 目录里的 best 权重发布到 model_dir；仅在显式要求时调用。

    原子替换：先写临时文件再 os.replace，避免复制中断留下半截权重
    （以及“新权重 + 旧 manifest”的中间窗口）。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    weight_tmp = target.with_name(target.name + ".publishing.tmp")
    manifest_tmp = target.with_name(target.name + ".manifest.json.publishing.tmp")
    try:
        shutil.copy2(best_path, weight_tmp)
        sidecar = best_path.with_name(best_path.name + ".manifest.json")
        if sidecar.is_file():
            shutil.copy2(sidecar, manifest_tmp)
        os.replace(weight_tmp, target)
        if manifest_tmp.is_file():
            os.replace(manifest_tmp, target.with_name(target.name + ".manifest.json"))
    finally:
        weight_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    logger.info("已发布 best 权重到 %s", target)
