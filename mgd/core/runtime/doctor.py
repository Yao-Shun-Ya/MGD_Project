"""换机器第一件事：`mgd doctor`。

检查 Python、torch/设备、关键依赖、数据目录、权重 manifest、磁盘与写权限；
输出人类可读报告，并在关键项失败时返回非零退出码（可直接进 CI）。
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

from ..config import Settings, get_settings, list_trained_models
from ..models.manifest import manifest_path_for, sha256_file, validate_weight
from ..runtime.logging import get_logger

logger = get_logger("runtime.doctor")

REQUIRED_PACKAGES = {
    "numpy": "1.26.4",
    "opencv-python": ">=4.8,<5",
    "Pillow": ">=10.4,<13",
    "scikit-image": "0.25.2",
    "monai": ">=1.5.2,<1.6",
    "pandas": "2.3.3",
    "matplotlib": "3.10.8",
    "scipy": "1.15.3",
    "PyYAML": ">=6,<7",
}
OPTIONAL_PACKAGES = {"gradio": "6.16.0", "weasyprint": ">=62,<63"}


@dataclass
class Check:
    name: str
    status: str            # ok / warn / fail
    detail: str
    hint: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == "fail"


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def ok(self) -> bool:
        return not any(check.blocking for check in self.checks)

    def as_text(self, verbose: bool = True) -> str:
        icons = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
        lines = ["腺而易见 · 环境自检（mgd doctor）", "=" * 46]
        for check in self.checks:
            lines.append(f"{icons.get(check.status, '[????]')} {check.name}：{check.detail}")
            if verbose and check.hint:
                lines.append(f"        → {check.hint}")
        lines.append("=" * 46)
        failed = [check.name for check in self.checks if check.blocking]
        warned = [check.name for check in self.checks if check.status == "warn"]
        lines.append(f"结论：{'通过' if self.ok else '不可用'}｜失败 {len(failed)} 项｜警告 {len(warned)} 项")
        if failed:
            lines.append("必须先处理：" + "、".join(failed))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": check.name, "status": check.status, "detail": check.detail, "hint": check.hint}
                for check in self.checks
            ],
        }


def _check_python() -> Check:
    version_text = sys.version.split()[0]
    if sys.version_info < (3, 10):
        return Check("Python", "fail", f"{version_text}（需要 ≥ 3.10）", "安装 Python 3.10/3.11 后重试")
    return Check("Python", "ok", f"{version_text} · {platform.platform()}")


def _check_torch() -> Check:
    try:
        import torch
    except Exception as exc:
        return Check("PyTorch", "fail", f"未安装或无法导入：{exc}", "按 RUNBOOK 安装 torch 2.5.1（CUDA 或 CPU）")
    detail = f"torch {torch.__version__}"
    hints: list[str] = []
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            detail += f" · CUDA {torch.version.cuda} · {props.name} ({props.total_memory / 1024**3:.1f} GB)"
        else:
            detail += " · CPU（可跑，但训练会非常慢）"
            hints.append("CPU 模式只建议做推理/冒烟；论文训练请用 CUDA 机器")
    except Exception as exc:
        return Check("PyTorch", "warn", f"{detail} · 设备探测失败：{exc}")
    return Check("PyTorch", "ok", detail, "；".join(hints))


def _check_packages() -> Check:
    missing: list[str] = []
    versions: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            versions.append(f"{package}=={version(package)}")
        except Exception:
            missing.append(package)
    optional_missing = []
    for package in OPTIONAL_PACKAGES:
        try:
            version(package)
        except Exception:
            optional_missing.append(package)
    detail = " · ".join(versions[:4]) + (" …" if len(versions) > 4 else "")
    if missing:
        return Check("依赖", "fail", f"缺少：{', '.join(missing)}", "pip install -e \".[cuda]\" 或按 RUNBOOK 装依赖")
    return Check(
        "依赖", "ok", detail,
        ("可选依赖未安装：" + ", ".join(optional_missing)) if optional_missing else "",
    )


def _check_data(settings: Settings) -> Check:
    data = settings.data_dir
    if not data.is_dir():
        return Check("数据目录", "fail", f"不存在：{data}", "放置 images / gland_masks / eyelid_masks 或设置 MGD_DATA_DIR")
    images = list(settings.images_dir.glob("*")) if settings.images_dir.is_dir() else []
    gland = list(settings.gland_masks_dir.glob("*")) if settings.gland_masks_dir.is_dir() else []
    eyelid = list(settings.eyelid_masks_dir.glob("*")) if settings.eyelid_masks_dir.is_dir() else []
    if not images:
        return Check("数据目录", "fail", f"影像目录为空：{settings.images_dir}", "先放置原始影像")
    counts = f"影像 {len(images)} / 腺体掩膜 {len(gland)} / 眼睑掩膜 {len(eyelid)}"
    if len(gland) < len(images):
        return Check("数据目录", "warn", f"{counts}（腺体掩膜少于影像）", "运行 mgd-data check 查看缺失清单")
    return Check("数据目录", "ok", f"{data} · {counts}")


def _check_weights(settings: Settings) -> Check:
    models = list_trained_models(settings)
    candidates = sorted(settings.model_dir.glob("*.pth")) + sorted(settings.home.glob("*.pth"))
    if not candidates and not models:
        return Check("权重", "warn", "未发现任何权重文件（可先训练或放置 canonical 权重）", "")
    verified = 0
    pending = 0
    details: list[str] = []
    for path in candidates[:6]:
        status = validate_weight(path, settings=settings)
        if status.verified:
            verified += 1
        else:
            pending += 1
        details.append(f"{path.name}: {status.status}")
    return Check(
        "权重",
        "ok" if verified else "warn",
        f"已注册模型 {models or '无'} · verified {verified} / pending {pending}"
        + (" ｜ " + "；".join(details) if details else ""),
        "pending 权重只能以 allow_unverified 方式加载；转正需 mgd-verify --evaluate" if pending else "",
    )


def _check_disk(settings: Settings) -> Check:
    targets = [settings.data_dir, settings.output_dir, settings.model_dir]
    details = []
    fails = []
    for path in targets:
        try:
            path.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(path).free / 1024**3
            details.append(f"{path.name or path}: {free:.1f} GB 可用")
            if free < 2.0:
                fails.append(f"{path} 可用空间仅 {free:.1f} GB")
        except OSError as exc:
            fails.append(f"{path}：{exc}")
    if fails:
        return Check("磁盘", "fail", "；".join(fails), "清理磁盘或修改 MGD_OUTPUT_DIR")
    return Check("磁盘", "ok", " · ".join(details))


def _check_write(settings: Settings) -> Check:
    problems: list[str] = []
    for path in (settings.output_dir, settings.data_dir, settings.model_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".mgd_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            problems.append(f"{path}：{exc}")
    if problems:
        return Check("写权限", "fail", "；".join(problems), "检查目录权限或以有写权限的账号运行")
    return Check("写权限", "ok", "输出 / 数据 / 权重目录均可写")


def first_run_hint(settings: Settings) -> str:
    steps = [
        f"1) 放置数据：{settings.data_dir}/images|gland_masks|eyelid_masks",
        f"2) 划分：python -m mgd.adapters.cli.data_cli split",
        f"3) 冒烟：mgd-train --profile quick --vram auto",
        f"4) 评估：mgd-eval --scope test",
    ]
    return "\n".join(steps)


def run_doctor(settings: Settings | None = None, verbose: bool = True) -> DoctorReport:
    settings = settings or get_settings()
    report = DoctorReport()
    report.add(_check_python())
    report.add(_check_torch())
    report.add(_check_packages())
    report.add(_check_data(settings))
    report.add(_check_weights(settings))
    report.add(_check_disk(settings))
    report.add(_check_write(settings))
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="腺而易见环境自检")
    parser.add_argument("--json", action="store_true", help="输出 JSON（CI 用）")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    report = run_doctor(settings)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.as_text())
        if not report.ok:
            print("\n首次运行建议：")
            print(first_run_hint(settings))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
