"""权重 manifest：sha256 + 架构 + 输入口径，用于把「权重文件」变成可追溯资产。

约定：sidecar 与权重同目录，文件名 `<weights>.manifest.json`。
状态机：
- missing：文件不存在
- pending：文件存在但缺少 manifest / sha256 不符 / 架构或输入口径不符 / verified=false
- verified：全部校验通过

canonical 权重 MGD_UNet_v1.0.0.pth 默认没有 manifest，因此状态是 pending；
只有显式跑过口径验证并写入 verified=true 的 manifest 后才允许加载。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..config import Settings, get_settings
from ..runtime.errors import ManifestError

MANIFEST_FORMAT_VERSION = 1
NORMALIZATION_TAG = "scale01+clahe"

ManifestStatusName = Literal["verified", "pending", "missing"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(weights_path: str | Path) -> Path:
    weights = Path(weights_path)
    return weights.with_name(weights.name + ".manifest.json")


@dataclass
class ModelManifest:
    name: str
    arch: str
    model_version: str
    sha256: str
    input_size: list[int]
    normalization: str = NORMALIZATION_TAG
    clahe_clip_limit: float = 0.02
    verified: bool = False
    verified_at: str | None = None
    source: str = "unknown"
    created_at: str = field(default_factory=utc_now)
    format_version: int = MANIFEST_FORMAT_VERSION
    trained_on: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelManifest":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = {k: v for k, v in payload.items() if k not in known}
        data = {k: v for k, v in payload.items() if k in known}
        if unknown:
            data.setdefault("extra", {}).update(unknown)
        data.setdefault("extra", {})
        return cls(**data)


@dataclass
class ManifestStatus:
    path: Path
    status: ManifestStatusName
    reason: str
    manifest: ModelManifest | None = None

    @property
    def verified(self) -> bool:
        return self.status == "verified"


def build_manifest(
    weights_path: str | Path,
    arch: str,
    model_version: str,
    *,
    verified: bool,
    source: str,
    input_size: tuple[int, int] | list[int] | None = None,
    clahe_clip_limit: float = 0.02,
    trained_on: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> ModelManifest:
    settings = settings or get_settings()
    weights = Path(weights_path).resolve()
    if not weights.is_file():
        raise ManifestError(f"权重文件不存在：{weights}")
    size = list(input_size) if input_size is not None else list(settings.target_size)
    return ModelManifest(
        name=weights.name,
        arch=arch,
        model_version=model_version,
        sha256=sha256_file(weights),
        input_size=[int(size[0]), int(size[1])],
        clahe_clip_limit=float(clahe_clip_limit),
        verified=bool(verified),
        verified_at=utc_now() if verified else None,
        source=source,
        trained_on=trained_on,
        extra=extra or {},
    )


def write_manifest(manifest: ModelManifest, weights_path: str | Path) -> Path:
    """原子写入 sidecar，避免半截 JSON。"""
    target = manifest_path_for(weights_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def read_manifest(weights_path: str | Path) -> ModelManifest | None:
    sidecar = manifest_path_for(weights_path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest 无法解析：{sidecar}", hint=str(exc)) from exc
    return ModelManifest.from_dict(payload)


def validate_weight(
    weights_path: str | Path,
    expected_arch: str | None = None,
    *,
    expected_input_size: tuple[int, int] | list[int] | None = None,
    expected_sha256: str | None = None,
    settings: Settings | None = None,
) -> ManifestStatus:
    """校验权重；任何「无法证明」的情况都返回 pending，而不是放行。"""
    settings = settings or get_settings()
    weights = Path(weights_path)
    if not weights.is_file():
        return ManifestStatus(weights, "missing", "权重文件不存在")

    try:
        manifest = read_manifest(weights)
    except ManifestError as exc:
        return ManifestStatus(weights, "pending", f"manifest 损坏：{exc}")

    if manifest is None:
        return ManifestStatus(
            weights,
            "pending",
            "缺少 manifest，无法确认架构 / 输入尺寸 / 归一化口径 / sha256",
        )

    reasons: list[str] = []
    if not manifest.verified:
        reasons.append("manifest 标记为未验证（verified=false）")

    if manifest.sha256 != (expected_sha256 or sha256_file(weights)):
        reasons.append("sha256 与文件内容不一致")

    if expected_arch is not None and manifest.arch != expected_arch:
        reasons.append(f"架构不符：manifest={manifest.arch}, 期望={expected_arch}")

    size = tuple(expected_input_size) if expected_input_size is not None else settings.target_size
    if tuple(manifest.input_size) != tuple(int(v) for v in size):
        reasons.append(f"输入尺寸不符：manifest={manifest.input_size}, 期望={list(size)}")

    if manifest.normalization != NORMALIZATION_TAG:
        reasons.append(f"归一化口径不符：manifest={manifest.normalization}, 期望={NORMALIZATION_TAG}")

    if reasons:
        return ManifestStatus(weights, "pending", "；".join(reasons), manifest)
    return ManifestStatus(weights, "verified", "校验通过", manifest)


def assert_loadable(status: ManifestStatus, *, allow_unverified: bool = False) -> None:
    """加载策略：pending 默认拒绝；显式放开才能加载，且必须在日志中留痕。"""
    if status.status == "missing":
        raise ManifestError(f"权重文件不存在：{status.path}")
    if status.status == "pending" and not allow_unverified:
        raise ManifestError(
            f"权重 {status.path.name} 处于 pending 状态：{status.reason}",
            hint=(
                "先运行 `python -m mgd.core.models.verify <weights>` 完成 640x1280+CLAHE 口径验证，"
                "或在明确知情的情况下使用 allow_unverified=True"
            ),
        )
