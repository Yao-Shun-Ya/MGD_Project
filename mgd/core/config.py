"""唯一配置源。

所有路径都以「代码所在位置」为锚点（mgd/ 目录），不依赖当前工作目录 CWD。
允许通过环境变量覆盖，便于外壳与测试传入不同数据集/权重目录。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from .runtime.errors import ConfigError

# mgd/core/config.py -> mgd/
PACKAGE_DIR = Path(__file__).resolve().parents[1]
#: 仓库/发行包根目录（包含 data/、models_weights/、outputs/ 的那一层）
REPO_ROOT = PACKAGE_DIR.parent

TARGET_HEIGHT = 640
TARGET_WIDTH = 1280
CLAHE_CLIP_LIMIT = 0.02

#: 交付方提供的 canonical 权重文件名（状态由 manifest 决定，见 core/models/manifest.py）
CANONICAL_WEIGHT_NAME = "MGD_UNet_v1.0.0.pth"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


@dataclass(frozen=True)
class Settings:
    """运行期配置。构造后所有 Path 字段均为绝对路径。"""

    home: Path = REPO_ROOT
    data_dir: Path | None = None
    model_dir: Path | None = None
    output_dir: Path | None = None
    preset_file: Path | None = None
    log_file: Path | None = None
    target_height: int = TARGET_HEIGHT
    target_width: int = TARGET_WIDTH
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    auth_token: str | None = None
    device: str | None = None
    num_workers: int = 0
    image_subdir: str = "images"
    gland_subdir: str = "gland_masks"
    eyelid_subdir: str = "eyelid_masks"
    # ---- v2.4.0 诊断中心级增量（全部带默认值，向后兼容） ----
    db_path: Path | None = None                  # 随访数据库；默认 home/"db"/"mgd_clinic.db"
    calibration_dir: Path | None = None          # 共形校准参数；默认 output_dir/"calibration"
    atlas_index_path: Path | None = None         # 形态图谱索引；默认 home/"db"/"atlas_index.json"
    qc_policy: str = "default"                   # 质控策略名（阈值组）；default = 工程默认
    physical_scale_mm_per_px: float | None = None  # 物理标定；None = 未标定（长度类指标保持 px）
    # ---- v2.5.0 接线增量（治理/权限/脱敏） ----
    role: str = "doctor"                         # 本机角色（RBAC）：tech/doctor/chief/admin
    deid_salt: str | None = None                 # 报告脱敏盐；设置后 patient_id 落盘为伪名

    def __post_init__(self) -> None:
        home = Path(self.home).expanduser().resolve()
        object.__setattr__(self, "home", home)
        for name, default in (
            ("data_dir", home / "data"),
            ("model_dir", home / "models_weights"),
            ("output_dir", home / "outputs"),
            ("preset_file", home / "infer_presets.json"),
            ("log_file", home / "logs" / "mgd.log"),
            ("db_path", home / "db" / "mgd_clinic.db"),
            ("atlas_index_path", home / "db" / "atlas_index.json"),
        ):
            value = getattr(self, name)
            path = Path(value) if value is not None else default
            if not path.is_absolute():
                path = home / path
            object.__setattr__(self, name, path.resolve())

        # calibration_dir 默认跟随（已解析的）output_dir，故不能放进上面的固定默认表
        calib_path = Path(self.calibration_dir) if self.calibration_dir is not None else self.output_dir / "calibration"
        if not calib_path.is_absolute():
            calib_path = home / calib_path
        object.__setattr__(self, "calibration_dir", calib_path.resolve())

        if not (0 < float(self.clahe_clip_limit) <= 1.0):
            raise ConfigError("clahe_clip_limit 必须在 (0, 1] 区间")
        if int(self.target_height) <= 0 or int(self.target_width) <= 0:
            raise ConfigError("target_height / target_width 必须为正整数")
        if not str(self.qc_policy).strip():
            raise ConfigError("qc_policy 不能为空字符串")
        if self.physical_scale_mm_per_px is not None and float(self.physical_scale_mm_per_px) <= 0:
            raise ConfigError("physical_scale_mm_per_px 必须为正数（或 None 表示未标定）")
        if str(self.role).strip() not in ("tech", "doctor", "chief", "admin"):
            raise ConfigError(
                f"未知角色 {self.role!r}",
                hint="MGD_ROLE 可选：tech / doctor / chief / admin",
            )

    @property
    def target_size(self) -> tuple[int, int]:
        """(height, width)。"""
        return int(self.target_height), int(self.target_width)

    @property
    def images_dir(self) -> Path:
        return self.data_dir / self.image_subdir

    @property
    def gland_masks_dir(self) -> Path:
        return self.data_dir / self.gland_subdir

    @property
    def eyelid_masks_dir(self) -> Path:
        return self.data_dir / self.eyelid_subdir

    def with_overrides(self, **overrides) -> "Settings":
        return replace(self, **overrides)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """从环境变量构造；overrides 优先级最高。"""
        env = os.environ
        kwargs = dict(overrides)
        if "MGD_HOME" in env:
            kwargs.setdefault("home", env["MGD_HOME"])
        if "MGD_DATA_DIR" in env:
            kwargs.setdefault("data_dir", env["MGD_DATA_DIR"])
        if "MGD_MODEL_DIR" in env:
            kwargs.setdefault("model_dir", env["MGD_MODEL_DIR"])
        if "MGD_OUTPUT_DIR" in env:
            kwargs.setdefault("output_dir", env["MGD_OUTPUT_DIR"])
        if "MGD_HOST" in env:
            kwargs.setdefault("host", env["MGD_HOST"])
        if "MGD_PORT" in env:
            try:
                kwargs.setdefault("port", int(env["MGD_PORT"]))
            except ValueError as exc:
                raise ConfigError("MGD_PORT 必须为整数") from exc
        if "MGD_AUTH_TOKEN" in env:
            kwargs.setdefault("auth_token", env["MGD_AUTH_TOKEN"] or None)
        if "MGD_DEVICE" in env:
            kwargs.setdefault("device", env["MGD_DEVICE"])
        if "MGD_NUM_WORKERS" in env:
            try:
                kwargs.setdefault("num_workers", int(env["MGD_NUM_WORKERS"]))
            except ValueError as exc:
                raise ConfigError("MGD_NUM_WORKERS 必须为整数") from exc
        if "MGD_DB_PATH" in env:
            kwargs.setdefault("db_path", env["MGD_DB_PATH"] or None)
        if "MGD_CALIBRATION_DIR" in env:
            kwargs.setdefault("calibration_dir", env["MGD_CALIBRATION_DIR"] or None)
        if "MGD_ATLAS_INDEX" in env:
            kwargs.setdefault("atlas_index_path", env["MGD_ATLAS_INDEX"] or None)
        if "MGD_ROLE" in env:
            kwargs.setdefault("role", env["MGD_ROLE"])
        if "MGD_DEID_SALT" in env:
            kwargs.setdefault("deid_salt", env["MGD_DEID_SALT"] or None)
        if "MGD_SCALE_MM_PER_PX" in env:
            try:
                kwargs.setdefault(
                    "physical_scale_mm_per_px",
                    float(env["MGD_SCALE_MM_PER_PX"]) if env["MGD_SCALE_MM_PER_PX"].strip() else None,
                )
            except ValueError as exc:
                raise ConfigError("MGD_SCALE_MM_PER_PX 必须为数字") from exc
        return cls(**kwargs)


_cached_settings: Settings | None = None


def get_settings(**overrides) -> Settings:
    """进程级单例配置。测试请使用 reset_settings()。"""
    global _cached_settings
    if overrides:
        return (_cached_settings or Settings.from_env()).with_overrides(**overrides)
    if _cached_settings is None:
        _cached_settings = Settings.from_env()
    return _cached_settings


def set_settings(settings: Settings) -> Settings:
    global _cached_settings
    _cached_settings = settings
    return settings


def reset_settings() -> None:
    global _cached_settings
    _cached_settings = None


def model_best_path(model_name: str, settings: Settings | None = None) -> Path:
    """最佳权重路径。只计算，不创建目录。"""
    settings = settings or get_settings()
    return settings.model_dir / model_name / f"{model_name.lower()}_best.pth"


def checkpoint_path(model_name: str, epoch: int, settings: Settings | None = None) -> Path:
    """完整训练状态检查点路径（含优化器/调度器/AMP/epoch）。"""
    settings = settings or get_settings()
    return settings.model_dir / model_name / f"{model_name.lower()}_epoch_{int(epoch):03d}.mgd.ckpt"


def canonical_weight_candidates(settings: Settings | None = None) -> list[Path]:
    """canonical 权重的搜索顺序（只读引用，不复制、不移动交付文件）。"""
    settings = settings or get_settings()
    return [
        settings.home / CANONICAL_WEIGHT_NAME,
        settings.model_dir / CANONICAL_WEIGHT_NAME,
        settings.model_dir / "UNet" / CANONICAL_WEIGHT_NAME,
        # 仓库内既有交付位置（public/file），保持原位以免破坏网站直链
        PACKAGE_DIR.parent / "public" / "file" / CANONICAL_WEIGHT_NAME,
    ]


def list_trained_models(settings: Settings | None = None) -> list[str]:
    """列出已有 best 权重的模型（顺序与 registry 一致）。"""
    from .models.registry import MODEL_NAMES

    settings = settings or get_settings()
    return [name for name in MODEL_NAMES if model_best_path(name, settings).is_file()]
