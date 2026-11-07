"""显存自适应档位（6 / 8 / 12 / 16 / 24 / 40+ GB）。

每个档位给出：分块尺寸、batch、梯度累积、AMP、channels_last、深监督、TTA。
`evidence` 明确标注该档是**实测**还是**估算**，并写依据（吞吐 / 显存峰值 / 外推方式）。
启动时探测显存并推荐档位；探测失败时回退 6 GB 安全档。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .engine import TrainConfig


@dataclass(frozen=True)
class VramTier:
    key: str
    label: str
    tile_height: int
    batch_size: int
    amp: bool
    grad_accum_steps: int
    channels_last: bool
    deep_supervision: bool
    tta: str
    status: str                      # 实测 / 估算
    basis: str
    peak_gb: float | None = None
    samples_per_second: float | None = None

    @property
    def is_measured(self) -> bool:
        return self.status == "实测"


TIERS: dict[str, VramTier] = {
    "6": VramTier(
        "6", "6 GB（安全档）", 256, 1, True, 4, True, False, "关闭",
        "实测", "MGD-Former 256×512 / batch 1 / AMP / 累积 4，本机 RTX 4060 Laptop 1 epoch 冒烟："
                "峰值 0.89 GB、3.54 样本/秒（700 训练样本，2026-11-07）",
        peak_gb=0.89,
        samples_per_second=3.54,
    ),
    "8": VramTier(
        "8", "8 GB（本机 RTX 4060）", 384, 1, True, 4, True, True, "关闭",
        "实测", "MGD-Former 384×768 / batch 1 / AMP / 累积 4，本机 RTX 4060 Laptop 1 epoch 冒烟："
                "峰值 1.23 GB、3.10 样本/秒（700 训练样本、单轮 225.6 s，2026-11-07）",
        peak_gb=1.227,
        samples_per_second=3.103,
    ),
    "12": VramTier(
        "12", "12 GB", 384, 2, True, 2, True, True, "关闭",
        "估算", "8 GB 实测档的 batch 翻倍外推；参数显存约 +1.2 GB",
        peak_gb=None,
    ),
    "16": VramTier(
        "16", "16 GB", 512, 2, True, 1, True, True, "关闭",
        "估算", "分块 512×1024 + batch 2 外推（约 9–11 GB）；论文机常见档位",
        peak_gb=None,
    ),
    "24": VramTier(
        "24", "24 GB", 640, 2, True, 1, True, True, "8倍增强 (多尺度+翻转)",
        "估算", "整图 640×1280 + batch 2 外推（约 14–18 GB）；TTA 可在推理时开启",
        peak_gb=None,
    ),
    "40": VramTier(
        "40", "40 GB+ (A100 / A800)", 640, 4, True, 1, True, True, "8倍增强 (多尺度+翻转)",
        "估算", "整图 + batch 4 外推（约 26–32 GB）；未在实机验证",
        peak_gb=None,
    ),
}

#: 探测失败时使用的安全档（宁可慢，不可 OOM）
SAFE_TIER = "6"


def detect_vram_gb() -> float | None:
    """返回当前 CUDA 设备显存（GB）。CPU 或未安装 torch 时返回 None。"""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return round(float(props.total_memory) / (1024 ** 3), 2)
    except Exception:
        return None


def detect_device_text() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = detect_vram_gb() or 0.0
            return f"CUDA · {name} · {total:.1f} GB"
        return "CPU"
    except Exception:
        return "CPU（torch 不可用）"


def recommend_tier(vram_gb: float | None = None) -> tuple[str, str]:
    """返回 (档位 key, 推荐理由)。

    规则：取「标称容量 ≤ 探测显存」的最大档位；探测失败 → 6 GB 安全档。
    桌面驱动只报告标称容量（8 GB 卡显示 8.0），运行期还需给系统留余量，
    因此同容量档位按实测可用，若首轮 OOM 由用户显式降一档。
    """
    if vram_gb is None:
        vram_gb = detect_vram_gb()
    if vram_gb is None:
        return SAFE_TIER, "未检测到 CUDA 显存（CPU 或探测失败），使用 6 GB 安全档"
    order = ["6", "8", "12", "16", "24", "40"]
    fitting = [key for key in order if float(key) <= float(vram_gb)]
    if not fitting:
        return "6", f"显存 {vram_gb:.1f} GB 低于 6 GB，使用最小安全档"
    chosen = fitting[-1]
    fallback = fitting[-2] if len(fitting) > 1 else "6"
    return chosen, (
        f"探测到 {vram_gb:.1f} GB，推荐 {chosen} GB 档"
        f"（按标称容量；首轮若显存不足请用 --vram {fallback} 降档）"
    )


def get_tier(key: str | None, vram_gb: float | None = None) -> tuple[VramTier, str]:
    """解析档位；auto → 探测推荐。返回 (档位, 说明)。"""
    text = str(key or "auto").strip().lower()
    if text in ("", "auto", "none"):
        chosen, reason = recommend_tier(vram_gb)
        return TIERS[chosen], reason
    normalized = text.replace("gb", "").replace("+", "").strip()
    if normalized == "40" or normalized.startswith("40"):
        return TIERS["40"], "显式指定 40 GB+ 档"
    if normalized not in TIERS:
        raise ValueError(f"未知显存档位 {key!r}，可选：auto/{'/'.join(TIERS)}")
    return TIERS[normalized], f"显式指定 {normalized} GB 档"


def apply_tier(config: TrainConfig, tier_key: str | None, vram_gb: float | None = None) -> tuple[TrainConfig, dict[str, Any]]:
    """把档位作用到配置；显式设置过的 tile/batch 等仍会以档位为准（档位就是安全边界）。"""
    tier, reason = get_tier(tier_key, vram_gb)
    merged = replace(
        config,
        vram_tier=tier.key,
        tile_height=int(tier.tile_height),
        tile_mode=True,
        batch_size=int(tier.batch_size),
        amp=bool(tier.amp),
        grad_accum_steps=int(tier.grad_accum_steps),
        channels_last=bool(tier.channels_last),
        deep_supervision=bool(tier.deep_supervision),
        eval_tta=str(tier.tta),
    )
    info = {
        "tier": tier.key,
        "label": tier.label,
        "status": tier.status,
        "basis": tier.basis,
        "reason": reason,
        "tile": f"{tier.tile_height}×{tier.tile_height * 2}",
        "batch_size": tier.batch_size,
        "grad_accum_steps": tier.grad_accum_steps,
        "amp": tier.amp,
        "channels_last": tier.channels_last,
        "deep_supervision": tier.deep_supervision,
        "tta": tier.tta,
        "peak_gb": tier.peak_gb,
        "samples_per_second": tier.samples_per_second,
    }
    return merged, info


def tier_table() -> list[dict[str, Any]]:
    """给 UI / 文档用的档位表。"""
    return [
        {
            "档位": tier.label,
            "分块": f"{tier.tile_height}×{tier.tile_height * 2}",
            "batch": tier.batch_size,
            "梯度累积": tier.grad_accum_steps,
            "AMP": "是" if tier.amp else "否",
            "channels_last": "是" if tier.channels_last else "否",
            "深监督": "是" if tier.deep_supervision else "否",
            "TTA": tier.tta,
            "状态": tier.status,
            "依据": tier.basis,
            "显存峰值(GB)": tier.peak_gb if tier.peak_gb is not None else "未实测",
            "吞吐(样本/秒)": tier.samples_per_second if tier.samples_per_second is not None else "未实测",
        }
        for tier in TIERS.values()
    ]


#: 命令行预设（中层入口）。结构与 YAML 完全一致，三层共用同一 schema。
TRAIN_PROFILES: dict[str, dict[str, Any]] = {
    "academic": {
        "label": "学术档",
        "description": "论文口径：5 折交叉验证、余弦调度、8 倍 TTA 评估、自动生成 reference_report",
        "overrides": {
            "data": {"split_mode": "kfold", "n_folds": 5, "augmentation": True},
            "train": {"epochs": 60, "lr": 1e-4, "scheduler": "Cosine Annealing"},
            "eval": {"tta": "8倍增强 (多尺度+翻转)"},
            "artifacts": {"export_reference_report": True, "publish_best": False},
        },
    },
    "quick": {
        "label": "快速试跑档",
        "description": "1–2 epoch 冒烟，只验证可训练性与吞吐/显存，不产出精度结论",
        "overrides": {
            "data": {"split_mode": "holdout"},
            "train": {"epochs": 2, "lr": 1e-4, "scheduler": "ReduceLROnPlateau (默认)"},
            "eval": {"tta": "关闭"},
            "artifacts": {"export_reference_report": False, "publish_best": False},
        },
    },
    "lowvram": {
        "label": "低显存档",
        "description": "6 GB 档：256×512 分块、关闭深监督、梯度累积 4",
        "overrides": {
            "run": {"vram_tier": "6"},
            "data": {"tile_mode": True},
            "train": {"epochs": 20, "scheduler": "Cosine Annealing"},
            "artifacts": {"export_reference_report": False, "publish_best": False},
        },
    },
}


def profile_names() -> list[str]:
    return list(TRAIN_PROFILES)


def profile_overrides(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    key = str(name).strip().lower()
    if key not in TRAIN_PROFILES:
        raise ValueError(f"未知预设 {name!r}，可选：{', '.join(TRAIN_PROFILES)}")
    return dict(TRAIN_PROFILES[key]["overrides"])

