"""论文级图表：雷达图、小提琴图、Bland–Altman 图（matplotlib Agg，无 GUI 依赖）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RADAR_METRICS = ("Dice", "IoU", "clDice", "1-NHD95", "Meiboscore Acc")
VIOLIN_METRIC_LABELS = {"dice": "Dice", "iou": "IoU", "cldice": "clDice"}


def radar_chart(scores: dict[str, dict[str, float]], path: str | Path, title: str = "Model capability radar") -> Path:
    labels = list(RADAR_METRICS)
    keys = ["dice", "iou", "cldice", "nhd95", "meiboscore_accuracy"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 6))
    axis = fig.add_subplot(111, polar=True)
    for model, values in scores.items():
        raw = [float(values.get(key, float("nan"))) for key in keys]
        raw = [0.0 if not np.isfinite(v) else max(0.0, min(1.0, v)) for v in raw]
        raw += raw[:1]
        axis.plot(angles, raw, linewidth=2, label=model)
        axis.fill(angles, raw, alpha=0.12)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels, fontsize=10)
    axis.set_ylim(0, 1)
    axis.set_title(title, pad=20)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=9)
    fig.tight_layout()
    target = Path(path)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target


def violin_plot(values: dict[str, list[float]], path: str | Path, metric: str = "dice", title: str | None = None) -> Path:
    data = []
    labels = []
    for model, items in values.items():
        clean = [v for v in items if np.isfinite(v)]
        if clean:
            data.append(clean)
            labels.append(model)
    fig, axis = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4.5))
    if data:
        parts = axis.violinplot(data, showmeans=True, showmedians=True)
        for body in parts["bodies"]:
            body.set_alpha(0.7)
        axis.set_xticks(range(1, len(labels) + 1))
        axis.set_xticklabels(labels, rotation=15, fontsize=9)
    axis.set_ylabel(VIOLIN_METRIC_LABELS.get(metric, metric))
    axis.set_title(title or f"{VIOLIN_METRIC_LABELS.get(metric, metric)} distribution")
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    target = Path(path)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target


def bland_altman(predicted: list[float], reference: list[float], path: str | Path, title: str = "R_loss Bland–Altman") -> Path:
    pred = np.asarray([v for v in predicted if np.isfinite(v)], dtype=np.float32)
    ref = np.asarray([v for v in reference if np.isfinite(v)], dtype=np.float32)
    n = min(len(pred), len(ref))
    pred, ref = pred[:n], ref[:n]
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    if n >= 2:
        mean = (pred + ref) / 2.0
        diff = pred - ref
        bias = float(diff.mean())
        std = float(diff.std(ddof=1)) if n > 1 else 0.0
        axis.scatter(mean, diff, s=14, alpha=0.6)
        axis.axhline(bias, color="#0b3d6b", linestyle="--", label=f"Bias={bias:.2f}")
        axis.axhline(bias + 1.96 * std, color="#d9822b", linestyle=":", label=f"+1.96SD={bias + 1.96 * std:.2f}")
        axis.axhline(bias - 1.96 * std, color="#d9822b", linestyle=":", label=f"-1.96SD={bias - 1.96 * std:.2f}")
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "Insufficient samples (n<2)", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("Mean of predicted and reference")
    axis.set_ylabel("Predicted − Reference")
    axis.set_title(title)
    axis.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    target = Path(path)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target
