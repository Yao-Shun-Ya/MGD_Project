"""论文口径的统计与报告：bootstrap CI、配对显著性、reference_report。

硬约束：任何报告都必须携带 `split_signature`；缺签名视为无效报告（不写文件）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ... import __version__

METRIC_KEYS = ("dice", "iou", "hd95", "assd", "precision", "recall", "cldice", "eyelid_dice")


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """按样本重采样给出均值的 95% 置信区间。"""
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, clean.size, size=(int(n_bootstrap), clean.size))
    means = clean[indexes].mean(axis=1)
    return {
        "mean": float(clean.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(clean.size),
    }


def wilcoxon_pairs(a_values: list[float], b_values: list[float]) -> dict[str, Any]:
    """配对 Wilcoxon 符号秩检验；长度不一致或有效配对 < 5 时返回不可用。"""
    pairs = [(a, b) for a, b in zip(a_values, b_values) if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 5:
        return {"available": False, "reason": f"有效配对仅 {len(pairs)} 例（<5）"}
    try:
        from scipy.stats import wilcoxon

        statistic, p_value = wilcoxon(
            np.asarray([a for a, _ in pairs]), np.asarray([b for _, b in pairs])
        )
        return {
            "available": True,
            "n_pairs": len(pairs),
            "statistic": float(statistic),
            "p_value": float(p_value),
            "method": "wilcoxon signed-rank（双侧，零假设：分布相同）",
        }
    except Exception as exc:
        return {"available": False, "reason": f"检验失败：{exc}"}


def _per_sample_values(report, key: str) -> list[float]:
    values: list[float] = []
    for item in report.per_sample:
        value = getattr(item, key, None)
        if value is None:
            value = getattr(item.metrics, key, float("nan"))
        values.append(float(value))
    return values


def reference_report_payload(
    report,
    *,
    split_signature: str,
    data_signature: str | None = None,
    n_bootstrap: int = 1000,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把一次评估转成可复核的报告；同时给出 per-sample 指标供配对检验。"""
    if not split_signature:
        raise ValueError("reference_report 必须携带 split_signature（反泄漏硬约束）")
    metrics: dict[str, Any] = {}
    for key in METRIC_KEYS:
        if key in ("cldice", "eyelid_dice"):
            values = _per_sample_values(report, key)
        else:
            values = [float(item.metrics.dice if key == "dice" else getattr(item.metrics, key))
                      for item in report.per_sample]
        metrics[key] = {
            **bootstrap_ci(values, n_bootstrap=n_bootstrap),
            "values": [None if not np.isfinite(v) else round(float(v), 6) for v in values],
        }
    payload = {
        "format_version": 2,
        "mgd_version": __version__,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": report.scope,
        "scope_label": report.scope_label,
        "samples": report.samples,
        "arch": report.arch,
        "device": report.device,
        "weights": str(report.weights),
        "split_signature": split_signature,
        "data_signature": data_signature,
        "bootstrap": {"iterations": int(n_bootstrap), "alpha": 0.05},
        "metrics": metrics,
        "summary": {key: value for key, value in report.summary.items()},
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "valid": True if split_signature else False,
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_reference_report(path: str | Path, payload: dict[str, Any]) -> Path:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return target
