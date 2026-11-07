"""Paired per-sample comparisons on the patient-level test set.

Reads the per-sample CSVs written by evaluate_study.py, computes paired
Wilcoxon tests and bootstrap CIs for Dice differences, and writes
eval/paired_tests.json for macro injection.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs_mgd" / "study_9h" / "eval"


def load(name: str) -> dict[str, float]:
    path = EVAL / f"{name}_per_sample.csv"
    if not path.is_file():
        return {}
    result = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                result[row["stem"]] = float(row["dice"])
            except (KeyError, TypeError, ValueError):
                continue
    return result


def main() -> int:
    mf = load("mgdformer_patient")
    if not mf:
        print("no MGD-Former per-sample CSV")
        return 1
    pairs = {
        "MF_vs_Att": "attention_patient",
        "MF_vs_UNet": "unet_patient",
        "MF_vs_Prior": "priornet_patient",
    }
    rng = np.random.default_rng(42)
    out = {}
    for key, name in pairs.items():
        other = load(name)
        common = sorted(set(mf) & set(other))
        if len(common) < 5:
            continue
        diff = np.array([mf[stem] - other[stem] for stem in common], dtype=float)
        try:
            stat, p_value = wilcoxon(diff, alternative="two-sided")
        except ValueError:
            stat, p_value = float("nan"), 1.0
        boots = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(1000)]
        out[key] = {
            "n": len(common),
            "mean_diff": float(np.mean(diff)),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "wilcoxon_stat": float(stat),
            "p": float(p_value),
            "comparison": f"MGD-Former - {name}",
        }
        print(f"{key}: diff={out[key]['mean_diff']:+.4f} "
              f"CI=[{out[key]['ci_low']:+.4f},{out[key]['ci_high']:+.4f}] p={p_value:.4g} n={len(common)}")
    (EVAL / "paired_tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {EVAL / 'paired_tests.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
