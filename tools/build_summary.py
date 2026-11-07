"""Rebuild eval/summary.json from the existing per-sample CSVs + a fresh
leakage evaluation. Used after the main evaluation finished but the leakage
section crashed on a helper bug.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from mgd.core.config import get_settings
from mgd.core.data.pairing import scan_dataset
from mgd.core.inference.service import ModelService

from tools.evaluate_study import aggregate, auroc, evaluate_model, load_metadata, pairs_by_stem, qwk

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs_mgd" / "study_9h" / "eval"
STATUS = ROOT / "outputs_mgd" / "study_9h" / "study_status.json"


def load_rows(name: str) -> list[dict]:
    path = EVAL / f"{name}_per_sample.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def numeric(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if isinstance(v, (int, float)) and math.isfinite(v)} for r in rows]


def main() -> int:
    metadata = load_metadata()
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    experiments = {item["name"]: item for item in status.get("experiments", [])}

    specs = [
        ("mgdformer_patient", "MGD-Former"),
        ("unet_patient", "UNet"),
        ("attention_patient", "AttentionUNet"),
        ("priornet_patient", "MGD-PriorNet"),
    ]
    summary: list[dict] = []
    for name, arch in specs:
        rows = load_rows(name)
        agg = aggregate(numeric(rows))
        tile_rows = [{"dice": r.get("dice_tiles"), "cldice": r.get("cldice_tiles"), "hd95": r.get("hd95_tiles")} for r in rows]
        tile_agg = aggregate(numeric(tile_rows))
        agg.update({f"tiles_{k}": v for k, v in tile_agg.items()})
        clinical = []
        for r in rows:
            stem = str(r.get("stem"))
            if stem not in metadata or metadata[stem].get("meiboscore", "") == "":
                continue
            try:
                clinical.append((int(metadata[stem]["meiboscore"]), int(r["meiboscore_pred"])))
            except (TypeError, ValueError):
                continue
        if clinical:
            agg["clinical_meiboscore_accuracy"] = float(sum(a == b for a, b in clinical) / len(clinical))
            agg["clinical_meiboscore_mae"] = float(sum(abs(a - b) for a, b in clinical) / len(clinical))
            agg["clinical_meiboscore_qwk"] = qwk([a for a, _ in clinical], [b for _, b in clinical])
        for subgroup in ("upper", "lower"):
            subset = [r for r in rows if r.get("eyelid") == subgroup and "dice" in r]
            if subset:
                agg[f"dice_{subgroup}"] = float(sum(r["dice"] for r in subset) / len(subset))
                agg[f"cldice_{subgroup}"] = float(sum(r["cldice"] for r in subset) / len(subset))
        labels = [1 if (isinstance(r.get("dice"), float) and r["dice"] < 0.7) else 0 for r in rows]
        agg["auroc_tile_variance"] = auroc([r.get("tile_variance", 0.0) for r in rows], labels)
        agg["auroc_entropy"] = auroc([r.get("entropy", 0.0) for r in rows], labels)
        if any("dice_tta8" in r for r in rows):
            agg["auroc_tta8_error"] = auroc([1 - r["dice_tta8"] for r in rows if "dice_tta8" in r],
                                            [1 if r.get("dice", 1) < 0.7 else 0 for r in rows if "dice_tta8" in r])
        summary.append({"name": name, "arch": arch, "weights": experiments.get(name, {}).get("best_weights"), **agg})
        print(f"[summary] {name}: dice={agg['dice']:.4f} cldice={agg['cldice']:.4f} hd95={agg['hd95']:.2f}")

    # Leakage control on clean-official stems.
    settings = get_settings()
    index = scan_dataset(settings)
    clean_path = ROOT / "data" / "splits" / "clean_official_stems.txt"
    clean_stems = [line.strip() for line in clean_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    clean_pairs = pairs_by_stem(index.pairs, clean_stems)
    for name in ("unet_patient", "unet_official_leaky"):
        exp = experiments.get(name)
        if not exp or not exp.get("best_weights"):
            continue
        service = ModelService(settings, allow_unverified=True, max_cached=1)
        rows, _ = evaluate_model(service, clean_pairs, exp["arch"], exp["best_weights"], settings, metadata, ("single",))
        agg = aggregate(numeric(rows))
        summary.append({"name": f"leakage_{name}", "arch": exp["arch"], "weights": exp["best_weights"], **agg})
        print(f"[leakage] {name}: dice={agg['dice']:.4f} on {len(clean_pairs)}")

    (EVAL / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with open(EVAL / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        keys = sorted({k for row in summary for k in row})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[summary] wrote {EVAL / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
