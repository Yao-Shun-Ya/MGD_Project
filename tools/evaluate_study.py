"""Evaluate the 9h study: segmentation, clinical grading, subgroups, tiling
ensemble, uncertainty, cross-model ensemble, and the official-split leakage
comparison.

Usage:
    python tools/evaluate_study.py --quick
    python tools/evaluate_study.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from mgd.core.config import get_settings
from mgd.core.data.pairing import scan_dataset
from mgd.core.data.splits import SplitSpec, get_splits
from mgd.core.imaging.clinical import cldice_score, meiboscore_from_rloss, r_loss
from mgd.core.imaging.metrics import mask_metrics
from mgd.core.imaging.preprocess import load_gray, resize_gray
from mgd.core.imaging.visualize import pixel_entropy
from mgd.core.inference.service import InferenceOptions, ModelService

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_mgd" / "study_9h" / "eval"
OUT.mkdir(parents=True, exist_ok=True)
STATUS = ROOT / "outputs_mgd" / "study_9h" / "study_status.json"
META = ROOT / "data" / "dataset_metadata.csv"
OFFICIAL = ROOT / "data" / "splits" / "official_leaky.json"


def load_metadata() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(META, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["stem"]] = row
    return rows


def load_status(path: Path = STATUS) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload.get("experiments", []) if item.get("best_weights")]


def pairs_by_stem(pairs, stems: list[str]):
    """Select Pair objects by stem. `pairs` is already a list of Pair."""
    lookup = {pair.stem: pair for pair in pairs}
    return [lookup[s] for s in stems if s in lookup]


def gt_masks(pair, settings):
    size = settings.target_size
    gland = (resize_gray(load_gray(pair.gland_mask), size, cv2.INTER_NEAREST) > 0).astype(np.uint8)
    eyelid = (resize_gray(load_gray(pair.eyelid_mask), size, cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return gland, eyelid


def metrics_from_prob(prob: np.ndarray, pair, settings, threshold: float = 0.7) -> dict:
    gt_gland, gt_eyelid = gt_masks(pair, settings)
    pred = (prob >= threshold).astype(np.uint8)
    result = mask_metrics(pred, gt_gland)
    r_pred = r_loss(pred, gt_eyelid)
    r_gt = r_loss(gt_gland, gt_eyelid)
    return {
        "dice": result.dice,
        "iou": result.iou,
        "hd95": result.hd95,
        "assd": result.assd,
        "precision": result.precision,
        "recall": result.recall,
        "cldice": cldice_score(pred, gt_gland),
        "r_loss_pred": r_pred.r_loss_percent,
        "r_loss_gt": r_gt.r_loss_percent,
        "meiboscore_pred": meiboscore_from_rloss(r_pred.r_loss_percent),
        "meiboscore_mask": meiboscore_from_rloss(r_gt.r_loss_percent),
        "pred_area": int(pred.sum()),
        "gt_area": int(gt_gland.sum()),
        "errors": " | ".join(result.errors),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = ["dice", "iou", "hd95", "assd", "precision", "recall", "cldice", "r_loss_pred", "r_loss_gt", "dice_tta8"]
    out = {}
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
        out[f"{key}_n"] = len(values)
    out["samples"] = len(rows)
    return out


def qwk(y_true: list[int], y_pred: list[int], labels=(0, 1, 2, 3)) -> float:
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    total = matrix.sum()
    if total == 0:
        return float("nan")
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / total
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    observed = (weights * matrix).sum() / total
    exp = (weights * expected).sum() / total
    return float(1 - observed / exp) if exp > 0 else float("nan")


def auroc(scores: list[float], labels: list[int]) -> float:
    """Tie-aware AUROC; labels=1 means error/failure."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = rank
        i = j + 1
    pos_rank_sum = sum(ranks[k] for k in range(n) if pairs[k][1] == 1)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_model(service, pairs, arch, weights, settings, metadata, protocols=("single",)):
    rows = []
    tile_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    for pair in pairs:
        rec = {"stem": pair.stem, "arch": arch, "eyelid": metadata.get(pair.stem, {}).get("eyelid", "unknown")}
        if "single" in protocols:
            opts = InferenceOptions(tile_infer=True, tile_size=384, tile_overlap=0.25, threshold=0.7, tta="关闭")
            prob, _, _, _ = service.probability_dual(pair.image, arch, opts, weights)
            rec.update(metrics_from_prob(prob, pair, settings))
        if "tta8" in protocols:
            opts = InferenceOptions(tile_infer=True, tile_size=384, tile_overlap=0.25, threshold=0.7, tta="8倍增强 (多尺度+翻转)")
            prob, _, _, stack = service.probability_dual(pair.image, arch, opts, weights)
            rec["dice_tta8"] = metrics_from_prob(prob, pair, settings)["dice"]
        if "tiles" in protocols:
            probs = []
            for tile in (256, 384, 512):
                opts = InferenceOptions(tile_infer=True, tile_size=tile, tile_overlap=0.25, threshold=0.7, tta="关闭")
                p, _, _, _ = service.probability_dual(pair.image, arch, opts, weights)
                probs.append(p.astype(np.float32))
            stack = np.stack(probs, axis=0)
            mean = stack.mean(axis=0)
            rec.update({f"{k}_tiles": v for k, v in metrics_from_prob(mean, pair, settings).items()})
            rec["tile_variance"] = float(stack.var(axis=0).mean())
            rec["entropy"] = float(pixel_entropy(mean).mean())
            tile_probs[pair.stem] = probs
        rows.append(rec)
    return rows, tile_probs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="single-tile patient test only")
    parser.add_argument("--status", type=Path, default=STATUS, help="study_status.json to read")
    args = parser.parse_args()

    settings = get_settings()
    index = scan_dataset(settings)
    split = get_splits(index.pairs, SplitSpec(), settings)
    test_pairs = split.test
    metadata = load_metadata()
    experiments = load_status(args.status)
    if not experiments:
        print("no completed experiments in study_status.json")
        return 1

    summary = []
    all_rows: dict[str, list[dict]] = {}
    for exp in experiments:
        name = exp["name"]
        arch = exp["arch"]
        weights = exp["best_weights"]
        if not weights or not Path(weights).is_file():
            continue
        # The leaky control is evaluated only on the clean-official subset;
        # scoring it on the patient-level test would itself be leaky.
        # WRCANet only completed one epoch on the 8GB card (38 min/epoch);
        # it is budget-limited and must not enter the main ranking.
        if name in ("unet_official_leaky", "wrcanet_patient"):
            continue
        protocols = ("single",) if args.quick else ("single", "tiles")
        if name in ("mgdformer_patient", "unet_patient"):
            protocols = tuple(protocols) + ("tta8",)
        service = ModelService(settings, allow_unverified=True, max_cached=2)
        rows, _ = evaluate_model(service, test_pairs, arch, weights, settings, metadata, protocols)
        all_rows[name] = rows
        agg = aggregate([{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in rows])
        if "tiles" in protocols:
            tile_rows = [{"dice": r.get("dice_tiles"), "cldice": r.get("cldice_tiles"), "hd95": r.get("hd95_tiles")} for r in rows]
            tile_agg = aggregate(tile_rows)
            agg.update({f"tiles_{k}": v for k, v in tile_agg.items()})
        # clinical Meiboscore agreement
        clinical = [(int(metadata[r["stem"]]["meiboscore"]), int(r["meiboscore_pred"])) for r in rows if r["stem"] in metadata and metadata[r["stem"]].get("meiboscore", "") != ""]
        if clinical:
            y_true = [a for a, _ in clinical]
            y_pred = [b for _, b in clinical]
            agg["clinical_meiboscore_accuracy"] = float(np.mean([a == b for a, b in clinical]))
            agg["clinical_meiboscore_mae"] = float(np.mean([abs(a - b) for a, b in clinical]))
            agg["clinical_meiboscore_qwk"] = qwk(y_true, y_pred)
        # subgroup
        for subgroup in ("upper", "lower"):
            subset = [r for r in rows if r["eyelid"] == subgroup and "dice" in r]
            if subset:
                agg[f"dice_{subgroup}"] = float(np.mean([r["dice"] for r in subset]))
                agg[f"cldice_{subgroup}"] = float(np.mean([r["cldice"] for r in subset]))
        # uncertainty AUROC (failure = Dice < 0.7)
        if "tiles" in protocols:
            labels = [1 if (r.get("dice") is not None and r["dice"] < 0.7) else 0 for r in rows]
            agg["auroc_tile_variance"] = auroc([r.get("tile_variance", 0.0) for r in rows], labels)
            agg["auroc_entropy"] = auroc([r.get("entropy", 0.0) for r in rows], labels)
            if "dice_tta8" in rows[0]:
                agg["auroc_tta8_error"] = auroc([1 - r["dice_tta8"] for r in rows], labels)
        summary.append({"name": name, "arch": arch, "weights": weights, **agg})
        with open(OUT / f"{name}_per_sample.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eval] {name}: dice={agg['dice']:.4f} cldice={agg['cldice']:.4f} hd95={agg['hd95']:.2f}")

    # Leakage control: patient-level UNet vs image-level leaky UNet on the
    # clean-official subset (patients absent from the patient-level train set).
    clean_path = ROOT / "data" / "splits" / "clean_official_stems.txt"
    if clean_path.is_file():
        clean_stems = [line.strip() for line in clean_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        clean_pairs = pairs_by_stem(index.pairs, clean_stems)
        for exp in experiments:
            if exp["name"] not in ("unet_patient", "unet_official_leaky"):
                continue
            service = ModelService(settings, allow_unverified=True, max_cached=1)
            rows, _ = evaluate_model(service, clean_pairs, exp["arch"], exp["best_weights"], settings, metadata, ("single",))
            agg = aggregate([{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in rows])
            summary.append({"name": f"leakage_{exp['name']}", "arch": exp["arch"], "weights": exp["best_weights"], **agg})
            print(f"[leak] {exp['name']}: dice={agg['dice']:.4f} on {len(clean_pairs)} clean-official images")

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with open(OUT / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        keys = sorted({k for row in summary for k in row})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[eval] wrote {OUT/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
