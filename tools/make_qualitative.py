"""Generate a 4-case qualitative figure (input, GT, prediction, error map).

Run after the 9h study. Uses the MGD-Former patient-level best weight.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mgd.core.config import get_settings
from mgd.core.data.pairing import scan_dataset
from mgd.core.data.splits import SplitSpec, get_splits
from mgd.core.imaging.metrics import mask_metrics
from mgd.core.imaging.preprocess import load_gray, resize_gray
from mgd.core.inference.service import InferenceOptions, ModelService

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "xianeryijian" / "public" / "Paper"
STATUS = ROOT / "outputs_mgd" / "study_9h" / "study_status.json"
META = ROOT / "data" / "dataset_metadata.csv"


def main() -> int:
    if not STATUS.is_file():
        print("study_status.json not found")
        return 1
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    entry = next((e for e in payload.get("experiments", []) if e.get("name") == "mgdformer_patient"), None)
    if not entry or not entry.get("best_weights"):
        print("MGD-Former best weights not found")
        return 1
    weights = entry["best_weights"]

    settings = get_settings()
    index = scan_dataset(settings)
    split = get_splits(index.pairs, SplitSpec(), settings)
    metadata = {}
    with open(META, encoding="utf-8-sig") as handle:
        import csv
        for row in csv.DictReader(handle):
            metadata[row["stem"]] = row

    upper = [p for p in split.test if metadata.get(p.stem, {}).get("eyelid") == "upper"]
    lower = [p for p in split.test if metadata.get(p.stem, {}).get("eyelid") == "lower"]
    selected = upper[:2] + lower[:2]
    if len(selected) < 4:
        selected = split.test[:4]

    service = ModelService(settings, allow_unverified=True, max_cached=1)
    options = InferenceOptions(tile_infer=True, tile_size=384, tile_overlap=0.25, threshold=0.7, tta="关闭")
    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3.0 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["Input", "Reference", "Prediction", "Error map"]
    for row, pair in enumerate(selected):
        prob, _, pre, _ = service.probability_dual(pair.image, "MGD-Former", options, weights)
        gt = (resize_gray(load_gray(pair.gland_mask), settings.target_size, cv2.INTER_NEAREST) > 0).astype(np.uint8)
        pred = (prob >= 0.7).astype(np.uint8)
        gray = pre.resized
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gt_overlay = base.copy()
        gt_overlay[gt > 0] = (0.2 * gt_overlay[gt > 0] + 0.8 * np.array([0, 220, 120])).astype(np.uint8)
        pred_overlay = base.copy()
        pred_overlay[pred > 0] = (0.25 * pred_overlay[pred > 0] + 0.75 * np.array([40, 140, 255])).astype(np.uint8)
        error = base.copy()
        fp = (pred > 0) & (gt == 0)
        fn = (pred == 0) & (gt > 0)
        error[fp] = [230, 60, 60]
        error[fn] = [60, 120, 230]
        dice = mask_metrics(pred, gt).dice
        panels = [base, gt_overlay, pred_overlay, error]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=10)
        axes[row, 0].set_ylabel(f"{pair.stem}\nDice={dice:.3f}", fontsize=7)
    fig.tight_layout()
    for lang in ("zh", "en"):
        target = PAPER / lang / "figures" / "qualitative.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=220, bbox_inches="tight")
        print(f"wrote {target}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
