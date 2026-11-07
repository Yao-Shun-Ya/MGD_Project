"""Write a human-readable study report from status + evaluation summary."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_mgd" / "study_9h"
STATUS = OUT / "study_status.json"
SUMMARY = OUT / "eval" / "summary.json"
REPORT = OUT / "REPORT.md"


def main() -> int:
    lines = ["# 9h MGD study report", ""]
    if STATUS.is_file():
        status = json.loads(STATUS.read_text(encoding="utf-8"))
        lines += ["## Training runs", "", "| experiment | status | best weights |", "|---|---|---|"]
        for item in status.get("experiments", []):
            lines.append(f"| {item.get('name')} | {item.get('status')} | `{item.get('best_weights')}` |")
        lines.append("")
        lines.append(f"Wall time: {status.get('wall_seconds')} s; finished: {status.get('finished_at')}")
        lines.append("")

    if SUMMARY.is_file():
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        lines += ["## Evaluation summary", "", "| experiment | Dice | IoU | clDice | HD95 | ASSD |", "|---|---|---|---|---|---|"]
        for item in summary:
            def f(key, digits=4):
                value = item.get(key)
                try:
                    return f"{float(value):.{digits}f}"
                except (TypeError, ValueError):
                    return "—"
            lines.append(
                f"| {item.get('name')} | {f('dice')} | {f('iou')} | {f('cldice')} | {f('hd95', 2)} | {f('assd', 2)} |"
            )
        lines.append("")

        mf = next((x for x in summary if x.get("name") == "mgdformer_patient"), {})
        if mf:
            lines += ["## MGD-Former: MTE, uncertainty, clinical, subgroups", ""]
            for key, label, digits in (
                ("dice", "Dice (384)", 4), ("tiles_dice", "Dice (MTE)", 4),
                ("tiles_cldice", "clDice (MTE)", 4), ("tiles_hd95", "HD95 (MTE)", 2),
                ("dice_tta8", "Dice (8x TTA)", 4),
                ("auroc_tile_variance", "AUROC tile variance", 3),
                ("auroc_entropy", "AUROC entropy", 3),
                ("auroc_tta8_error", "AUROC TTA8 error", 3),
                ("clinical_meiboscore_accuracy", "Clinical Meiboscore accuracy", 3),
                ("clinical_meiboscore_mae", "Clinical Meiboscore MAE", 3),
                ("clinical_meiboscore_qwk", "Clinical Meiboscore QWK", 3),
                ("dice_upper", "Dice upper", 4), ("dice_lower", "Dice lower", 4),
            ):
                value = mf.get(key)
                try:
                    lines.append(f"- **{label}**: {float(value):.{digits}f}")
                except (TypeError, ValueError):
                    lines.append(f"- **{label}**: —")
            lines.append("")

        clean = next((x for x in summary if x.get("name") == "leakage_unet_patient"), {})
        leaky = next((x for x in summary if x.get("name") == "leakage_unet_official_leaky"), {})
        lines += ["## Leakage control (clean-official subset)", ""]
        for label, item in (("patient-level", clean), ("image-level leaky", leaky)):
            try:
                lines.append(f"- **{label}** Dice: {float(item.get('dice')):.4f}")
            except (TypeError, ValueError):
                lines.append(f"- **{label}** Dice: —")
        try:
            delta = float(leaky["dice"]) - float(clean["dice"])
            lines.append(f"- **leaky - clean**: {delta:+.4f}")
        except (KeyError, TypeError, ValueError):
            lines.append("- **leaky - clean**: —")
        lines.append("")

    lines += ["## Reproducibility", "",
              "- Split: patient-level 694/153/152 (239/50/49 patients)",
              "- File-list signature: 6bdacfb0…; assignment signature: 7a6fd581…",
              "- Official 200 test: 166 patients; 326/800 non-test images from the same patients",
              "- Key artefacts: `outputs_mgd/study_9h/eval/summary.json`, per-sample CSVs, run dirs",
              "", "## Next steps", "",
              "1. Run 5-fold for the two leading models if the single-holdout ranking is close.",
              "2. Check the clean-official leakage gap against the confidence interval; do not overclaim.",
              "3. If MTE beats 384 but not TTA, report it as a cost-aware alternative, not a new SOTA.",
              "4. Update the paper macros with `tools/inject_results.py` and recompile both PDFs."]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
