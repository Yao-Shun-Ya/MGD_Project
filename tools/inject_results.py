"""Inject evaluated numbers from outputs_mgd/study_9h/eval/summary.json into
public/Paper/{zh,en}/data/results.tex.

Only macros that have a value in the summary are replaced; unresolved macros
stay as \ph so the paper never silently shows a fabricated number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "xianeryijian" / "public" / "Paper"
SUMMARY = ROOT / "outputs_mgd" / "study_9h" / "eval" / "summary.json"


def fmt(value, digits=4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return f"{number:.{digits}f}"


def build_macros(summary: list[dict]) -> dict[str, str]:
    by_name = {item["name"]: item for item in summary}
    macros: dict[str, str] = {}

    model_map = {
        "mgdformer_patient": "MF",
        "unet_patient": "UNet",
        "attention_patient": "Att",
        "wrcanet_patient": "WRCANet",
        "priornet_patient": "Prior",
    }
    for name, suffix in model_map.items():
        item = by_name.get(name)
        if not item:
            continue
        assd_macro = "\\gASSDAlt" if suffix == "Att" else f"\\gASSD{suffix}"
        for key, macro in (("dice", f"\\gDice{suffix}"), ("iou", f"\\gIoU{suffix}"),
                           ("cldice", f"\\gClDice{suffix}"), ("hd95", f"\\gHD{suffix}"),
                           ("assd", assd_macro)):
            value = fmt(item.get(key))
            if value is not None:
                macros[macro] = value

    mf = by_name.get("mgdformer_patient", {})
    if mf:
        for key, macro, digits in (
            ("tiles_dice", "\\mteDiceMF", 4), ("tiles_hd95", "\\mteHDMF", 2),
            ("tiles_cldice", "\\mteClDiceMF", 4), ("dice_tta8", "\\ttaEight", 4),
            ("dice", "\\ttaOff", 4), ("auroc_tile_variance", "\\aurocVarMF", 3),
            ("auroc_entropy", "\\aurocEntropyMF", 3), ("auroc_tta8_error", "\\aurocTTAEightMF", 3),
            ("clinical_meiboscore_accuracy", "\\accMF", 3), ("clinical_meiboscore_mae", "\\maeMF", 3),
            ("clinical_meiboscore_qwk", "\\qwkMF", 3), ("dice_upper", "\\diceUpperMF", 4),
            ("dice_lower", "\\diceLowerMF", 4),
        ):
            value = fmt(mf.get(key), digits)
            if value is not None:
                macros[macro] = value
        if fmt(mf.get("tiles_dice")) and fmt(mf.get("dice")):
            macros["\\mteGainMF"] = f"{float(mf['tiles_dice']) - float(mf['dice']):+.4f}"
        if fmt(mf.get("dice_upper")) and fmt(mf.get("dice_lower")):
            macros["\\diceGapMF"] = f"{float(mf['dice_upper']) - float(mf['dice_lower']):+.4f}"

    if "unet_patient" in by_name:
        macros["\\ablFull"] = fmt(by_name["unet_patient"].get("dice")) or "\\ph"
    if "unet_no_cldice" in by_name:
        macros["\\ablNoClDice"] = fmt(by_name["unet_no_cldice"].get("dice")) or "\\ph"

    paired = ROOT / "outputs_mgd" / "study_9h" / "eval" / "paired_tests.json"
    if paired.is_file():
        data = json.loads(paired.read_text(encoding="utf-8"))
        for key, macro in (("MF_vs_Att", "\\pMFAtt"), ("MF_vs_UNet", "\\pMFUNet"),
                           ("MF_vs_Prior", "\\pMFPrior")):
            item = data.get(key)
            if item and fmt(item.get("p"), 3) is not None:
                macros[macro] = f"{float(item['p']):.3g}"

    clean = by_name.get("leakage_unet_patient", {}).get("dice")
    leaky = by_name.get("leakage_unet_official_leaky", {}).get("dice")
    if fmt(clean) is not None:
        macros["\\leakDiceClean"] = fmt(clean)
    if fmt(leaky) is not None:
        macros["\\leakDiceLeaky"] = fmt(leaky)
    if fmt(clean) is not None and fmt(leaky) is not None:
        macros["\\leakDelta"] = f"{float(leaky) - float(clean):+.4f}"
    return macros


def inject(path: Path, macros: dict[str, str]) -> int:
    text = path.read_text(encoding="utf-8")
    count = 0
    for macro, value in macros.items():
        if value == "\\ph":
            continue
        pattern = re.compile(re.escape(f"\\newcommand{{{macro}}}") + r"\{[^}]*\}")
        replacement = f"\\newcommand{{{macro}}}{{{value}}}"
        # Use a function replacement: macro names contain \g..., which
        # re.sub would otherwise parse as an invalid group reference.
        text, n = pattern.subn(lambda _match, repl=replacement: repl, text)
        count += n
    # Only enable the pgfplots figures if every value they need is present.
    required = ("\\leakDiceClean", "\\leakDiceLeaky", "\\gDiceMF", "\\mteDiceMF",
                "\\ttaEight", "\\aurocVarMF", "\\aurocEntropyMF", "\\aurocTTAEightMF")
    if all(macro in macros for macro in required):
        text = text.replace("\\newcommand{\\resultsavailable}{0}", "\\newcommand{\\resultsavailable}{1}")
    path.write_text(text, encoding="utf-8")
    return count


def main() -> int:
    if not SUMMARY.is_file():
        print(f"summary not found: {SUMMARY}")
        return 1
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    macros = build_macros(summary)
    print(f"macros: {len(macros)}")
    for lang in ("zh", "en"):
        path = PAPER / lang / "data" / "results.tex"
        if path.is_file():
            n = inject(path, macros)
            print(f"injected {n} into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
