"""Run the official-200 leaky UNet control, restore the patient split, and
finalize study_status.json so wait_and_eval.py can evaluate everything.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_mgd"
STATUS = OUT / "study_9h" / "study_status.json"
DATA = ROOT / "data"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_run(name: str) -> dict | None:
    runs = [p for p in (OUT / "runs").iterdir() if p.is_dir() and name in p.name]
    if not runs:
        return None
    run = max(runs, key=lambda p: p.stat().st_mtime)
    best = sorted((run / "checkpoints").glob("*_best.pth"))
    ckpts = sorted((run / "checkpoints").glob("*.mgd.ckpt"))
    return {
        "run_dir": str(run),
        "best_weights": str(best[-1]) if best else None,
        "latest_checkpoint": str(ckpts[-1]) if ckpts else None,
    }


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["MGD_DATA_DIR"] = str(DATA)
    env["MGD_MODEL_DIR"] = str(ROOT / "models_weights")
    env["MGD_OUTPUT_DIR"] = str(OUT)
    env["MGD_PRETRAINED_DIR"] = str(ROOT / "models_weights" / "pretrained")

    config = {
        "model_name": "UNet",
        "epochs": 20,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "tile_mode": True,
        "tile_height": 384,
        "amp": True,
        "channels_last": True,
        "dual_task": True,
        "pretrained": "none",
        "loss_name": "MGD 复合损失",
        "num_workers": 0,
        "publish_best": False,
        "export_reference_report": False,
        "run_name": "study_unet_official_leaky",
        "save_every": 1,
        "keep_checkpoints": 2,
        "ema_decay": 0.99,
    }
    config_path = ROOT / "tools" / "study_9h" / "configs" / "unet_official_leaky.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = OUT / "study_9h" / "logs" / "unet_official_leaky.log"
    print("[leaky] start", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(ROOT / "tools" / "run_train.py"), "--config", str(config_path)],
                       cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    shutil.copy2(DATA / "splits" / "patient_holdout.json", DATA / "dataset_splits.json")
    print("[leaky] restored patient split", flush=True)

    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    entries = {item.get("name"): item for item in payload.get("experiments", [])}

    cont = find_run("attention_continue")
    if cont and cont.get("best_weights") and "attention_patient" in entries:
        entries["attention_patient"].update({
            "run_dir": cont["run_dir"],
            "latest_checkpoint": cont["latest_checkpoint"],
            "best_weights": cont["best_weights"],
            "status": "finished_continued",
            "finished_at": now(),
        })

    leaky = find_run("study_unet_official_leaky")
    if leaky and leaky.get("best_weights"):
        entries["unet_official_leaky"] = {
            "name": "unet_official_leaky", "arch": "UNet",
            "status": "finished", "finished_at": now(), **leaky,
        }
    entries.pop("unet_no_cldice", None)
    payload["experiments"] = list(entries.values())
    payload["deadline"] = "2026-11-07T09:29:49+00:00"
    payload["finished_at"] = now()
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[leaky] status finalized", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
