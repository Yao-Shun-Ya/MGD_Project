"""Finish the 9h study with a targeted reallocation.

Plan:
1. Wait for the MGD-PriorNet entry to appear in study_status.json.
2. Stop the orchestrator so it does not start the clDice ablation.
3. Continue AttentionUNet from its last checkpoint for 6 more epochs
   (it reached val Dice 0.744 at epoch 9 and was still improving).
4. Run the official-200 leaky UNet control.
5. Update study_status.json so wait_and_eval.py evaluates the final set.

The clDice ablation is intentionally dropped: a fair AttentionUNet comparison
is more valuable than a budget-truncated ablation. This is recorded in the
paper as "not completed under the 9h budget".
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_mgd"
STATUS = OUT / "study_9h" / "study_status.json"
DATA = ROOT / "data"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def env() -> dict:
    value = os.environ.copy()
    value["PYTHONPATH"] = str(ROOT)
    value["MGD_DATA_DIR"] = str(DATA)
    value["MGD_MODEL_DIR"] = str(ROOT / "models_weights")
    value["MGD_OUTPUT_DIR"] = str(OUT)
    value["MGD_PRETRAINED_DIR"] = str(ROOT / "models_weights" / "pretrained")
    return value


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


def kill_orchestrator() -> None:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*run_9h.py*' -or $_.CommandLine -like '*run_train.py*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True)
    pids = [int(x) for x in result.stdout.split() if x.strip().isdigit()]
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    print(f"[finish] stopped pids: {pids}", flush=True)
    time.sleep(5)


def write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_train(config_path: Path, log_path: Path) -> None:
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "run_train.py"), "--config", str(config_path)],
            cwd=str(ROOT), env=env(), stdout=log, stderr=subprocess.STDOUT, check=False,
        )


def main() -> int:
    while True:
        if STATUS.is_file():
            try:
                payload = json.loads(STATUS.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            names = {item.get("name") for item in payload.get("experiments", [])}
            if "priornet_patient" in names:
                break
        time.sleep(60)

    print("[finish] priornet finished; stopping remaining orchestrator experiments", flush=True)
    kill_orchestrator()

    # Remove an incomplete clDice run, if the orchestrator already started it.
    for run in (OUT / "runs").iterdir():
        if run.is_dir() and "study_unet_no_cldice" in run.name:
            print(f"[finish] removing incomplete {run.name}", flush=True)
            shutil.rmtree(run, ignore_errors=True)

    # ---- AttentionUNet continuation ----
    att_runs = [p for p in (OUT / "runs").iterdir() if p.is_dir() and "study_attention_patient" in p.name]
    att_runs.sort(key=lambda p: p.stat().st_mtime)
    if not att_runs:
        print("[finish] no AttentionUNet run; abort", flush=True)
        return 1
    att = att_runs[-1]
    ckpts = sorted((att / "checkpoints").glob("*.mgd.ckpt"))
    if not ckpts:
        print("[finish] no AttentionUNet checkpoint; abort", flush=True)
        return 1
    resume = ckpts[-1]
    print(f"[finish] resuming AttentionUNet from {resume.name}", flush=True)
    att_cfg = {
        "model_name": "AttentionUNet",
        "epochs": 15,
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
        "run_name": "attention_continue",
        "save_every": 1,
        "keep_checkpoints": 2,
        "ema_decay": 0.99,
        "resume_from": str(resume),
    }
    att_cfg_path = ROOT / "tools" / "attention_continue.json"
    write_config(att_cfg_path, att_cfg)
    run_train(att_cfg_path, OUT / "study_9h" / "logs" / "attention_continue.log")

    cont = find_run("attention_continue")
    old_best = sorted((att / "checkpoints").glob("*_best.pth"))
    att_best = (cont or {}).get("best_weights") or (str(old_best[-1]) if old_best else None)
    print(f"[finish] AttentionUNet best -> {att_best}", flush=True)

    # ---- Official-200 leaky control ----
    shutil.copy2(DATA / "splits" / "official_leaky.json", DATA / "dataset_splits.json")
    leaky_cfg = {
        "model_name": "UNet",
        "epochs": 25,
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
    leaky_cfg_path = ROOT / "tools" / "study_9h" / "configs" / "unet_official_leaky.json"
    write_config(leaky_cfg_path, leaky_cfg)
    print("[finish] running official-200 leaky UNet", flush=True)
    run_train(leaky_cfg_path, OUT / "study_9h" / "logs" / "unet_official_leaky.log")
    shutil.copy2(DATA / "splits" / "patient_holdout.json", DATA / "dataset_splits.json")
    leaky = find_run("study_unet_official_leaky")

    # ---- Update status ----
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    entries = {item.get("name"): item for item in payload.get("experiments", [])}
    if "priornet_patient" not in entries:
        prior = find_run("study_priornet_patient")
        if prior:
            entries["priornet_patient"] = {
                "name": "priornet_patient", "status": "timeout_kept_best",
                "finished_at": now(), **prior,
            }
    if "attention_patient" in entries and att_best:
        cont = cont or {}
        entries["attention_patient"].update({
            "run_dir": cont.get("run_dir", entries["attention_patient"].get("run_dir")),
            "latest_checkpoint": cont.get("latest_checkpoint", entries["attention_patient"].get("latest_checkpoint")),
            "best_weights": att_best,
            "status": "finished_continued",
            "finished_at": now(),
        })
    entries.pop("unet_no_cldice", None)
    if leaky and leaky.get("best_weights"):
        entries["unet_official_leaky"] = {
            "name": "unet_official_leaky", "arch": "UNet",
            "status": "finished", "finished_at": now(), **leaky,
        }
    payload["experiments"] = list(entries.values())
    payload["deadline"] = "2026-11-07T09:29:49+00:00"
    payload["finished_at"] = now()
    payload["wall_seconds"] = round(time.time() - 1757204989.0, 1)
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[finish] status updated; wait_and_eval will take over", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
