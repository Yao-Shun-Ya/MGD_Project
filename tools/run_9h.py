"""9-hour training orchestrator with hard per-model budgets.

Each experiment runs in its own subprocess. If the budget is exceeded, the
process is terminated; the best checkpoint saved by the engine remains usable.
Status is written after every experiment so a crash never loses the ledger.
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
OUT = ROOT / "outputs_mgd" / "study_9h"
OUT.mkdir(parents=True, exist_ok=True)
SCHEDULE = ROOT / "tools" / "study_schedule.json"
STATUS = OUT / "study_status.json"
CONFIGS = OUT / "configs"
LOGS = OUT / "logs"
CONFIGS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_run(name: str) -> dict:
    runs = sorted(
        (p for p in (ROOT / "outputs_mgd" / "runs").iterdir() if p.is_dir() and name in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return {"run_dir": None, "best_weights": None, "latest_checkpoint": None}
    run = runs[0]
    best = sorted((run / "checkpoints").glob("*_best.pth"))
    ckpts = sorted((run / "checkpoints").glob("*.mgd.ckpt"))
    return {
        "run_dir": str(run),
        "best_weights": str(best[-1]) if best else None,
        "latest_checkpoint": str(ckpts[-1]) if ckpts else None,
    }


def _prevent_sleep() -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    except Exception:
        pass


def main() -> int:
    _prevent_sleep()
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    started = time.time()
    deadline = started + float(schedule.get("total_hours", 9.0)) * 3600
    reserve = float(schedule.get("reserve_eval_seconds", 4200))
    status = {"started_at": now(), "deadline": now(), "experiments": []}

    for exp in schedule["experiments"]:
        remaining = deadline - time.time()
        if remaining <= reserve:
            status["experiments"].append({"name": exp["name"], "status": "skipped_deadline"})
            break
        budget = min(float(exp["budget_seconds"]), max(60.0, remaining - reserve))
        config = {
            "model_name": exp["arch"],
            "epochs": int(exp["epochs"]),
            "batch_size": int(exp["batch_size"]),
            "grad_accum_steps": int(exp["grad_accum_steps"]),
            "tile_mode": True,
            "tile_height": 384,
            "amp": True,
            "channels_last": True,
            "dual_task": True,
            "pretrained": exp.get("pretrained", "none"),
            "loss_name": exp.get("loss_name", "MGD 复合损失"),
            "num_workers": 0,
            "publish_best": False,
            "export_reference_report": False,
            "run_name": f"study_{exp['name']}",
            "save_every": 1,
            "keep_checkpoints": 2,
            "eval_tta": "关闭",
            "eval_threshold": 0.7,
            # 9h budget: 0.999 would still be dominated by the random decoder
            # after only a few hundred steps; 0.99 tracks the online model.
            "ema_decay": 0.99,
        }
        cfg_path = CONFIGS / f"{exp['name']}.json"
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path = LOGS / f"{exp['name']}.log"
        entry = {
            "name": exp["name"],
            "arch": exp["arch"],
            "budget_seconds": round(budget, 1),
            "started_at": now(),
            "config": str(cfg_path),
        }

        data_dir = Path(exp["data_dir"]) if exp.get("data_dir") else ROOT / "data"
        data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
        default_split = ROOT / "data" / "splits" / "patient_holdout.json"
        split_file = Path(exp["split_file"]) if exp.get("split_file") else default_split
        split_file = split_file if split_file.is_absolute() else ROOT / split_file
        if split_file.is_file():
            shutil.copy2(split_file, data_dir / "dataset_splits.json")
            entry["split_file"] = str(split_file)
        env = os.environ.copy()
        env["MGD_DATA_DIR"] = str(data_dir)
        env["MGD_MODEL_DIR"] = str(ROOT / "models_weights")
        env["MGD_OUTPUT_DIR"] = str(ROOT / "outputs_mgd")
        env["MGD_PRETRAINED_DIR"] = str(ROOT / "models_weights" / "pretrained")
        env["PYTHONPATH"] = str(ROOT)

        print(f"[start] {exp['name']} arch={exp['arch']} budget={budget/60:.1f}min split={split_file.name}", flush=True)
        cmd = [sys.executable, str(ROOT / "tools" / "run_train.py"), "--config", str(cfg_path)]
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    code = proc.wait(timeout=budget)
                    entry["status"] = "finished" if code == 0 else f"exit_{code}"
                except subprocess.TimeoutExpired:
                    # On Windows the venv python is a launcher; kill the whole
                    # process tree so the real training process cannot keep
                    # holding the GPU while the next experiment starts.
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            capture_output=True,
                        )
                    else:
                        proc.kill()
                    proc.wait(timeout=60)
                    entry["status"] = "timeout_kept_best"
        except Exception as exc:  # noqa: BLE001 - ledger must survive
            entry["status"] = f"launcher_error: {type(exc).__name__}: {exc}"
        # Restore the patient-level split regardless of outcome.
        if default_split.is_file():
            shutil.copy2(default_split, data_dir / "dataset_splits.json")
        entry.update(find_run(exp["name"]))
        entry["finished_at"] = now()
        status["experiments"].append(entry)
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[done]  {exp['name']} status={entry['status']} best={entry.get('best_weights')}", flush=True)

    status["finished_at"] = now()
    status["wall_seconds"] = round(time.time() - started, 1)
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass
    print("[all done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
