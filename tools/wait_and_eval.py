"""Wait for the 9h study to finish, then run evaluation and inject paper macros.

This is deliberately separate from run_9h.py so it can be launched now and
survive independently of the training orchestrator.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "outputs_mgd" / "study_9h" / "study_status.json"
EXPECTED = 7


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["MGD_DATA_DIR"] = str(ROOT / "data")
    env["MGD_MODEL_DIR"] = str(ROOT / "models_weights")
    env["MGD_OUTPUT_DIR"] = str(ROOT / "outputs_mgd")
    env["MGD_PRETRAINED_DIR"] = str(ROOT / "models_weights" / "pretrained")

    started = time.time()
    while True:
        if STATUS.is_file():
            try:
                payload = json.loads(STATUS.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            # finished_at is the authoritative signal: the orchestrator may
            # skip an experiment near the deadline and still finish cleanly.
            if payload.get("finished_at"):
                break
        if time.time() - started > 10.5 * 3600:
            print("[wait] fallback timeout; running evaluation on whatever finished", flush=True)
            break
        time.sleep(300)

    print("[wait] training finished; running evaluation", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "evaluate_study.py")],
                   cwd=str(ROOT), env=env, check=False)
    print("[wait] running paired comparisons", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "compare_models.py")],
                   cwd=str(ROOT), env=env, check=False)
    print("[wait] injecting results into paper macros", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "inject_results.py")],
                   cwd=str(ROOT), env=env, check=False)
    print("[wait] generating qualitative figure", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_qualitative.py")],
                   cwd=str(ROOT), env=env, check=False)
    print("[wait] writing report", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "study_report.py")],
                   cwd=str(ROOT), env=env, check=False)

    # Recompile both manuscripts so the PDFs contain the injected numbers.
    paper_root = ROOT.parent / "xianeryijian" / "public" / "Paper"
    for lang, engine in (("zh", "xelatex"), ("en", "pdflatex")):
        paper = paper_root / lang
        if not (paper / "main.tex").is_file():
            continue
        print(f"[wait] compiling {lang} paper", flush=True)
        subprocess.run([engine, "-interaction=nonstopmode", "main.tex"], cwd=str(paper), check=False)
        subprocess.run(["bibtex", "main"], cwd=str(paper), check=False)
        subprocess.run([engine, "-interaction=nonstopmode", "main.tex"], cwd=str(paper), check=False)
        subprocess.run([engine, "-interaction=nonstopmode", "main.tex"], cwd=str(paper), check=False)
    print("[wait] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
