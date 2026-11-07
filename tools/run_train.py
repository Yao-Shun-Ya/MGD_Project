"""Run one TrainConfig from a JSON file; used by the 9h orchestrator.

Usage:
    python tools/run_train.py --config path/to/config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgd.core.training.engine import TrainConfig
from mgd.core.training.runner import run_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = TrainConfig(**payload)
    result, context = run_training(config, {})
    payload = dict(result.__dict__)
    payload["run_id"] = context.run_id
    payload["run_dir"] = str(context.run_dir)
    print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
