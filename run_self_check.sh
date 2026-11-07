#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MGD_DATA_DIR="${MGD_DATA_DIR:-$PWD/data}"
export MGD_MODEL_DIR="${MGD_MODEL_DIR:-$PWD/models_weights}"
export MGD_OUTPUT_DIR="${MGD_OUTPUT_DIR:-$PWD/outputs}"
if [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -m mgd.verify_pipeline --epochs 1 --samples 6
