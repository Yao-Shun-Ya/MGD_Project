#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MGD_DATA_DIR="${MGD_DATA_DIR:-$PWD/data}"
export MGD_MODEL_DIR="${MGD_MODEL_DIR:-$PWD/models_weights}"
export MGD_OUTPUT_DIR="${MGD_OUTPUT_DIR:-$PWD/outputs}"
exec bash mgd/packaging/start_linux.sh "$@"
