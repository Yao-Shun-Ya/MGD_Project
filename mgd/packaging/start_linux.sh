#!/usr/bin/env bash
# Linux 交付承诺：只训练 CLI（薄壳 adapter，算法全部来自 mgd.core）。
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ ! -x "venv/bin/python" ]; then
  echo "[错误] 未找到 venv，请先运行 mgd/packaging/install_linux.sh" >&2
  exit 1
fi
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MGD_DATA_DIR="${MGD_DATA_DIR:-$(pwd)/data}"
export MGD_MODEL_DIR="${MGD_MODEL_DIR:-$(pwd)/models_weights}"
export MGD_OUTPUT_DIR="${MGD_OUTPUT_DIR:-$(pwd)/outputs}"
exec venv/bin/python -m mgd.adapters.cli.interactive
