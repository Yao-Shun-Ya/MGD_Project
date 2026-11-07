#!/usr/bin/env bash
# Linux 安装：默认 CUDA，传 --cpu 可切换 CPU 轮子。
# Linux 交付承诺为「只训练 CLI」，因此不安装 gradio（见 mgd/requirements/ui.txt）。
set -euo pipefail
cd "$(dirname "$0")/../.."

TORCH_REQ="mgd/requirements/cuda.txt"
if [ "${1:-}" = "--cpu" ]; then
  TORCH_REQ="mgd/requirements/cpu.txt"
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("需要 Python 3.10+，当前为 %s" % sys.version.split()[0])
PY

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
echo "[1/3] 安装 PyTorch（${TORCH_REQ}）..."
python -m pip install -r "${TORCH_REQ}"
echo "[2/3] 安装公共依赖..."
python -m pip install -r mgd/requirements/base.txt
echo "[3/3] 自检..."
python -c "import cv2, torch, monai; print('cv2', cv2.__version__, '| torch', torch.__version__, '| monai', monai.__version__)"
echo "[完成] 依赖安装成功。启动：mgd/packaging/start_linux.sh"
