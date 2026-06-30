#!/usr/bin/env bash
#
# xwsdwebui one-click installer (uv-based, fast).
#
# Creates an isolated virtual environment and installs a modern stack:
#   Python 3.12 (override with PYTHON_VERSION) + torch 2.7.1 / CUDA 12.8 + xformers.
#
# Everything is overridable via env vars, so the SAME script also installs the old
# stack, e.g. for an RTX 20xx / CUDA 11 box:
#   PYTHON_VERSION=3.10 TORCH_VERSION=2.3.1 TORCHVISION_VERSION=0.18.1 \
#   XFORMERS_VERSION=0.0.27 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install.sh
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
XFORMERS_VERSION="${XFORMERS_VERSION:-0.0.31}"
VENV_DIR="${VENV_DIR:-venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo ">> [1/5] Creating venv '$VENV_DIR' on Python $PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" --seed "$VENV_DIR"
PY="$VENV_DIR/bin/python"

echo ">> [2/5] Installing torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION + xformers $XFORMERS_VERSION"
echo "         (index: $TORCH_INDEX_URL)"
uv pip install --python "$PY" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "xformers==$XFORMERS_VERSION" \
  --index-url "$TORCH_INDEX_URL"

echo ">> [3/5] Installing requirements_versions.txt"
uv pip install --python "$PY" -r requirements_versions.txt

echo ">> [4/5] Installing CLIP (wheel drop-in) + bitsandbytes (Flux/NF4)"
uv pip install --python "$PY" "clip-anytorch>=2.6.0" bitsandbytes

echo ">> [5/5] Verifying CUDA"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" - <<'PY'
import torch
print(f"   torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   device: {torch.cuda.get_device_name(0)}")
PY

echo ""
echo ">> Done. Start the WebUI with:  ./run.sh"
