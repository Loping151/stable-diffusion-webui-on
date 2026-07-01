#!/usr/bin/env bash
#
# stable-diffusion-webui-on — launcher.
#
# Defaults to the first GPU (CUDA0). Override the card with e.g. CUDA_VISIBLE_DEVICES=1 ./run.sh
# Local launch flags are read from webui-user.sh (COMMANDLINE_ARGS); extra args are passed through:
#   ./run.sh --port 7860 --listen
#
set -uo pipefail
cd "$(dirname "$0")"

# Pick the idle card by default.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Load local, gitignored settings (COMMANDLINE_ARGS, proxies, ...) if present.
if [ -f webui-user.sh ]; then
  # shellcheck disable=SC1091
  source webui-user.sh
fi

VENV_DIR="${VENV_DIR:-venv}"
PY="$VENV_DIR/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: venv not found at '$VENV_DIR'. Run ./install.sh first."
  exit 1
fi

echo ">> Launching on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec "$PY" -u launch.py ${COMMANDLINE_ARGS:-} "$@"
