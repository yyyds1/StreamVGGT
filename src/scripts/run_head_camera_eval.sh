#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=$'29555','RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'1','TOKENIZERS_PARALLELISM':'false','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','ROBOTWIN_CPU_ONLY_TORCH_SEED':'1'

cd "$(dirname "$0")/.."
: "${PORT:=1106}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a checkpoint_step_* directory}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in \
        /opt/conda/envs/RoboTwin2/bin/python \
        "${PYTHON_BIN:-$(command -v python3 || command -v python)}" \
        /opt/miniconda/envs/RoboTwin2/bin/python \
        "$(command -v python3 2>/dev/null || true)"; do
        if [[ -x "$candidate" ]] && "$candidate" -c "import torch" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
: "${PYTHON_BIN:?Could not find a Python executable with torch; set PYTHON_BIN to the RoboTwin2 environment Python}"
# va_server loads CHECKPOINT_DIR/training_config.json unless EVAL_CONFIG is explicit.
args=(--checkpoint-dir "$CHECKPOINT_DIR" --port "$PORT")
if [[ -n "${EVAL_CONFIG:-}" ]]; then args+=(--eval-config "$EVAL_CONFIG"); fi
if [[ $# -gt 0 ]]; then args+=("$@"); fi
exec "$PYTHON_BIN" va_server.py "${args[@]}"
