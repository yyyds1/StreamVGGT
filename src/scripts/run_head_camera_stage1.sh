#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${NGPU:=8}"
: "${MASTER_PORT:=29511}"
: "${LOG_RANK:=0}"
: "${TORCHFT_LIGHTHOUSE:=http://localhost:29510}"
: "${CONFIG_NAME:=head_camera_temporal_train}"
: "${SAVE_ROOT:=./train_out/head_camera_stage1}"
: "${SINGLE_TASK:=}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in \
        /opt/conda/envs/RoboTwin2/bin/python \
        /root/miniforge3/envs/RoboTwin2/bin/python \
        /opt/miniconda/envs/RoboTwin2/bin/python \
        "$(command -v python3 2>/dev/null || true)"; do
        if [[ -x "$candidate" ]] && "$candidate" -c "import torch" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
: "${PYTHON_BIN:?Could not find a Python executable with torch; set PYTHON_BIN to the RoboTwin2 environment Python}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHFT_LIGHTHOUSE
NVIDIA_LIB_DIRS="$("$PYTHON_BIN" -c 'import pathlib, site; print(":".join(str(p) for base in site.getsitepackages() for p in sorted((pathlib.Path(base) / "nvidia").glob("*/lib")) if p.is_dir()))' 2>/dev/null || true)"
if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

args=(--config-name "$CONFIG_NAME" --save-root "$SAVE_ROOT" --no-use-lora --freeze-visual-backbone)
if [[ -n "$SINGLE_TASK" ]]; then args+=(--single-task "$SINGLE_TASK"); fi
if [[ $# -gt 0 ]]; then args+=("$@"); fi
exec "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$NGPU" --local-ranks-filter="$LOG_RANK" \
    --master_port "$MASTER_PORT" --tee 3 -m train_va "${args[@]}"
