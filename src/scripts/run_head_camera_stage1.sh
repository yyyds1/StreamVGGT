#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NGPU:=1}"
: "${MASTER_PORT:=29512}"
: "${LOG_RANK:=0}"
: "${TORCHFT_LIGHTHOUSE:=http://localhost:29510}"
: "${CONFIG_NAME:=head_camera_temporal_train}"
: "${SAVE_ROOT:=./train_out/head_camera_sparse_se3_stage1}"
: "${PYTHON_BIN:=$(command -v python3 || command -v python)}"
: "${SINGLE_TASK:=adjust_bottle}"
: "${DATASET_PATH:=${ROBOTWIN_ROOT:-$(cd ../../../robotwin-labeled 2>/dev/null && pwd)}/data/adjust_bottle/demo_randomized/lerobot}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" TORCHFT_LIGHTHOUSE
NVIDIA_LIB_DIRS="$($PYTHON_BIN -c 'import pathlib,site;print(":".join(str(p) for b in site.getsitepackages() for p in sorted((pathlib.Path(b)/"nvidia").glob("*/lib")) if p.is_dir()))' 2>/dev/null || true)"
if [[ -n "$NVIDIA_LIB_DIRS" ]]; then export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NGPU" --local-ranks-filter="$LOG_RANK" --master_port "$MASTER_PORT" --tee 3 -m train_va --config-name "$CONFIG_NAME" --save-root "$SAVE_ROOT" --single-task "$SINGLE_TASK" --dataset-path "$DATASET_PATH" "$@"
