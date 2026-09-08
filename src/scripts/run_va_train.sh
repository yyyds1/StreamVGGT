#!/usr/bin/env bash

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29511"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"vga_robotwin_train"}
SAVE_ROOT=${SAVE_ROOT:-"./train_out"}
# Preserve an explicitly empty SINGLE_TASK value. Empty means train on all tasks.
SINGLE_TASK=${SINGLE_TASK-"adjust_bottle"}

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export TORCHFT_LIGHTHOUSE=${torchft_lighthouse}

# Prefer CUDA/cuDNN libraries bundled with the active Python environment.
# This avoids loading incompatible system cuDNN from /usr/lib.
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, site; print(":".join(str(p) for base in site.getsitepackages() for p in sorted((pathlib.Path(base) / "nvidia").glob("*/lib")) if p.is_dir()))' 2>/dev/null || true)"
if [ -n "${NVIDIA_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cmd=(
    python -m torch.distributed.run
    --nproc_per_node="${num_gpu}"
    --local-ranks-filter="${log_rank}"
    --master_port "${master_port}"
    --tee 3
    -m train_va
    --config-name "${config_name}"
)

if [ -n "${SAVE_ROOT}" ]; then
    cmd+=(--save-root "${SAVE_ROOT}")
fi

if [ -n "${SINGLE_TASK}" ]; then
    cmd+=(--single-task "${SINGLE_TASK}")
fi

if [ "$#" -gt 0 ]; then
    cmd+=("$@")
fi

"${cmd[@]}"
