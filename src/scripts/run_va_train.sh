#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29511"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"vga_robotwin_train"}
SAVE_ROOT=${SAVE_ROOT:-"./train_out"}
SINGLE_TASK=${SINGLE_TASK:-"adjust_bottle"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m train_va --config-name ${config_name} ${overrides} \
    $(if [ -n "${SAVE_ROOT}" ]; then printf '%s' "--save-root ${SAVE_ROOT}"; fi) \
    $(if [ -n "${SINGLE_TASK}" ]; then printf '%s' "--single-task ${SINGLE_TASK}"; fi)