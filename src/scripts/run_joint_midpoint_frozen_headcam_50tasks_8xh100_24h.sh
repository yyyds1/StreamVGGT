#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/src
: "${PYTHON_BIN:=/root/miniforge3/envs/RoboTwin2/bin/python}"
: "${NGPU:=8}"
: "${MASTER_PORT:=29647}"
: "${SAVE_ROOT:=/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/train_out/joint_midpoint_frozen_headcam_50tasks_8xh100_p4_24h_20261101}"
: "${CONFIG_NAME:=head_camera_sparse_joint_midpoint_absolute_50tasks_train}"
mkdir -p "$SAVE_ROOT"
(set -o noclobber; printf '%s\n' "$(date -Is)" > "$SAVE_ROOT/.launched")
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=offline
"$PYTHON_BIN" -c 'import os,torch; expected=int(os.environ["NGPU"]); print("RUNTIME",torch.__version__,"GPUs",torch.cuda.device_count(),"expected",expected,flush=True); assert torch.cuda.device_count()==expected'
"$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NGPU" --master_port "$MASTER_PORT" --local-ranks-filter=0 --tee 3 -m train_va --config-name "$CONFIG_NAME" --save-root "$SAVE_ROOT" 2>&1 | tee "$SAVE_ROOT/train.console.log"
