#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SRC_DIR}"

umask 007

NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29511"}
LOG_RANK=${LOG_RANK:-"0,1,2,3"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"vga_robotwin_train"}
SAVE_ROOT=${SAVE_ROOT:-"./train_out"}
DATASET_PATH=${DATASET_PATH:-"/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data"}
# Action space: "ee" uses end-effector pose; "joint" uses joint_action.vector.
ROBOTWIN_ACTION_SPACE=${ROBOTWIN_ACTION_SPACE:-"joint"}
# EE representation: "absolute" or "relative" when ROBOTWIN_ACTION_SPACE="ee".
ACTION_REPRESENTATION=${ACTION_REPRESENTATION:-"absolute"}
# Joint representation: "absolute" or "delta" when ROBOTWIN_ACTION_SPACE="joint".
JOINT_ACTION_REPRESENTATION=${JOINT_ACTION_REPRESENTATION:-"absolute"}
# Image stream: 0 uses raw RGB; 1 uses rgb_expert_marked visual labels.
USE_EXPERT_MARKED_RGB=${USE_EXPERT_MARKED_RGB:-"1"}
# Resume controls. RESUME=1 resumes the latest complete checkpoint under SAVE_ROOT.
# Set RESUME_CKPT_DIR to pin a specific checkpoint_step_* directory.
RESUME=${RESUME:-"0"}
RESUME_CKPT_DIR=${RESUME_CKPT_DIR:-}

# Edit this list to choose the tasks used for training.
# Leave it empty to train on all tasks under DATASET_PATH.
TASK_LIST=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

TASK_SELECTION=""
if [ "${#TASK_LIST[@]}" -gt 0 ]; then
    TASK_SELECTION="$(IFS=,; printf '%s' "${TASK_LIST[*]}")"
fi

extra_args=()

marked_rgb_flag=(--no-use-expert-marked-rgb)
if [[ "${USE_EXPERT_MARKED_RGB}" == "1" || "${USE_EXPERT_MARKED_RGB}" == "true" || "${USE_EXPERT_MARKED_RGB}" == "True" ]]; then
    marked_rgb_flag=(--use-expert-marked-rgb)
fi

if [ "$#" -gt 0 ]; then
    extra_args+=("$@")
fi

export TOKENIZERS_PARALLELISM=false
export TORCHFT_LIGHTHOUSE
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

echo "[train_robotwin_lerobot] action space: ${ROBOTWIN_ACTION_SPACE}"
echo "[train_robotwin_lerobot] ee representation: ${ACTION_REPRESENTATION}"
echo "[train_robotwin_lerobot] joint representation: ${JOINT_ACTION_REPRESENTATION}"
echo "[train_robotwin_lerobot] use expert marked RGB: ${USE_EXPERT_MARKED_RGB}"
echo "[train_robotwin_lerobot] resume: ${RESUME}${RESUME_CKPT_DIR:+ | checkpoint=${RESUME_CKPT_DIR}}"

# Prefer CUDA/cuDNN libraries bundled with the active Python environment.
# This avoids accidentally loading incompatible system cuDNN from /usr/lib.
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, site; print(":".join(str(p) for base in site.getsitepackages() for p in sorted((pathlib.Path(base) / "nvidia").glob("*/lib")) if p.is_dir()))' 2>/dev/null || true)"
if [ -n "${NVIDIA_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cmd=(
    python -m torch.distributed.run
    --nproc_per_node="${NGPU}"
    --local-ranks-filter="${LOG_RANK}"
    --master_port "${MASTER_PORT}"
    --tee 3
    -m train_va
    --config-name "${CONFIG_NAME}"
    --dataset-type robotwin_lerobot
    --dataset-path "${DATASET_PATH}"
    --robotwin-lerobot-epoch-unit episode_strided
    --robotwin-lerobot-windows-per-episode-stride 32
    --robotwin-lerobot-max-windows-per-episode 8
    --robotwin-action-space "${ROBOTWIN_ACTION_SPACE}"
    --action-representation "${ACTION_REPRESENTATION}"
    --joint-action-representation "${JOINT_ACTION_REPRESENTATION}"
    "${marked_rgb_flag[@]}"
    "${extra_args[@]}"
)

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" || -n "${RESUME_CKPT_DIR}" ]]; then
    cmd+=(--resume)
    if [[ -n "${RESUME_CKPT_DIR}" ]]; then
        cmd+=(--resume-checkpoint-dir "${RESUME_CKPT_DIR}")
    fi
fi

if [ -n "${SAVE_ROOT}" ]; then
    cmd+=(--save-root "${SAVE_ROOT}")
fi

if [ -n "${TASK_SELECTION}" ]; then
    cmd+=(--single-task "${TASK_SELECTION}")
fi

"${cmd[@]}"
