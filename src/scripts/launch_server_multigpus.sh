#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SRC_DIR}"

START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
NGPU=${NGPU:-8}
CONFIG_NAME=${CONFIG_NAME:-vga_robotwin}
TRAIN_OUT_ROOT=${TRAIN_OUT_ROOT:-"${SRC_DIR}/train_out"}
SAVE_ROOT=${SAVE_ROOT:-"${SRC_DIR}/visualization"}
CKPT_DIR=${CKPT_DIR:-${1:-}}
EVAL_CONFIG=${EVAL_CONFIG:-}
SINGLE_TRAJECTORY=${SINGLE_TRAJECTORY:-0}
SINGLE_TRAJECTORY_EPISODE_INDEX=${SINGLE_TRAJECTORY_EPISODE_INDEX:-}
SINGLE_TRAJECTORY_REPO_ID=${SINGLE_TRAJECTORY_REPO_ID:-}
LOG_DIR=${LOG_DIR:-"${SRC_DIR}/logs"}

find_latest_checkpoint() {
    python - "$TRAIN_OUT_ROOT" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
pattern = re.compile(r"checkpoint_step_(\d+)$")
candidates = []
for cfg in root.rglob("training_config.json"):
    ckpt = cfg.parent
    match = pattern.match(ckpt.name)
    transformer = ckpt / "transformer" / "diffusion_pytorch_model.safetensors"
    action_head = ckpt / "action_head" / "diffusion_pytorch_model.safetensors"
    if match and transformer.is_file() and action_head.is_file():
        candidates.append((cfg.stat().st_mtime, int(match.group(1)), ckpt))
if not candidates:
    raise SystemExit(f"No valid checkpoint found under {root}")
print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
}

if [[ -z "${CKPT_DIR}" || "${CKPT_DIR}" == "latest" ]]; then
    CKPT_DIR="$(find_latest_checkpoint)"
fi
CKPT_DIR="$(cd "${CKPT_DIR}" && pwd)"
EVAL_CONFIG=${EVAL_CONFIG:-"${CKPT_DIR}/training_config.json"}

if [[ ! -f "${EVAL_CONFIG}" ]]; then
    echo "Missing evaluation config: ${EVAL_CONFIG}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"

checkpoint_args=(
    --checkpoint-dir "${CKPT_DIR}"
    --eval-config "${EVAL_CONFIG}"
)
if [[ -n "${RDT_EE_TARGET_CONDITION_SOURCE:-}" ]]; then
    checkpoint_args+=(--rdt-ee-target-condition-source "${RDT_EE_TARGET_CONDITION_SOURCE}")
fi

batch_time=$(date +%Y%m%d_%H%M%S)

echo "[launch_server_multigpus] checkpoint: ${CKPT_DIR}"
echo "[launch_server_multigpus] eval config: ${EVAL_CONFIG}"
echo "[launch_server_multigpus] GPUs: ${NGPU}"

for ((i = 0; i < NGPU; i++)); do
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Task ${i}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    optional_args=()
    if [[ "${SINGLE_TRAJECTORY}" == "1" || "${SINGLE_TRAJECTORY}" == "true" || "${SINGLE_TRAJECTORY}" == "True" ]]; then
        optional_args+=(--single-trajectory)
    fi
    if [[ -n "${SINGLE_TRAJECTORY_EPISODE_INDEX}" ]]; then
        optional_args+=(--single-trajectory-episode-index "${SINGLE_TRAJECTORY_EPISODE_INDEX}")
    fi
    if [[ -n "${SINGLE_TRAJECTORY_REPO_ID}" ]]; then
        optional_args+=(--single-trajectory-repo-id "${SINGLE_TRAJECTORY_REPO_ID}")
    fi

    CUDA_VISIBLE_DEVICES="${i}" \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port "${CURRENT_MASTER_PORT}" \
        va_server.py \
        --config-name "${CONFIG_NAME}" \
        --save_root "${SAVE_ROOT}" \
        --port "${CURRENT_PORT}" \
        "${checkpoint_args[@]}" \
        "${optional_args[@]}" \
        > "${LOG_FILE}" 2>&1 &
    sleep 2
done

echo "All ${NGPU} instances have been launched in the background."
wait
