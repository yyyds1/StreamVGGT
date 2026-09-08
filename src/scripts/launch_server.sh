#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${SERVER_GPU:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${SERVER_GPU}"
fi

# Prefer CUDA/cuDNN/NCCL libraries bundled with the active conda environment.
# Loading system cuDNN first can abort inference with errors such as:
#   libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig
if [[ "${PREFER_CONDA_CUDA_LIBS:-1}" == "1" ]]; then
    NVIDIA_LIB_DIRS="$(python -c 'import os, pathlib, site, sys
paths = []
for base in site.getsitepackages():
    paths.extend(p for p in sorted((pathlib.Path(base) / "nvidia").glob("*/lib")) if p.is_dir())
prefix = os.environ.get("CONDA_PREFIX")
if prefix:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_dir = pathlib.Path(prefix) / "lib" / pyver / "site-packages"
    paths.extend(p for p in sorted((site_dir / "nvidia").glob("*/lib")) if p.is_dir())
print(":".join(dict.fromkeys(str(p) for p in paths)))' 2>/dev/null || true)"
    if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
if [[ "${DEBUG_CUDA:-0}" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SRC_DIR}"

START_PORT=${START_PORT:-29055}
MASTER_PORT=${MASTER_PORT:-29061}
NGPU=${NGPU:-1}
CONFIG_NAME=${CONFIG_NAME:-vga_robotwin}
SAVE_ROOT=${SAVE_ROOT:-"${SRC_DIR}/visualization"}
TRAIN_OUT_ROOT=${TRAIN_OUT_ROOT:-"${SRC_DIR}/train_out"}

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
    step = int(match.group(1)) if match else -1
    transformer = ckpt / "transformer" / "diffusion_pytorch_model.safetensors"
    action_head = ckpt / "action_head" / "diffusion_pytorch_model.safetensors"
    if transformer.is_file() and action_head.is_file():
        candidates.append((cfg.stat().st_mtime, step, ckpt))
if not candidates:
    raise SystemExit(f"No valid checkpoint with training_config.json found under {root}")
print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
}

CKPT_DIR=${CKPT_DIR:-${1:-}}
if [[ -z "${CKPT_DIR}" || "${CKPT_DIR}" == "latest" ]]; then
    CKPT_DIR="$(find_latest_checkpoint)"
fi
CKPT_DIR="$(cd "${CKPT_DIR}" && pwd)"
EVAL_CONFIG=${EVAL_CONFIG:-"${CKPT_DIR}/training_config.json"}

if [[ ! -f "${EVAL_CONFIG}" ]]; then
    echo "Missing evaluation config: ${EVAL_CONFIG}" >&2
    exit 1
fi

mkdir -p "${SAVE_ROOT}"

extra_args=()
if [[ -n "${ROBOTWIN_ACTION_SPACE:-}" ]]; then
    extra_args+=(--robotwin-action-space "${ROBOTWIN_ACTION_SPACE}")
fi
if [[ -n "${ACTION_REPRESENTATION:-}" ]]; then
    extra_args+=(--action-representation "${ACTION_REPRESENTATION}")
fi
if [[ -n "${JOINT_ACTION_REPRESENTATION:-}" ]]; then
    extra_args+=(--joint-action-representation "${JOINT_ACTION_REPRESENTATION}")
fi
if [[ -n "${ACTION_CHUNK_EXEC_STEPS:-}" ]]; then
    extra_args+=(--action-chunk-exec-steps "${ACTION_CHUNK_EXEC_STEPS}")
fi
if [[ -n "${JOINT_DELTA_ACTION_CHUNK_EXEC_STEPS:-}" ]]; then
    extra_args+=(--joint-delta-action-chunk-exec-steps "${JOINT_DELTA_ACTION_CHUNK_EXEC_STEPS}")
fi
if [[ -n "${RDT_EE_TARGET_CONDITION_SOURCE:-}" ]]; then
    extra_args+=(--rdt-ee-target-condition-source "${RDT_EE_TARGET_CONDITION_SOURCE}")
fi
if [[ "${DISABLE_JOINT_DELTA_WARM_START_EVAL:-0}" == "1" || "${DISABLE_JOINT_DELTA_WARM_START_EVAL:-0}" == "true" || "${DISABLE_JOINT_DELTA_WARM_START_EVAL:-0}" == "True" ]]; then
    extra_args+=(--disable-joint-delta-warm-start-eval)
fi

echo "[launch_server] checkpoint: ${CKPT_DIR}"
echo "[launch_server] eval config: ${EVAL_CONFIG}"
echo "[launch_server] port: ${START_PORT}"
echo "[launch_server] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if [[ -n "${ACTION_CHUNK_EXEC_STEPS:-}" ]]; then
    echo "[launch_server] action_chunk_exec_steps override: ${ACTION_CHUNK_EXEC_STEPS}"
fi
if [[ -n "${JOINT_DELTA_ACTION_CHUNK_EXEC_STEPS:-}" ]]; then
    echo "[launch_server] joint_delta_action_chunk_exec_steps: ${JOINT_DELTA_ACTION_CHUNK_EXEC_STEPS}"
fi
if [[ -n "${RDT_EE_TARGET_CONDITION_SOURCE:-}" ]]; then
    echo "[launch_server] RDT ee-target condition source: ${RDT_EE_TARGET_CONDITION_SOURCE}"
fi
if [[ "${DISABLE_JOINT_DELTA_WARM_START_EVAL:-0}" == "1" || "${DISABLE_JOINT_DELTA_WARM_START_EVAL:-0}" == "true" || "${DISABLE_JOINT_DELTA_WARM_START_EVAL:-0}" == "True" ]]; then
    echo "[launch_server] joint-delta warm start: disabled"
fi
if [[ "${DEBUG_NATIVE:-0}" == "1" ]]; then
    python - <<'PY'
import os
import site
import sys
from pathlib import Path

print("[launch_server][debug] python:", sys.executable)
print("[launch_server][debug] LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))
print("[launch_server][debug] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
print("[launch_server][debug] nvidia lib dirs:")
for base in site.getsitepackages():
    for path in sorted((Path(base) / "nvidia").glob("*/lib")):
        if path.is_dir():
            print("  ", path)
PY
fi

python -m torch.distributed.run \
    --nproc_per_node "${NGPU}" \
    --master_port "${MASTER_PORT}" \
    va_server.py \
    --config-name "${CONFIG_NAME}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --eval-config "${EVAL_CONFIG}" \
    --port "${START_PORT}" \
    --save_root "${SAVE_ROOT}" \
    "${extra_args[@]}"
