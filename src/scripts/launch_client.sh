#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${EVAL_GPU:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${EVAL_GPU}"
fi

# Avoid forcing system library paths by default. On cluster/container machines this
# can make SAPIEN/Vulkan/CUDA extensions load incompatible native libraries.
if [[ "${PREPEND_SYSTEM_LIBS:-0}" == "1" ]]; then
    export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
fi
# Prefer CUDA/cuDNN/NCCL libraries bundled with the active conda environment.
# On some machines, accidentally loading /usr/lib CUDA libraries makes even
# torch.cuda.manual_seed_all fail with "illegal instruction".
if [[ "${PREFER_CONDA_CUDA_LIBS:-1}" == "1" ]]; then
    NVIDIA_LIB_DIRS="$(python -c 'import pathlib, site; print(":".join(str(p) for base in site.getsitepackages() for p in sorted((pathlib.Path(base) / "nvidia").glob("*/lib")) if p.is_dir()))' 2>/dev/null || true)"
    if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export ROBOTWIN_CPU_ONLY_TORCH_SEED=${ROBOTWIN_CPU_ONLY_TORCH_SEED:-1}
if [[ "${DEBUG_CUDA:-0}" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${SRC_DIR}/.." && pwd)"
cd "${SRC_DIR}"

TRAIN_OUT_ROOT=${TRAIN_OUT_ROOT:-"${SRC_DIR}/train_out"}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-${START_PORT:-29055}}
TASK_NAME=${TASK_NAME:-${1:-}}
SAVE_ROOT=${SAVE_ROOT:-${2:-"${SRC_DIR}/results"}}
CKPT_DIR=${CKPT_DIR:-${3:-}}
POLICY_NAME=${POLICY_NAME:-ACT}
EVAL_CONFIG=${EVAL_CONFIG:-}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-0}
MODEL_NAME=${MODEL_NAME:-0}
CKPT_SETTING=${CKPT_SETTING:-}
SEED=${SEED:-0}
HEADLESS=${HEADLESS:-1}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-400}
EXPERT_TARGET_POS_THRESHOLD=${EXPERT_TARGET_POS_THRESHOLD:-0.10}
EXPERT_TARGET_ROT_THRESHOLD=${EXPERT_TARGET_ROT_THRESHOLD:-60.0}
EXPERT_TARGET_GRIPPER_THRESHOLD=${EXPERT_TARGET_GRIPPER_THRESHOLD:-0.2}
VIDEO_GUIDANCE_SCALE=${VIDEO_GUIDANCE_SCALE:-5}
ACTION_GUIDANCE_SCALE=${ACTION_GUIDANCE_SCALE:-1}
TEST_NUM=${TEST_NUM:-10}
STOP_ON_ERROR=${STOP_ON_ERROR:-0}
SUMMARY_FILE=${SUMMARY_FILE:-}
JOINT_USE_DIRECT_CONTROL=${JOINT_USE_DIRECT_CONTROL:-1}
JOINT_DIRECT_CONTROL_STEPS=${JOINT_DIRECT_CONTROL_STEPS:-15}
CLIP_JOINT_ACTION_TO_DATASET_BOUNDS=${CLIP_JOINT_ACTION_TO_DATASET_BOUNDS:-1}
DEBUG_NATIVE=${DEBUG_NATIVE:-0}
DEBUG_NATIVE_ONLY=${DEBUG_NATIVE_ONLY:-0}

# Edit this list for batch evaluation. If TASK_NAME or the first positional
# argument is set, only that one task is evaluated.
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

if [[ -z "${CKPT_DIR}" || "${CKPT_DIR}" == "latest" ]]; then
    CKPT_DIR="$(find_latest_checkpoint)"
fi
CKPT_DIR="$(cd "${CKPT_DIR}" && pwd)"
TRAINING_CONFIG="${CKPT_DIR}/training_config.json"
if [[ ! -f "${TRAINING_CONFIG}" ]]; then
    echo "Missing checkpoint training config: ${TRAINING_CONFIG}" >&2
    exit 1
fi

eval "$(
    python - "$TRAINING_CONFIG" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
values = {
    "CFG_ROBOTWIN_ACTION_SPACE": cfg.get("robotwin_action_space", "ee"),
    "CFG_ACTION_REPRESENTATION": cfg.get("action_representation", "absolute"),
    "CFG_JOINT_ACTION_REPRESENTATION": cfg.get("joint_action_representation", "absolute"),
    "CFG_USE_EXPERT_MARKED_RGB": cfg.get("use_expert_marked_rgb", False),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

ROBOTWIN_ACTION_SPACE=${ROBOTWIN_ACTION_SPACE:-${CFG_ROBOTWIN_ACTION_SPACE}}
ACTION_REPRESENTATION=${ACTION_REPRESENTATION:-${CFG_ACTION_REPRESENTATION}}
JOINT_ACTION_REPRESENTATION=${JOINT_ACTION_REPRESENTATION:-${CFG_JOINT_ACTION_REPRESENTATION}}
USE_EXPERT_MARKED_RGB=${USE_EXPERT_MARKED_RGB:-${CFG_USE_EXPERT_MARKED_RGB}}
if [[ -z "${CKPT_SETTING}" ]]; then
    CKPT_SETTING="$(basename "${CKPT_DIR}")"
fi

if [[ -z "${EVAL_CONFIG}" ]]; then
    if [[ -f "${REPO_DIR}/robotwin-labeled/policy/${POLICY_NAME}/deploy_policy.yml" ]]; then
        EVAL_CONFIG="${REPO_DIR}/robotwin-labeled/policy/${POLICY_NAME}/deploy_policy.yml"
    else
        EVAL_CONFIG="/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/policy/${POLICY_NAME}/deploy_policy.yml"
    fi
fi

headless_flag=()
if [[ "${HEADLESS}" == "1" || "${HEADLESS}" == "true" || "${HEADLESS}" == "True" ]]; then
    headless_flag=(--headless)
fi

rgb_flag=(--no-use_expert_marked_rgb)
if [[ "${USE_EXPERT_MARKED_RGB}" == "1" || "${USE_EXPERT_MARKED_RGB}" == "true" || "${USE_EXPERT_MARKED_RGB}" == "True" ]]; then
    rgb_flag=(--use_expert_marked_rgb)
fi

if [[ "${DEBUG_NATIVE}" == "1" ]]; then
    python - <<'PY'
import os
import subprocess
import sys

print("python:", sys.executable)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))
print("VK_ICD_FILENAMES:", os.environ.get("VK_ICD_FILENAMES", ""))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
print("DISPLAY:", os.environ.get("DISPLAY", ""))
print("XDG_RUNTIME_DIR:", os.environ.get("XDG_RUNTIME_DIR", ""))

def check(name, code):
    print(f"\n[check] {name}")
    try:
        ns = {}
        exec(code, ns, ns)
        print("[ok]", name)
    except Exception as exc:
        print("[fail]", name, repr(exc))

check("torch", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())")
check("torch cuda device", "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))")
check("torch cuda manual_seed", "import torch; torch.cuda.manual_seed_all(0); torch.cuda.synchronize(); print('manual_seed_all ok')")
check("numpy/scipy", "import numpy, scipy; print(numpy.__version__, scipy.__version__)")
check("sapien import", "import sapien; print(getattr(sapien, '__version__', 'unknown'))")
check("sapien renderer", "import sapien.core as sapien; r = sapien.SapienRenderer(); print(type(r).__name__)")
check("warp", "import warp; print(getattr(warp, '__version__', 'unknown'), hasattr(warp, 'torch'))")
check("curobo", "from curobo.types.math import Pose; import curobo; print(getattr(curobo, '__file__', 'unknown'))")
subprocess.run(["bash", "-lc", "nvidia-smi | head -n 12"], check=False)
subprocess.run(["bash", "-lc", "vulkaninfo --summary 2>&1 | sed -n '1,80p'"], check=False)
PY
    if [[ "${DEBUG_NATIVE_ONLY}" == "1" ]]; then
        exit 0
    fi
fi

mkdir -p "${SAVE_ROOT}"

echo "[launch_client] checkpoint: ${CKPT_DIR}"
echo "[launch_client] action space: ${ROBOTWIN_ACTION_SPACE}"
echo "[launch_client] ee representation: ${ACTION_REPRESENTATION}"
echo "[launch_client] joint representation: ${JOINT_ACTION_REPRESENTATION}"
echo "[launch_client] task config: ${TASK_CONFIG}"
echo "[launch_client] use expert marked RGB: ${USE_EXPERT_MARKED_RGB}"
echo "[launch_client] expert target thresholds: pos=${EXPERT_TARGET_POS_THRESHOLD}, rot=${EXPERT_TARGET_ROT_THRESHOLD}, gripper=${EXPERT_TARGET_GRIPPER_THRESHOLD}"
echo "[launch_client] joint direct control: ${JOINT_USE_DIRECT_CONTROL}, steps=${JOINT_DIRECT_CONTROL_STEPS}"
echo "[launch_client] clip joint action to dataset bounds: ${CLIP_JOINT_ACTION_TO_DATASET_BOUNDS}"
echo "[launch_client] server: ${HOST}:${PORT}"

if [[ -z "${SUMMARY_FILE}" ]]; then
    SUMMARY_FILE="${SAVE_ROOT}/success_rates_${CKPT_SETTING}_$(date +%Y%m%d_%H%M%S).tsv"
fi
mkdir -p "$(dirname "${SUMMARY_FILE}")"
if [[ ! -f "${SUMMARY_FILE}" ]]; then
    printf "task\tsuccess_rate\tsuccesses\ttotal\tstatus\tmetrics_file\n" > "${SUMMARY_FILE}"
fi
echo "[launch_client] summary: ${SUMMARY_FILE}"

selected_tasks=()
if [[ -n "${TASKS:-}" ]]; then
    TASKS="${TASKS//,/ }"
    # shellcheck disable=SC2206
    selected_tasks=(${TASKS})
elif [[ -n "${TASK_NAME}" ]]; then
    selected_tasks=("${TASK_NAME}")
else
    selected_tasks=("${TASK_LIST[@]}")
fi

append_summary() {
    local task="$1"
    local status="$2"
    local metrics_file="$3"
    python - "$SUMMARY_FILE" "$task" "$status" "$metrics_file" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
task = sys.argv[2]
status = sys.argv[3]
metrics_path = Path(sys.argv[4])

rate = "nan"
succ = "nan"
total = "nan"
if metrics_path.is_file():
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rate = str(float(data.get("succ_rate", float("nan"))))
    succ = str(int(float(data.get("succ_num", float("nan")))))
    total = str(int(float(data.get("total_num", float("nan")))))

with summary_path.open("a", encoding="utf-8") as f:
    f.write(f"{task}\t{rate}\t{succ}\t{total}\t{status}\t{metrics_path}\n")
PY
}

latest_metrics_file() {
    local task="$1"
    python - "$SAVE_ROOT" "$task" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
task = sys.argv[2]
candidates = list(root.glob(f"stseed-*/metrics/{task}/res.json"))
if candidates:
    print(max(candidates, key=lambda p: p.stat().st_mtime))
PY
}

run_one_task() {
    local task="$1"
    local status="ok"
    local metrics_file=""

    echo "[launch_client] task: ${task} | test_num=${TEST_NUM}"
    joint_direct_flag=(--joint_use_direct_control)
    if [[ "${JOINT_USE_DIRECT_CONTROL}" == "0" || "${JOINT_USE_DIRECT_CONTROL}" == "false" || "${JOINT_USE_DIRECT_CONTROL}" == "False" ]]; then
        joint_direct_flag=(--no-joint_use_direct_control)
    fi
    joint_clip_flag=(--clip_joint_action_to_dataset_bounds)
    if [[ "${CLIP_JOINT_ACTION_TO_DATASET_BOUNDS}" == "0" || "${CLIP_JOINT_ACTION_TO_DATASET_BOUNDS}" == "false" || "${CLIP_JOINT_ACTION_TO_DATASET_BOUNDS}" == "False" ]]; then
        joint_clip_flag=(--no-clip_joint_action_to_dataset_bounds)
    fi
    if ! PYTHONWARNINGS=ignore::UserWarning \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python -m evaluation.robotwin.eval_polict_client_openpi \
            --config "${EVAL_CONFIG}" \
            "${headless_flag[@]}" \
            --host "${HOST}" \
            --port "${PORT}" \
            --save_root "${SAVE_ROOT}" \
            --video_guidance_scale "${VIDEO_GUIDANCE_SCALE}" \
            --action_guidance_scale "${ACTION_GUIDANCE_SCALE}" \
            --test_num "${TEST_NUM}" \
            --max_episode_steps "${MAX_EPISODE_STEPS}" \
            "${joint_direct_flag[@]}" \
            --joint_direct_control_steps "${JOINT_DIRECT_CONTROL_STEPS}" \
            "${joint_clip_flag[@]}" \
            "${rgb_flag[@]}" \
            --overrides \
            --task_name "${task}" \
            --task_config "${TASK_CONFIG}" \
            --train_config_name "${TRAIN_CONFIG_NAME}" \
            --model_name "${MODEL_NAME}" \
            --ckpt_setting "${CKPT_SETTING}" \
            --seed "${SEED}" \
            --policy_name "${POLICY_NAME}" \
            --robotwin_action_space "${ROBOTWIN_ACTION_SPACE}" \
            --action_representation "${ACTION_REPRESENTATION}" \
            --joint_action_representation "${JOINT_ACTION_REPRESENTATION}" \
            --expert_target_pos_threshold "${EXPERT_TARGET_POS_THRESHOLD}" \
            --expert_target_rot_threshold "${EXPERT_TARGET_ROT_THRESHOLD}" \
            --expert_target_gripper_threshold "${EXPERT_TARGET_GRIPPER_THRESHOLD}" \
            --checkpoint_dir "${CKPT_DIR}"; then
        status="failed"
    fi

    metrics_file="$(latest_metrics_file "${task}")"
    if [[ "${status}" == "ok" && -z "${metrics_file}" ]]; then
        status="missing_metrics"
    fi
    append_summary "${task}" "${status}" "${metrics_file:-missing}"
    echo "[launch_client] recorded ${task} -> ${SUMMARY_FILE}"

    if [[ "${status}" != "ok" && "${STOP_ON_ERROR}" == "1" ]]; then
        return 1
    fi
    return 0
}

echo "[launch_client] evaluating ${#selected_tasks[@]} task(s)"
for task in "${selected_tasks[@]}"; do
    run_one_task "${task}"
done

python - "$SUMMARY_FILE" <<'PY'
import math
import sys
from pathlib import Path

summary = Path(sys.argv[1])
rows = []
for line in summary.read_text(encoding="utf-8").splitlines()[1:]:
    parts = line.split("\t")
    if len(parts) >= 5 and parts[4] == "ok":
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if not math.isnan(value):
            rows.append(value)
if rows:
    print(f"[launch_client] mean_success_rate={sum(rows) / len(rows):.6f} over {len(rows)} completed task(s)")
print(f"[launch_client] summary_file={summary}")
PY
