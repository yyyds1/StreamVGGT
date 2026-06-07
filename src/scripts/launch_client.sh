#!/bin/bash
# Avoid forcing system library paths by default. On cluster/container machines this
# can make SAPIEN/Vulkan/CUDA extensions load incompatible native libraries.
if [[ "${PREPEND_SYSTEM_LIBS:-0}" == "1" ]]; then
  export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
fi
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
DEBUG_NATIVE=${DEBUG_NATIVE:-0}
DEBUG_NATIVE_ONLY=${DEBUG_NATIVE_ONLY:-0}

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

save_root=${1:-'./results'}
task_name=${2:-"adjust_bottle"}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0
HOST='127.0.0.1'
PORT=29055
HEADLESS=${HEADLESS:-1}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-"200"}
USE_EXPERT_MARKED_RGB=${USE_EXPERT_MARKED_RGB:-1}

headless_flag=""
if [[ "${HEADLESS}" == "1" ]]; then
  headless_flag="--headless"
fi

if [[ "${DEBUG_NATIVE}" == "1" ]]; then
  python - <<'PY'
import os
import sys
import subprocess

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
check("numpy/scipy", "import numpy, scipy; print(numpy.__version__, scipy.__version__)")
check("sapien import", "import sapien; print(getattr(sapien, '__version__', 'unknown'))")
check("sapien renderer", "import sapien.core as sapien; r = sapien.SapienRenderer(); print(type(r).__name__)")
check("warp", "import warp; print(getattr(warp, '__version__', 'unknown'), hasattr(warp, 'torch'))")
check("curobo", "from curobo.types.math import Pose; import curobo; print(getattr(curobo, '__file__', 'unknown'))")

print("\n[check] nvidia-smi")
subprocess.run(["bash", "-lc", "nvidia-smi | head -n 12"], check=False)
print("\n[check] vulkaninfo summary")
subprocess.run(["bash", "-lc", "vulkaninfo --summary 2>&1 | sed -n '1,80p'"], check=False)
PY
  if [[ "${DEBUG_NATIVE_ONLY}" == "1" ]]; then
    exit 0
  fi
fi

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
    ${headless_flag} \
    $(if [ -n "${MAX_EPISODE_STEPS}" ]; then printf '%s' "--max_episode_steps ${MAX_EPISODE_STEPS}"; fi) \
    $(if [[ "${USE_EXPERT_MARKED_RGB}" == "1" || "${USE_EXPERT_MARKED_RGB}" == "true" || "${USE_EXPERT_MARKED_RGB}" == "True" ]]; then printf '%s' "--use_expert_marked_rgb"; else printf '%s' "--no-use_expert_marked_rgb"; fi) \
    --host ${HOST} \
    --port ${PORT} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --save_root ${save_root} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num 100
