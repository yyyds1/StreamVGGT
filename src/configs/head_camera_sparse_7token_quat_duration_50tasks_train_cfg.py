"""50-task seven-token structured waypoint guidance training config."""
from copy import deepcopy

from .head_camera_sparse_legacy_ee_frozen_train_cfg import (
    head_camera_sparse_legacy_ee_frozen_train_cfg,
)

head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg = deepcopy(
    head_camera_sparse_legacy_ee_frozen_train_cfg
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.__name__ = (
    "Config: seven-token SE(3) guidance, frozen VGA, 50 RoboTwin tasks"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.dataset_path = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data_sparse_awe"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.robotwin_lerobot_repo_subdir = (
    "demo_randomized/lerobot"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.single_task = (
    "adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,"
    "click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,"
    "hanging_mug,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,"
    "move_stapler_pad,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,"
    "place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_burger_fries,"
    "place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,place_empty_cup,"
    "place_fan,place_mouse_pad,place_object_basket,place_object_scale,place_object_stand,"
    "place_phone_stand,place_shoe,press_stapler,put_bottles_dustbin,put_object_cabinet,"
    "rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,"
    "stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.save_root = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/head_camera_sparse_7token_quat_duration_50tasks_stage1"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.transformer_resume = False
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.action_head_resume = False
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.transformer_warm_start = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/head_camera_sparse_legacy_ee_frozen_sparse_rgb_bs12_50tasks_stage1/"
    "train_log_20261013_140935/ckpt/checkpoint_step_30000/transformer/diffusion_pytorch_model.safetensors"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.action_head_warm_start = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/head_camera_sparse_legacy_ee_frozen_sparse_rgb_bs12_50tasks_stage1/"
    "train_log_20261013_140935/ckpt/checkpoint_step_30000/action_head/diffusion_pytorch_model.safetensors"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg.num_steps = 36000
