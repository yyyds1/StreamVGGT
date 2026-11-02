"""50-task sparse outer-gripper-midpoint guidance with an absolute EE action head."""
from copy import deepcopy

from .head_camera_sparse_7token_quat_duration_50tasks_scratch_train_cfg import (
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg,
)
from .va_robotwin_cfg import va_robotwin_cfg

head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg = deepcopy(
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg
)
cfg = head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg
cfg.__name__ = "Config: outer gripper midpoint sparse guidance + 30D absolute EE action, 50 RoboTwin tasks"
cfg.dataset_path = "/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data_sparse_awe_gripper_midpoint_outer_50tasks_20261026"
cfg.single_task = (
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
cfg.require_all_requested_tasks = True
cfg.robotwin_lerobot_repo_subdir = "demo_randomized/lerobot"

# Explicitly select the generated outer gripper-midpoint waypoint fields and videos.
cfg.sparse_waypoint_field_prefix = "sparse_gripper_midpoint_waypoints"
cfg.sparse_waypoint_video_suffix = ".rgb_sparse_gripper_midpoint_waypoints"
cfg.expected_sparse_waypoint_definition = "outermost fingertip mesh midpoint"
cfg.use_sparse_waypoint_guidance = True
cfg.sparse_waypoint_pose_mode = "gripper_midpoint"
cfg.sparse_waypoint_pose_offset_ee = [0.15557, 0.0, -0.000103515]  # online replay only; cached midpoint records are used as-is
cfg.enable_se3_guidance = True
cfg.se3_guidance_source = "client"  # use the selected cached midpoint waypoints during evaluation
cfg.use_sparse_waypoint_marked_rgb = True
cfg.use_expert_marked_rgb = False
cfg.use_expert_plan_marked_rgb = False
cfg.sparse_waypoint_max_count = 6
cfg.sparse_waypoint_max_duration = 32

# Train the policy action target as absolute EE pose + gripper (16 active of 30 channels).
cfg.robotwin_action_space = "ee"
cfg.action_representation = "absolute"
cfg.action_dim = 30
cfg.rdt.action_dim = 30
cfg.used_action_channel_ids = list(va_robotwin_cfg.used_action_channel_ids)
cfg.inverse_used_action_channel_ids = list(va_robotwin_cfg.inverse_used_action_channel_ids)
cfg.norm_stats_by_action_mode = deepcopy(va_robotwin_cfg.norm_stats_by_action_mode)
cfg.norm_stat = deepcopy(cfg.norm_stats_by_action_mode["ee_absolute"])
cfg.action_head_resume = False
cfg.action_head_resume_from = None
cfg.action_head_warm_start = None

# Warm-start the visual/guidance backbone from the sparse 50-task checkpoint.
# Its 14D joint action projections are skipped by the action-dimension adapter;
# the 30D EE action projections and entire RDT action head start fresh.
cfg.transformer_resume = False
cfg.transformer_resume_from = None
cfg.transformer_warm_start = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/head_camera_sparse_7token_quat_duration_50tasks_scratch_stage1/"
    "train_log_20261014_161354/ckpt/checkpoint_step_42000/transformer/"
    "diffusion_pytorch_model.safetensors"
)

cfg.save_root = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/outer_gripper_midpoint_absolute_ee_50tasks_8xh100_bs12_20261026"
)
cfg.batch_size = 12
cfg.metric_logger = "tensorboard"
cfg.enable_wandb = False
cfg.wandb_mode = "offline"
cfg.cfg_prob = 0.0
cfg.num_steps = 36000
cfg.gradient_accumulation_steps = 1
cfg.learning_rate = 1e-4
cfg.save_interval = 2000
cfg.load_worker = 2
cfg.dataset_init_worker = 0
cfg.require_all_requested_tasks = True
