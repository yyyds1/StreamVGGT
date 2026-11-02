"""50-task outer-gripper midpoint guidance with absolute joint actions."""
from copy import deepcopy

from .head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg import (
    head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg,
)

head_camera_sparse_joint_midpoint_absolute_50tasks_train_cfg = deepcopy(
    head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg
)
cfg = head_camera_sparse_joint_midpoint_absolute_50tasks_train_cfg
cfg.__name__ = "Config: outer gripper midpoint sparse guidance + absolute joint action, 50 RoboTwin tasks"
cfg.robotwin_action_space = "joint"
cfg.joint_action_representation = "absolute"
cfg.action_dim = 14
cfg.rdt.action_dim = 14
cfg.used_action_channel_ids = list(range(14))
cfg.inverse_used_action_channel_ids = list(range(14))
cfg.norm_stat = deepcopy(cfg.norm_stats_by_action_mode["joint_absolute"])
cfg.joint_norm_stat = deepcopy(cfg.norm_stats_by_action_mode["joint_absolute"])
cfg.joint_state_norm_stat = deepcopy(cfg.norm_stats_by_action_mode["joint_absolute"])
cfg.freeze_visual_backbone = True
cfg.use_lora = False
cfg.lora_rank = 128
cfg.obs_cam_keys = ["head_camera"]
cfg.current_obs_cam_keys = ["head_camera"]
cfg.history_obs_cam_keys = ["head_camera"]
cfg.view_position_cam_keys = ["head_camera"]
cfg.num_steps = None
cfg.max_training_time_seconds = 23 * 60 * 60 + 30 * 60
cfg.save_interval = 2000
cfg.save_root = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/joint_midpoint_frozen_headcam_50tasks_8xh100_p4_24h_20261101"
)
