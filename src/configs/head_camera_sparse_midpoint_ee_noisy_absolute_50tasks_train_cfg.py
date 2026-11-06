"""50-task midpoint guidance with baseline waypoint noise and absolute EE actions."""
from copy import deepcopy

from .head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg import (
    head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg,
)

head_camera_sparse_midpoint_ee_noisy_absolute_50tasks_train_cfg = deepcopy(
    head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg
)
cfg = head_camera_sparse_midpoint_ee_noisy_absolute_50tasks_train_cfg
cfg.__name__ = "Config: noisy outer gripper midpoint guidance + 30D absolute EE action, 50 RoboTwin tasks"

# Match the established baseline waypoint-noise amplitudes.
cfg.se3_guidance_train_noise = True
cfg.se3_guidance_position_noise_std = 0.005
cfg.se3_guidance_rotation_noise_deg = 5.0
cfg.se3_guidance_gripper_noise_std = 0.02
cfg.se3_guidance_duration_noise_std = 1.0

# Let the trainer stop cleanly before the platform's 24-hour hard limit.
cfg.num_steps = None
cfg.max_training_time_seconds = 23 * 60 * 60 + 30 * 60
cfg.save_interval = 2000
cfg.save_root = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/outer_gripper_midpoint_absolute_ee_noisy_50tasks_8xh100_p4_24h_20261103"
)
