"""Noisy outer gripper-midpoint + absolute EE + wrist DINOv2 + VGGT LoRA."""
from copy import deepcopy
from pathlib import Path
from .head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg import (
    head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg,
)

midpoint_ee_wrist_dinov2_noisy_lora_50tasks_train_cfg = deepcopy(
    head_camera_sparse_midpoint_ee_absolute_50tasks_train_cfg
)
cfg = midpoint_ee_wrist_dinov2_noisy_lora_50tasks_train_cfg
cfg.__name__ = "Config: noisy outer gripper midpoint + absolute EE + wrist DINOv2 + VGGT LoRA"

# Add two frozen wrist DINOv2-small views while retaining the head VGGT input.
cfg.dataset_type = "robotwin_lerobot"
cfg.require_all_requested_tasks = True
cfg.enable_wrist_dinov2 = True
cfg.wrist_dinov2_model = str(Path(__file__).resolve().parents[2] / "ckpt/dinov2-small")
cfg.wrist_dinov2_local_files_only = True
cfg.wrist_dinov2_image_size = 224
cfg.obs_cam_keys = ["head_camera", "left_camera", "right_camera"]
cfg.current_obs_cam_keys = ["head_camera"]
cfg.history_obs_cam_keys = ["head_camera"]
cfg.view_position_cam_keys = ["head_camera"]
cfg.wrist_obs_cam_keys = ["left_camera", "right_camera"]
cfg.separate_history_current_obs_views = True

# VGGT LoRA fine-tuning; keep the wrist encoders frozen.
cfg.use_lora = True
cfg.freeze_visual_backbone = False
cfg.lora_rank = 256
cfg.lora_alpha = 64.0
cfg.lora_dropout = 0.05

# Match the established waypoint-noise distribution.
cfg.enable_se3_guidance = True
cfg.se3_guidance_train_noise = True
cfg.se3_guidance_position_noise_std = 0.005
cfg.se3_guidance_rotation_noise_deg = 5.0
cfg.se3_guidance_gripper_noise_std = 0.02
cfg.se3_guidance_duration_noise_std = 1.0

# No trainer step or wall-clock deadline; platform-side defaults are not passed.
cfg.num_steps = None
cfg.max_training_time_seconds = None
cfg.batch_size = 12
cfg.gradient_accumulation_steps = 1
cfg.metric_logger = "tensorboard"
cfg.enable_wandb = False
cfg.wandb_mode = "offline"
cfg.save_interval = 2000
cfg.save_root = "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/train_out/midpoint_ee_wrist_dinov2_noisy_lora_50tasks_8xh100_20261104"
