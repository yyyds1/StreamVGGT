"""Single-stage: head VGGT LoRA + two independent frozen wrist DINOv2 encoders."""
from copy import deepcopy
from pathlib import Path
from .head_camera_sparse_7token_quat_duration_50tasks_noisy_train_cfg import (
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_noisy_train_cfg,
)

head_wrist_dinov2_noisy_lora_train_cfg = deepcopy(
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_noisy_train_cfg
)
cfg = head_wrist_dinov2_noisy_lora_train_cfg
cfg.__name__ = "Config: head VGGT LoRA + frozen independent wrist DINOv2 + noisy SE3, full RDT"
cfg.save_root = str(Path(__file__).resolve().parents[2] / "train_out/head_wrist_dinov2_noisy_lora")
cfg.dataset_type = "robotwin_lerobot"
cfg.require_all_requested_tasks = True
cfg.enable_wrist_dinov2 = True
cfg.wrist_dinov2_model = str(Path(__file__).resolve().parents[2] / "ckpt/dinov2-small")
cfg.wrist_dinov2_local_files_only = True
cfg.wrist_dinov2_image_size = 224
# Actual sparse-AWE dataset names (cam_high/left_wrist/right_wrist aliases are not present).
cfg.obs_cam_keys = ["head_camera", "left_camera", "right_camera"]
cfg.current_obs_cam_keys = ["head_camera"]
cfg.history_obs_cam_keys = ["head_camera"]
cfg.view_position_cam_keys = ["head_camera"]
cfg.wrist_obs_cam_keys = ["left_camera", "right_camera"]
cfg.separate_history_current_obs_views = True
cfg.use_lora = True
cfg.freeze_visual_backbone = False
cfg.lora_rank = 128
cfg.lora_alpha = 64.0
cfg.lora_dropout = 0.05
cfg.enable_se3_guidance = True
cfg.se3_guidance_train_noise = True
cfg.enable_ee_target_module = False
cfg.enable_ee_target_loss = False
cfg.enable_ee_target_head_eval = False
cfg.loss_weight_ee_target = 0.0
# Legacy-named switches route SE3 tokens, NOT an EE target head.
cfg.use_ee_target_as_rdt_condition = True
cfg.rdt_condition_tokens.use_ee_target_tokens = True
cfg.transformer_resume = False
cfg.action_head_resume = False
cfg.transformer_resume_from = None
cfg.action_head_resume_from = None
cfg.transformer_warm_start = None
cfg.action_head_warm_start = None
cfg.num_steps = 36000
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 1
cfg.metric_logger = "tensorboard"
cfg.enable_wandb = False
