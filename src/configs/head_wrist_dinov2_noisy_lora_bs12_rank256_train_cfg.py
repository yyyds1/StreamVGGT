"""Noisy waypoint + wrist DINOv2 + VGGT LoRA rank-256 training, batch size 12."""
from copy import deepcopy
from pathlib import Path
from .head_wrist_dinov2_noisy_lora_train_cfg import head_wrist_dinov2_noisy_lora_train_cfg

head_wrist_dinov2_noisy_lora_bs12_rank256_train_cfg = deepcopy(head_wrist_dinov2_noisy_lora_train_cfg)
cfg = head_wrist_dinov2_noisy_lora_bs12_rank256_train_cfg
cfg.__name__ = "Config: noisy SE3 + wrist DINOv2 + VGGT LoRA rank 256, batch 12"
cfg.save_root = str(Path(__file__).resolve().parents[2] / "train_out/head_wrist_dinov2_noisy_lora_bs12_rank256_20261016_130000")
cfg.batch_size = 12
cfg.use_lora = True
cfg.freeze_visual_backbone = False
cfg.lora_rank = 256
cfg.lora_alpha = 64.0
cfg.se3_guidance_train_noise = True
cfg.se3_guidance_position_noise_std = 0.005
cfg.se3_guidance_rotation_noise_deg = 5.0
cfg.se3_guidance_gripper_noise_std = 0.02
cfg.se3_guidance_duration_noise_std = 1.0
cfg.transformer_resume = False
cfg.action_head_resume = False
cfg.transformer_resume_from = None
cfg.action_head_resume_from = None
cfg.transformer_warm_start = None
cfg.action_head_warm_start = None
