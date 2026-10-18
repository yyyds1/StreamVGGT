"""Single-stage wrist-DINOv2 + VGGT LoRA training without waypoint noise."""
from copy import deepcopy
from pathlib import Path
from .head_wrist_dinov2_noisy_lora_train_cfg import head_wrist_dinov2_noisy_lora_train_cfg

head_wrist_dinov2_clean_lora_train_cfg = deepcopy(head_wrist_dinov2_noisy_lora_train_cfg)
cfg = head_wrist_dinov2_clean_lora_train_cfg
cfg.__name__ = "Config: head VGGT LoRA + frozen independent wrist DINOv2 + clean SE3, full RDT"
cfg.save_root = str(Path(__file__).resolve().parents[2] / "train_out/head_wrist_dinov2_clean_lora_20261016_110000")
cfg.se3_guidance_train_noise = False
cfg.se3_guidance_position_noise_std = 0.0
cfg.se3_guidance_rotation_noise_deg = 0.0
cfg.se3_guidance_gripper_noise_std = 0.0
cfg.se3_guidance_duration_noise_std = 0.0
cfg.transformer_resume = False
cfg.action_head_resume = False
cfg.transformer_resume_from = None
cfg.action_head_resume_from = None
cfg.transformer_warm_start = None
cfg.action_head_warm_start = None
