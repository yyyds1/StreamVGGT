"""Noisy waypoint + wrist DINOv2 + VGGT LoRA rank-256, 8-GPU, 100k-step run."""
from copy import deepcopy
from pathlib import Path
from .head_wrist_dinov2_noisy_lora_bs12_rank256_train_cfg import head_wrist_dinov2_noisy_lora_bs12_rank256_train_cfg

head_wrist_dinov2_noisy_lora_bs12_rank256_8gpu_100k_train_cfg = deepcopy(
    head_wrist_dinov2_noisy_lora_bs12_rank256_train_cfg
)
cfg = head_wrist_dinov2_noisy_lora_bs12_rank256_8gpu_100k_train_cfg
cfg.__name__ = "Config: noisy SE3 + wrist DINOv2 + VGGT LoRA rank 256, batch 12, 8 GPU, 100k steps"
cfg.save_root = str(Path(__file__).resolve().parents[2] / "train_out/head_wrist_dinov2_noisy_lora_bs12_rank256_8gpu_100k_20261017")
cfg.num_steps = 100000
