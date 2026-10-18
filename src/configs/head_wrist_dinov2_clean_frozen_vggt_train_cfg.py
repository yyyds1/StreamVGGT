"""Single-stage wrist-DINOv2 training with a frozen VGGT visual backbone and no waypoint noise."""
from copy import deepcopy
from pathlib import Path
from .head_wrist_dinov2_clean_lora_train_cfg import head_wrist_dinov2_clean_lora_train_cfg

head_wrist_dinov2_clean_frozen_vggt_train_cfg = deepcopy(head_wrist_dinov2_clean_lora_train_cfg)
cfg = head_wrist_dinov2_clean_frozen_vggt_train_cfg
cfg.__name__ = "Config: frozen VGGT + independent wrist DINOv2 + clean SE3, full RDT"
cfg.save_root = str(Path(__file__).resolve().parents[2] / "train_out/head_wrist_dinov2_clean_frozen_vggt_20261016_120000")
cfg.use_lora = False
cfg.freeze_visual_backbone = True
cfg.transformer_resume = False
cfg.action_head_resume = False
cfg.transformer_resume_from = None
cfg.action_head_resume_from = None
cfg.transformer_warm_start = None
cfg.action_head_warm_start = None
