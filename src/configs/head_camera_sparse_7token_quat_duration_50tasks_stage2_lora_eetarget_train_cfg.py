"""Stage-2 T_clean fine-tuning: LoRA VGA + full SE3 tokenizer + full RDT + EE target supervision."""
from copy import deepcopy
from .head_camera_sparse_7token_quat_duration_50tasks_scratch_train_cfg import (
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg,
)

head_camera_sparse_7token_quat_duration_50tasks_stage2_lora_eetarget_train_cfg = deepcopy(
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg
)
cfg = head_camera_sparse_7token_quat_duration_50tasks_stage2_lora_eetarget_train_cfg
cfg.__name__ = "Config: T_clean stage2 LoRA VGA + trainable SE3 guidance + RDT + EE target head"
cfg.save_root = "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/train_out/head_camera_sparse_7token_quat_duration_50tasks_stage2_lora_eetarget"
cfg.dataset_path = "/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data_sparse_awe"
cfg.single_task = None
cfg.enable_se3_guidance = True
cfg.use_sparse_waypoint_guidance = True
cfg.enable_ee_target_module = True
cfg.enable_ee_target_loss = True
cfg.enable_ee_target_head_eval = True
cfg.use_ee_target_as_rdt_condition = True
cfg.rdt_ee_target_condition_source = "oracle"
cfg.rdt_condition_tokens.use_ee_target_tokens = True
cfg.rdt_condition_tokens.ee_target_layers = [2, 5, 8, 11]
cfg.loss_weight_ee_target = 1.0
cfg.loss_weight_ee_target_xyz = 1.0
cfg.loss_weight_ee_target_gripper = 0.5
cfg.ee_target_sequence_len = 6
cfg.ee_target_waypoint_distance = 0.05
cfg.use_lora = True
cfg.lora_rank = 128
cfg.lora_alpha = 64.0
cfg.lora_dropout = 0.05
cfg.freeze_visual_backbone = False
cfg.transformer_resume = False
cfg.transformer_warm_start = "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/train_out/head_camera_sparse_7token_quat_duration_50tasks_scratch_stage1/train_log_20261014_161354/ckpt/checkpoint_step_42000/transformer/diffusion_pytorch_model.safetensors"
cfg.action_head_resume = False
cfg.action_head_warm_start = "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/train_out/head_camera_sparse_7token_quat_duration_50tasks_scratch_stage1/train_log_20261014_161354/ckpt/checkpoint_step_42000/action_head/diffusion_pytorch_model.safetensors"
cfg.num_steps = 36000
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 1
