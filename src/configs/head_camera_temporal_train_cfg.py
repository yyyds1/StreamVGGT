"""Training config for the head-camera temporal experiment."""
from copy import deepcopy

from .va_vga_robotwin_train_cfg import va_vga_robotwin_train_cfg


head_camera_temporal_train_cfg = deepcopy(va_vga_robotwin_train_cfg)
head_camera_temporal_train_cfg.__name__ = "Config: head camera temporal train"
# Keep the train-only dataset and normalization from the existing VGA RobotWin config.
for key in ("obs_cam_keys", "current_obs_cam_keys", "history_obs_cam_keys", "view_position_cam_keys"):
    setattr(head_camera_temporal_train_cfg, key, ["head_camera"])
head_camera_temporal_train_cfg.separate_history_current_obs_views = True
head_camera_temporal_train_cfg.multi_view_image_mode = "vertical"
head_camera_temporal_train_cfg.temporal_window = 8
head_camera_temporal_train_cfg.history_len = 8
head_camera_temporal_train_cfg.temporal_stride = 1
head_camera_temporal_train_cfg.history_frame_stride = 1
head_camera_temporal_train_cfg.image_frame_stride = 1
head_camera_temporal_train_cfg.freeze_visual_backbone = False
head_camera_temporal_train_cfg.use_lora = True
head_camera_temporal_train_cfg.lora_rank = 128
head_camera_temporal_train_cfg.lora_alpha = 64.0
head_camera_temporal_train_cfg.lora_dropout = 0.05
# Single-task smoke tests default to main-process loading to avoid /dev/shm pressure.
head_camera_temporal_train_cfg.single_task_load_worker = 0

# First sparse-guidance experiment: one task, six AWE-derived SE(3) blocks.
head_camera_temporal_train_cfg.dataset_path = "/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data_sparse_awe/adjust_bottle/demo_randomized/lerobot"
head_camera_temporal_train_cfg.single_task = "adjust_bottle"
head_camera_temporal_train_cfg.enable_se3_guidance = True
head_camera_temporal_train_cfg.use_sparse_waypoint_guidance = True
head_camera_temporal_train_cfg.enable_ee_target_module = False
head_camera_temporal_train_cfg.enable_ee_target_loss = False
head_camera_temporal_train_cfg.enable_ee_target_head_eval = False
head_camera_temporal_train_cfg.use_ee_target_as_rdt_condition = True
head_camera_temporal_train_cfg.rdt_condition_tokens.use_ee_target_tokens = True
head_camera_temporal_train_cfg.rdt_ee_target_condition_source = "oracle"
head_camera_temporal_train_cfg.sparse_waypoint_max_count = 6
head_camera_temporal_train_cfg.sparse_waypoint_max_duration = 32

# Stage-1 action-only objective; EE Target Head is reserved for Stage 2.
head_camera_temporal_train_cfg.loss_weight_ee_target = 0.0
head_camera_temporal_train_cfg.ee_target_condition_gt_mix_enabled = False
head_camera_temporal_train_cfg.ee_target_condition_gt_prob = 0.0
head_camera_temporal_train_cfg.single_task_load_worker = 0
head_camera_temporal_train_cfg.num_steps = 36000
head_camera_temporal_train_cfg.batch_size = 1
head_camera_temporal_train_cfg.se3_guidance_lr_initial = 1e-4
head_camera_temporal_train_cfg.se3_guidance_lr_max = 1e-4
head_camera_temporal_train_cfg.se3_guidance_lr_ramp_steps = 1
head_camera_temporal_train_cfg.se3_guidance_tokens_per_block = 7
head_camera_temporal_train_cfg.se3_guidance_rotation_encoding = "relative_quaternion"
head_camera_temporal_train_cfg.se3_guidance_duration_encoding = "normalized_scalar"
head_camera_temporal_train_cfg.se3_guidance_lr_max = 5e-5
head_camera_temporal_train_cfg.se3_guidance_lr_ramp_steps = 5000
head_camera_temporal_train_cfg.gradient_accumulation_steps = 1
