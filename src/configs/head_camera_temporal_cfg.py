"""Single-head-camera, fixed temporal-window evaluation configuration."""
from copy import deepcopy

from .va_vga_robotwin_cfg import va_vga_robotwin_cfg


head_camera_temporal_cfg = deepcopy(va_vga_robotwin_cfg)
head_camera_temporal_cfg.__name__ = "Config: head camera temporal"
head_camera_temporal_cfg.model_arch = "vga"
head_camera_temporal_cfg.dataset_type = "vga_robotwin"
head_camera_temporal_cfg.obs_cam_keys = ["head_camera"]
head_camera_temporal_cfg.current_obs_cam_keys = ["head_camera"]
head_camera_temporal_cfg.history_obs_cam_keys = ["head_camera"]
head_camera_temporal_cfg.view_position_cam_keys = ["head_camera"]
head_camera_temporal_cfg.separate_history_current_obs_views = True
head_camera_temporal_cfg.multi_view_image_mode = "vertical"
head_camera_temporal_cfg.temporal_window = 8
head_camera_temporal_cfg.history_len = 8
head_camera_temporal_cfg.temporal_stride = 1
head_camera_temporal_cfg.history_frame_stride = 1
head_camera_temporal_cfg.image_frame_stride = 1
# Stage 1 uses this switch; stage 2 scripts override it explicitly.
head_camera_temporal_cfg.freeze_visual_backbone = False
head_camera_temporal_cfg.use_lora = True
# Single-task smoke tests default to main-process loading to avoid /dev/shm pressure.
head_camera_temporal_cfg.single_task_load_worker = 0

# Structured waypoint conditions replace EE-target-head features for RDT.
head_camera_temporal_cfg.enable_se3_guidance = True
head_camera_temporal_cfg.se3_guidance_source = "oracle"
head_camera_temporal_cfg.se3_guidance_noisy_oracle = False
head_camera_temporal_cfg.sparse_waypoint_max_count = 6
head_camera_temporal_cfg.sparse_waypoint_max_duration = 32
head_camera_temporal_cfg.rdt_condition_tokens.use_ee_target_tokens = True
head_camera_temporal_cfg.use_ee_target_as_rdt_condition = True
