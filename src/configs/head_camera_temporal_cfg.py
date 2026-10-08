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
