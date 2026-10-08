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
