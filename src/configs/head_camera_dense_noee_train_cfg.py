"""Head-camera temporal action-only control using the dense pre-sparse dataset."""
from copy import deepcopy

from .head_camera_temporal_train_cfg import head_camera_temporal_train_cfg

head_camera_dense_noee_train_cfg = deepcopy(head_camera_temporal_train_cfg)
head_camera_dense_noee_train_cfg.__name__ = "Config: head camera dense no EE train"
head_camera_dense_noee_train_cfg.dataset_path = "/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data_future_target"
head_camera_dense_noee_train_cfg.enable_se3_guidance = False
head_camera_dense_noee_train_cfg.use_sparse_waypoint_guidance = False
head_camera_dense_noee_train_cfg.enable_ee_target_module = False
head_camera_dense_noee_train_cfg.enable_ee_target_loss = False
head_camera_dense_noee_train_cfg.enable_ee_target_head_eval = False
head_camera_dense_noee_train_cfg.use_ee_target_as_rdt_condition = False
head_camera_dense_noee_train_cfg.rdt_condition_tokens.use_ee_target_tokens = False
head_camera_dense_noee_train_cfg.rdt_condition_tokens.ee_target_layers = []
head_camera_dense_noee_train_cfg.loss_weight_ee_target = 0.0
head_camera_dense_noee_train_cfg.ee_target_condition_gt_mix_enabled = False
head_camera_dense_noee_train_cfg.ee_target_condition_gt_prob = 0.0
head_camera_dense_noee_train_cfg.rdt_ee_target_condition_source = "predicted"
