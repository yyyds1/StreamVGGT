# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)

va_robotwin_cfg.wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"

va_robotwin_cfg.attn_window = 72
va_robotwin_cfg.frame_chunk_size = 2
va_robotwin_cfg.env_type = 'robotwin_tshape'

va_robotwin_cfg.height = 256
va_robotwin_cfg.width = 320
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = ['observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist']
# va_robotwin_cfg.obs_cam_keys = ['observation.images.cam_high']
va_robotwin_cfg.guidance_scale = 5
va_robotwin_cfg.action_guidance_scale = 1

va_robotwin_cfg.num_inference_steps = 25
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = 100
# Number of actions to cache from each predicted chunk before the server infers again.
va_robotwin_cfg.action_chunk_exec_steps = 8

# Debug mode: restrict data usage to a single trajectory (episode).
va_robotwin_cfg.single_trajectory = False
# Optional explicit episode index. If None and single_trajectory=True, use first available episode.
va_robotwin_cfg.single_trajectory_episode_index = None
# Optional repo pin to disambiguate the same episode index across multiple RobotWin repos.
va_robotwin_cfg.single_trajectory_repo_id = None

# Shared by both training and online evaluation (va_server.py + train_va.py)
va_robotwin_cfg.multi_view_image_mode = 'vertical'
va_robotwin_cfg.image_height = 224
va_robotwin_cfg.image_width = 224
va_robotwin_cfg.chunk_size = 8
# Keep image/action time aligned: one decoded image frame corresponds to one action token.
va_robotwin_cfg.image_frame_stride = 1
va_robotwin_cfg.history_len = 4
va_robotwin_cfg.history_frame_stride = 1
va_robotwin_cfg.actionvggt_depth = 12
# Robot action/state representation used by both training and evaluation.
# "relative": poses are represented relative to the history-window anchor pose.
# "absolute": poses are represented in the simulator/world EE pose frame.
va_robotwin_cfg.action_representation = "absolute"
# "ee": end-effector pose/gripper action space. "joint": 14D joint action space.
va_robotwin_cfg.robotwin_action_space = "joint"
# Joint action representation when robotwin_action_space == "joint".
va_robotwin_cfg.joint_action_representation = "absolute"

# Separate checkpoint controls
# Priority per model in train_va.py:
# 1) *_resume_from, 2) *_pretrained, 3) random init
va_robotwin_cfg.transformer_resume = True
va_robotwin_cfg.transformer_resume_from = '/home/yds/code/StreamVGGT/src/train_out/checkpoint_step_35000/transformer/diffusion_pytorch_model.safetensors'
va_robotwin_cfg.transformer_pretrained = '/mnt/nas/share/home/yds/actionvggt.pth'

va_robotwin_cfg.action_head_resume = True
va_robotwin_cfg.action_head_resume_from = '/home/yds/code/StreamVGGT/src/train_out/checkpoint_step_35000/action_head/diffusion_pytorch_model.safetensors'
va_robotwin_cfg.action_head_pretrained = '/mnt/nas/share/home/yds/RDT.pth'

va_robotwin_cfg.gradient_checkpointing = False
va_robotwin_cfg.long_context = False

# RDT settings (from RDT2/configs/rdt/post_train.yaml)
va_robotwin_cfg.rdt = EasyDict()
va_robotwin_cfg.rdt.hidden_size = 1024
va_robotwin_cfg.rdt.depth = 7
va_robotwin_cfg.rdt.num_heads = 8
va_robotwin_cfg.rdt.num_register_tokens = 4
va_robotwin_cfg.rdt.norm_eps = 1e-5
va_robotwin_cfg.rdt.multiple_of = 256
va_robotwin_cfg.rdt.ffn_dim_multiplier = None
va_robotwin_cfg.rdt.num_kv_heads = 4
va_robotwin_cfg.rdt.use_flash_attn = True
va_robotwin_cfg.rdt.action_dim = va_robotwin_cfg.action_dim
va_robotwin_cfg.rdt.num_train_timesteps = 1000
va_robotwin_cfg.rdt.num_inference_steps = 100
va_robotwin_cfg.rdt.flow_match_shift = 3.0
va_robotwin_cfg.rdt.sigma_max = 1.0
va_robotwin_cfg.rdt.sigma_min = 0.003 / 1.002
va_robotwin_cfg.rdt.extra_one_step = True
va_robotwin_cfg.rdt.action_condition_noise_std = 0.01
va_robotwin_cfg.rdt.warm_start_blend = 1.0
va_robotwin_cfg.rdt.warm_start_noise_std = 0.0
# Online warm start begins from a partially noised previous action chunk.
# This must match flow-matching training: x_t = (1 - sigma) * x0 + sigma * noise.
va_robotwin_cfg.rdt.warm_start_sigma = 0.5
va_robotwin_cfg.rdt.action_smoothing_alpha = 1.0

# Online EE safety guard for absolute actions. This clips commanded end-effector
# targets before sending them to RoboTwin, reducing IK singularity/overreach
# failures such as a fully straightened arm getting stuck.
va_robotwin_cfg.ee_target_guard = EasyDict()
va_robotwin_cfg.ee_target_guard.enabled = True
va_robotwin_cfg.ee_target_guard.max_delta_xyz = 0.2
# va_robotwin_cfg.ee_target_guard.left_xyz_min = [-0.2979237735, -0.3138048649, 0.0]
# va_robotwin_cfg.ee_target_guard.left_xyz_max = [-0.06342231482, -0.001932744752, 1.212610745]
# va_robotwin_cfg.ee_target_guard.right_xyz_min = [-0.04576408118, -0.3128012717, 0.0]
# va_robotwin_cfg.ee_target_guard.right_xyz_max = [0.3064331412, -0.006573319435, 1.212978864]
va_robotwin_cfg.ee_target_guard.left_xyz_min = [-10, -10, -10.0]
va_robotwin_cfg.ee_target_guard.left_xyz_max = [10.0, 10.0, 10.0]
va_robotwin_cfg.ee_target_guard.right_xyz_min = [-10, -10, -10.0]
va_robotwin_cfg.ee_target_guard.right_xyz_max = [10.0, 10.0, 10.0]

va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0

va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robotwin_cfg.action_norm_method = 'quantiles'
# va_robotwin_cfg.norm_stat = {
#     "q01": [
#         -0.2979237735, -0.3138048649, 0.8695474267,
#         -1, -1, -1, -1,
#         -0.04576408118, -0.3128012717, 0.8713479042,
#         -1, -1, -1, -1,
#     ] + [0.] * 16,
#     "q99": [
#         0.03409340233, -0.03222606331, 1.077983499,
#         1, 1, 1, 1,
#         0.3064331412, -0.04851070791, 1.098479986,
#         1, 1, 1, 1,
#     ] + [1.0] * 16,
# }
va_robotwin_cfg.norm_stats_by_action_mode = {
    "ee_absolute": {
        "q01": [-0.3551848233, -0.3138048649, 0.847497046, -1, -1, -1, -1, -0.04975826666, -0.3128008544, 0.8267407417, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "q99": [0.05339170247, 0.08732383698, 1.090680718, 1, 1, 1, 1, 0.3451347053, 0.08006774634, 1.125197411, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    "ee_relative": {
        "q01": [-0.05087432265, -0.06358402222, -0.06772831827, -1, -1, -1, -1, -0.05816078559, -0.04257848486, -0.06448298693, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "q99": [0.0644294098, 0.05274088308, 0.07557101548, 1, 1, 1, 1, 0.04570635408, 0.05656298995, 0.07303394377, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    "joint_absolute": {
        "q01": [-1.012992382, -1.326246002e-05, -3.193847078e-05, -1.60202086, -0.6709504724, -2.026447296, 0, -0.1464507282, -6.741475772e-06, -3.12039374e-05, -1.695368886, -1.02519083, -1.555285215, 0],
        "q99": [0.1563098133, 2.574292898, 2.483379841, 1.326853991, 1.243466854, 1.625086308, 1, 0.9878454804, 2.597167969, 2.463290215, 1.278123617, 0.8286941648, 2.188305855, 1],
    },
    "joint_delta": {
        "q01": [-0.04006520659, -0.08611942828, -0.07334285975, -0.06555784494, -0.0265740566, -0.06931841373, -0.07537689805, -0.02876866236, -0.07184169441, -0.06339568645, -0.06431164593, -0.03299400955, -0.05093454197, -0.07537688315],
        "q99": [0.03682650626, 0.1077729464, 0.09027726948, 0.06595182419, 0.03488710523, 0.05634796992, 0.07537688315, 0.03830033168, 0.1077221259, 0.09055935591, 0.0599084869, 0.02597944252, 0.06889820099, 0.07537688315],
    },
}
va_robotwin_cfg.norm_stat = va_robotwin_cfg.norm_stats_by_action_mode["ee_absolute"]
va_robotwin_cfg.joint_norm_stat = va_robotwin_cfg.norm_stats_by_action_mode["joint_absolute"]
