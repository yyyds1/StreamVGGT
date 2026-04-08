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
va_robotwin_cfg.guidance_scale = 5
va_robotwin_cfg.action_guidance_scale = 1

va_robotwin_cfg.num_inference_steps = 25
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = 50

# Shared by both training and online evaluation (va_server.py + train_va.py)
va_robotwin_cfg.multi_view_image_mode = 'vertical'
va_robotwin_cfg.image_height = 224
va_robotwin_cfg.image_width = 224
va_robotwin_cfg.window_size = 4
va_robotwin_cfg.chunk_size = 24
va_robotwin_cfg.image_frame_stride = 8

# Separate checkpoint controls
# Priority per model in train_va.py:
# 1) *_resume_from, 2) *_pretrained, 3) random init
va_robotwin_cfg.transformer_resume = True
va_robotwin_cfg.transformer_resume_from = None
va_robotwin_cfg.transformer_pretrained = '/mnt/nas/share/home/yds/actionvggt.pth'

va_robotwin_cfg.action_head_resume = True
va_robotwin_cfg.action_head_resume_from = None
va_robotwin_cfg.action_head_pretrained = '/mnt/nas/share/home/yds/RDT.pth'

va_robotwin_cfg.gradient_checkpointing = False
va_robotwin_cfg.long_context = False

# RDT settings (from RDT2/configs/rdt/post_train.yaml)
va_robotwin_cfg.rdt = EasyDict()
va_robotwin_cfg.rdt.hidden_size = 1024
va_robotwin_cfg.rdt.depth = 14
va_robotwin_cfg.rdt.num_heads = 8
va_robotwin_cfg.rdt.num_register_tokens = 4
va_robotwin_cfg.rdt.norm_eps = 1e-5
va_robotwin_cfg.rdt.multiple_of = 256
va_robotwin_cfg.rdt.ffn_dim_multiplier = None
va_robotwin_cfg.rdt.num_kv_heads = 4
va_robotwin_cfg.rdt.use_flash_attn = True
va_robotwin_cfg.rdt.action_dim = va_robotwin_cfg.action_dim

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
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
