# Copyright 2024-2026 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_train_rdt_cfg = EasyDict(__name__='Config: VA robotwin train + RDT')
va_robotwin_train_rdt_cfg.update(va_robotwin_train_cfg)

# -----------------------------------------------------------------------------
# Separate checkpoint controls
# Priority per model in train_va.py:
# 1) *_resume_from, 2) *_pretrained, 3) random init
# -----------------------------------------------------------------------------
# ActionVGGT encoder checkpoints
va_robotwin_train_rdt_cfg.transformer_resume_from = None
va_robotwin_train_rdt_cfg.transformer_pretrained = '/mnt/nas/share/home/yds/actionvggt.pth'

# RDT action head checkpoints
va_robotwin_train_rdt_cfg.action_head_resume_from = None
va_robotwin_train_rdt_cfg.action_head_pretrained = '/mnt/nas/share/home/yds/RDT.pth'

va_robotwin_train_rdt_cfg.image_height = 224
va_robotwin_train_rdt_cfg.image_width = 224
va_robotwin_train_rdt_cfg.window_size = 4
va_robotwin_train_rdt_cfg.chunk_size = 24
va_robotwin_train_rdt_cfg.image_frame_stride = 8
# va_robotwin_train_rdt_cfg.multi_view_image_mode = 'vertical'    #['vertical', 'frame', 'first']
# va_robotwin_train_rdt_cfg.rdt_img_cond_mode = 'pool'
# va_robotwin_train_rdt_cfg.rdt_img_pool_size = 4
# va_robotwin_train_rdt_cfg.rdt_img_keep_summary_tokens = False

va_robotwin_train_rdt_cfg.gradient_checkpointing = False
va_robotwin_train_rdt_cfg.long_context = False

# -----------------------------------------------------------------------------
# RDT settings (from RDT2/configs/rdt/post_train.yaml)
# Note: train_va.py currently auto-builds some ActionVGGT-coupled dimensions
# (e.g., hidden_size/depth from ActionVGGT). These fields are provided here for
# training config completeness and future wiring.
# -----------------------------------------------------------------------------
va_robotwin_train_rdt_cfg.rdt = EasyDict()
va_robotwin_train_rdt_cfg.rdt.hidden_size = 1024
va_robotwin_train_rdt_cfg.rdt.depth = 14
va_robotwin_train_rdt_cfg.rdt.num_heads = 8
va_robotwin_train_rdt_cfg.rdt.num_register_tokens = 4
va_robotwin_train_rdt_cfg.rdt.norm_eps = 1e-5
va_robotwin_train_rdt_cfg.rdt.multiple_of = 256
va_robotwin_train_rdt_cfg.rdt.ffn_dim_multiplier = None
va_robotwin_train_rdt_cfg.rdt.num_kv_heads = 4
va_robotwin_train_rdt_cfg.rdt.use_flash_attn = True
va_robotwin_train_rdt_cfg.rdt.action_dim = va_robotwin_train_rdt_cfg.action_dim
