from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


va_vga_robotwin_cfg = EasyDict(__name__="Config: VGA robotwin")
va_vga_robotwin_cfg.update(va_robotwin_cfg)

va_vga_robotwin_cfg.model_arch = "vga"
va_vga_robotwin_cfg.dataset_type = "robotwin"
va_vga_robotwin_cfg.streamvggt_pretrained = "/home/yds/code/StreamVGGT/ckpt/checkpoints.pth"
va_vga_robotwin_cfg.text_tokenizer_name = "gemma"
va_vga_robotwin_cfg.text_model_name_or_path = "google/embeddinggemma-300M"
va_vga_robotwin_cfg.max_position_embeddings = 128
va_vga_robotwin_cfg.text_embedding_shape = [1, 768]
va_vga_robotwin_cfg.text_embed_dim = 768
va_vga_robotwin_cfg.preload_text_embedder_eval = True
va_vga_robotwin_cfg.text_embedder_warmup_prompt = "warmup"

# The image/action token layout changed to stride=1; do not resume stride=4 checkpoints.
va_vga_robotwin_cfg.transformer_resume = True
va_vga_robotwin_cfg.transformer_resume_from = '/home/yds/code/StreamVGGT/src/train_out/w_text/checkpoint_step_26000/transformer/diffusion_pytorch_model.safetensors'
va_vga_robotwin_cfg.transformer_pretrained = None
va_vga_robotwin_cfg.action_head_resume = True
va_vga_robotwin_cfg.action_head_resume_from = '/home/yds/code/StreamVGGT/src/train_out/w_text/checkpoint_step_26000/action_head/diffusion_pytorch_model.safetensors'
va_vga_robotwin_cfg.action_head_pretrained = None

# LoRA settings for the pretrained VGA backbone.
va_vga_robotwin_cfg.use_lora = True
va_vga_robotwin_cfg.lora_rank = 64
va_vga_robotwin_cfg.lora_alpha = 16.0
va_vga_robotwin_cfg.lora_dropout = 0.05
va_vga_robotwin_cfg.lora_target_modules = ("qkv", "proj", "fc1", "fc2")

# VGA heads: enabled during train, bypassed during eval for efficiency.
va_vga_robotwin_cfg.enable_geometry_heads_train = False
va_vga_robotwin_cfg.enable_geometry_heads_eval = False

# Loss toggles (camera/depth code is kept and can be re-enabled anytime).
va_vga_robotwin_cfg.enable_camera_loss = False
va_vga_robotwin_cfg.enable_depth_loss = False

# Default to action-loss-only training on RobotWin.
va_vga_robotwin_cfg.loss_weight_camera = 0.0
va_vga_robotwin_cfg.loss_weight_depth = 0.0
va_vga_robotwin_cfg.loss_weight_action = 1.0
va_vga_robotwin_cfg.rdt_state_token_loss_weight = 0.1

# Depth loss hyper-parameters.
va_vga_robotwin_cfg.depth_loss_grad_weight = 0.1

# p2p+RDT uses the clean first action/state token directly.
va_vga_robotwin_cfg.state_noise_std = 0.0
va_vga_robotwin_cfg.state_noise_clip = True
# Noise augmentation on robot state/action tokens before VGA encoding.
# This improves robustness to inference-time state estimation/prediction jitter.
va_vga_robotwin_cfg.vga_action_state_noise_std = 0.01
va_vga_robotwin_cfg.vga_action_state_noise_clip = True
# State condition strategy for RDT: "first_action", "episode_initial", "latest", or "null".
# p2p+RDT uses the first clean action/state token as the state condition.
va_vga_robotwin_cfg.state_condition_mode = "first_action"

# RDT condition token composition from VGA backbone outputs.
va_vga_robotwin_cfg.rdt_condition_tokens = EasyDict()
va_vga_robotwin_cfg.rdt_condition_tokens.use_action_queries = True
va_vga_robotwin_cfg.rdt_condition_tokens.use_image_tokens = True
va_vga_robotwin_cfg.rdt_condition_tokens.use_language_tokens = False

# Direct RDT language conditioning from the encoded instruction embedding.
# This is separate from VGA language tokens above: when enabled, the text-model
# embedding is projected inside the RDT action head and used as a cross-attention
# condition during action denoising.
va_vga_robotwin_cfg.rdt_use_language_condition = True
