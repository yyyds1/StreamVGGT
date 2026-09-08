from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


va_vga_robotwin_cfg = EasyDict(__name__="Config: VGA robotwin")
va_vga_robotwin_cfg.update(va_robotwin_cfg)

va_vga_robotwin_cfg.model_arch = "vga"
va_vga_robotwin_cfg.dataset_type = "vga_robotwin"
va_vga_robotwin_cfg.obs_cam_keys = [
    # "front_camera",
    "head_camera",
    "left_camera",
    "right_camera",
    "side_camera",
]
va_vga_robotwin_cfg.current_obs_cam_keys = list(va_vga_robotwin_cfg.obs_cam_keys)
va_vga_robotwin_cfg.history_obs_cam_keys = [
    cam for cam in va_vga_robotwin_cfg.obs_cam_keys if cam not in {"left_camera", "right_camera"}
]
va_vga_robotwin_cfg.view_position_cam_keys = list(va_vga_robotwin_cfg.obs_cam_keys)
va_vga_robotwin_cfg.separate_history_current_obs_views = True
va_vga_robotwin_cfg.use_expert_marked_rgb = True
# RDT action target source. "endpose" trains the action head on dense robot EE
# states from the dataset; "expert_target" trains it on high-level planner targets.
va_vga_robotwin_cfg.rdt_action_target_source = "endpose"
va_vga_robotwin_cfg.streamvggt_pretrained = "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/ckpt/checkpoints.pth"
va_vga_robotwin_cfg.text_tokenizer_name = "gemma"
va_vga_robotwin_cfg.text_model_name_or_path = "google/embeddinggemma-300M"
va_vga_robotwin_cfg.max_position_embeddings = 128
va_vga_robotwin_cfg.text_embedding_shape = [1, 768]
va_vga_robotwin_cfg.text_embed_dim = 768
va_vga_robotwin_cfg.preload_text_embedder_eval = False
va_vga_robotwin_cfg.disable_eval_text_embedder = True
va_vga_robotwin_cfg.text_embedder_warmup_prompt = "warmup"
va_vga_robotwin_cfg.use_language_condition = False

# The image/action token layout changed to stride=1; do not resume stride=4 checkpoints.
va_vga_robotwin_cfg.transformer_resume = False
va_vga_robotwin_cfg.transformer_resume_from = '/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/src/train_out/train_log_20260603_140850/ckpt/checkpoint_step_68000/transformer/diffusion_pytorch_model.safetensors'
va_vga_robotwin_cfg.transformer_pretrained = None
va_vga_robotwin_cfg.action_head_resume = False
va_vga_robotwin_cfg.action_head_resume_from = '/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/src/train_out/train_log_20260603_140850/ckpt/checkpoint_step_68000/action_head/diffusion_pytorch_model.safetensors'
va_vga_robotwin_cfg.action_head_pretrained = None

# LoRA settings for the pretrained VGA backbone.
va_vga_robotwin_cfg.use_lora = False
va_vga_robotwin_cfg.lora_rank = 128
va_vga_robotwin_cfg.lora_alpha = 64.0
va_vga_robotwin_cfg.lora_dropout = 0.05
va_vga_robotwin_cfg.lora_target_modules = ("qkv", "proj", "fc1", "fc2")

# VGA heads: enabled during train, bypassed during eval for efficiency.
va_vga_robotwin_cfg.enable_geometry_heads_train = False
va_vga_robotwin_cfg.enable_geometry_heads_eval = False
va_vga_robotwin_cfg.enable_ee_target_head_eval = True
# Convenience switch for the whole ee-target subsystem. Set this to False in a
# derived config to train/eval an action-head-only policy.
va_vga_robotwin_cfg.enable_ee_target_module = True
# Evaluation source for RDT EE-target condition tokens. Keep "predicted" for
# fair policy evaluation; "oracle" uses the live RoboTwin expert cursor to
# construct current-state-to-next-target waypoint transitions.
va_vga_robotwin_cfg.rdt_ee_target_condition_source = "oracle"

# Loss toggles (camera/depth code is kept and can be re-enabled anytime).
va_vga_robotwin_cfg.enable_camera_loss = False
va_vga_robotwin_cfg.enable_depth_loss = False
va_vga_robotwin_cfg.enable_ee_target_loss = True

# Default to action-loss-only training on RobotWin.
va_vga_robotwin_cfg.loss_weight_camera = 0.0
va_vga_robotwin_cfg.loss_weight_depth = 0.0
va_vga_robotwin_cfg.loss_weight_action = 0.4
va_vga_robotwin_cfg.loss_weight_ee_target = 1.0
va_vga_robotwin_cfg.loss_weight_ee_target_xyz = 1.0
va_vga_robotwin_cfg.loss_weight_ee_target_gripper = 0.5
va_vga_robotwin_cfg.ee_target_sequence_len = 6
va_vga_robotwin_cfg.ee_target_head_num_heads = 8
va_vga_robotwin_cfg.ee_target_head_trunk_depth = 4
va_vga_robotwin_cfg.ee_target_head_num_iterations = 4
va_vga_robotwin_cfg.ee_target_head_use_image_tokens = True
va_vga_robotwin_cfg.ee_target_head_image_cross_attn_depth = 1
va_vga_robotwin_cfg.rdt_state_token_loss_weight = 0.1

# During training, optionally mix GT expert targets into the RDT ee-target
# condition. The GT probability follows the recent ee-target loss EMA:
#   p = min + (max - min) * clamp((ema_loss - low) / (high - low), 0, 1)
# High ee-target loss therefore uses more GT condition; low loss uses the
# predicted ee-target head more often.
va_vga_robotwin_cfg.ee_target_condition_gt_mix_enabled = True
# Set to a float in [0, 1] to directly control the probability of replacing
# predicted EE transition condition tokens with GT tokens before RDT. If None,
# the adaptive EMA schedule below is used.
va_vga_robotwin_cfg.ee_target_condition_gt_prob = 1.0
va_vga_robotwin_cfg.ee_target_condition_gt_prob_min = 0.0
va_vga_robotwin_cfg.ee_target_condition_gt_prob_max = 0.8
va_vga_robotwin_cfg.ee_target_condition_loss_low = 0.005
va_vga_robotwin_cfg.ee_target_condition_loss_high = 0.05
va_vga_robotwin_cfg.ee_target_condition_loss_ema_decay = 0.98
va_vga_robotwin_cfg.ee_target_condition_initial_loss = 0.05

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
va_vga_robotwin_cfg.rdt_condition_tokens.use_ee_target_tokens = True
va_vga_robotwin_cfg.rdt_condition_tokens.layer_mode = "selected"
va_vga_robotwin_cfg.rdt_condition_tokens.image_layers = [2, 5, 8, 11]
va_vga_robotwin_cfg.rdt_condition_tokens.action_layers = [2, 5, 8, 11]
va_vga_robotwin_cfg.rdt_condition_tokens.ee_target_layers = [2, 5, 8, 11]
va_vga_robotwin_cfg.use_ee_target_as_rdt_condition = True
# With selected image/action layers, RDT alternates one image layer then one
# action/ee-target layer, so the effective RDT depth is the sum of selected
# condition stream layers.
va_vga_robotwin_cfg.rdt.depth = (
    len(va_vga_robotwin_cfg.rdt_condition_tokens.image_layers)
    + len(va_vga_robotwin_cfg.rdt_condition_tokens.action_layers)
    + len(va_vga_robotwin_cfg.rdt_condition_tokens.ee_target_layers)
)

# Direct RDT language conditioning from the encoded instruction embedding.
# This is separate from VGA language tokens above: when enabled, the text-model
# embedding is projected inside the RDT action head and used as a cross-attention
# condition during action denoising.
va_vga_robotwin_cfg.rdt_use_language_condition = False
