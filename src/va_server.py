# Copyright 2024-2026 The Robbyant Team Authors. All rights reserved.
import argparse
import json
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file
from safetensors import safe_open
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from actionvggt.models.actionvggt import ActionVGGT
from vga.models.vga import VGA
from configs import VA_CONFIGS
from rdt.model import RDT
from utils import (
    FlowMatchScheduler,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
)
from utils.text_embedding import encode_prompt, get_text_embedder


def get_effective_num_image_views(config):
    mode = getattr(config, "multi_view_image_mode", "vertical")
    if mode == "vertical":
        return len(config.obs_cam_keys)
    if mode in {"frame", "first"}:
        return 1
    raise ValueError(f"Unsupported multi_view_image_mode `{mode}`")


def _extract_modulelist_indices_from_state(state, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = set()
    for key in state.keys():
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _build_even_layer_map(src_indices, target_depth):
    if target_depth <= 0:
        raise ValueError(f"target_depth must be > 0, got {target_depth}")
    if len(src_indices) == 0:
        return {}

    src_count = len(src_indices)
    sampled_positions = [
        min((i * src_count) // target_depth, src_count - 1)
        for i in range(target_depth)
    ]
    sampled_src_indices = [src_indices[pos] for pos in sampled_positions]
    return {src_idx: dst_idx for dst_idx, src_idx in enumerate(sampled_src_indices)}


def _remap_modulelist_state_dict(state, prefix, target_depth):
    src_indices = _extract_modulelist_indices_from_state(state, prefix)
    if len(src_indices) == 0:
        return state, None

    if len(src_indices) == target_depth and src_indices == list(range(target_depth)):
        return state, src_indices

    src_to_dst = _build_even_layer_map(src_indices, target_depth)
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)(\..+)$")

    remapped = {}
    for key, value in state.items():
        match = pattern.match(key)
        if not match:
            remapped[key] = value
            continue

        src_idx = int(match.group(1))
        suffix = match.group(2)
        if src_idx in src_to_dst:
            dst_idx = src_to_dst[src_idx]
            remapped[f"{prefix}.{dst_idx}{suffix}"] = value

    sampled_src_indices = [src_idx for src_idx, _ in sorted(src_to_dst.items(), key=lambda x: x[1])]
    return remapped, sampled_src_indices


def _adapt_transformer_state_for_depth(state, target_depth):
    adapted = dict(state)
    prefixes = [
        "aggregator.frame_blocks",
        "aggregator.frame_blocks_image",
        "aggregator.frame_blocks_action",
        "aggregator.global_blocks",
        "aggregator.cross_blocks",
    ]
    for prefix in prefixes:
        adapted, sampled = _remap_modulelist_state_dict(adapted, prefix, target_depth)
        if sampled is not None and len(sampled) > 0:
            logger.info(
                f"Layer remap for {prefix}: sampled pretrained layers {sampled} -> target depth {target_depth}"
            )
    return adapted


def _adapt_rdt_state_for_depth(state, target_depth):
    adapted, sampled = _remap_modulelist_state_dict(state, "blocks", target_depth)
    if sampled is not None and len(sampled) > 0:
        logger.info(
            f"Layer remap for RDT blocks: sampled pretrained layers {sampled} -> target depth {target_depth}"
        )
    return adapted


CHECKPOINT_SUCCESS_MARKER = "_SUCCESS"


def _is_valid_safetensors(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            _ = list(f.keys())
        return True
    except Exception:
        return False


def _resize_1d_positional_embedding(src_emb, dst_emb):
    if src_emb.ndim != 3 or dst_emb.ndim != 3:
        return None
    if src_emb.shape[0] != dst_emb.shape[0] or src_emb.shape[2] != dst_emb.shape[2]:
        return None
    if src_emb.shape[1] == dst_emb.shape[1]:
        return src_emb

    src = src_emb.transpose(1, 2)
    resized = F.interpolate(src, size=dst_emb.shape[1], mode="linear", align_corners=False)
    return resized.transpose(1, 2)


def _adapt_rdt_state_for_model(state, model):
    adapted = _adapt_rdt_state_for_depth(state, target_depth=model.depth)
    model_state = model.state_dict()

    for pos_key in ["x_pos_emb", "img_pos_emb", "act_pos_emb"]:
        if pos_key not in adapted or pos_key not in model_state:
            continue
        src_pos = adapted[pos_key]
        dst_pos = model_state[pos_key]
        if src_pos.shape == dst_pos.shape:
            continue

        resized = _resize_1d_positional_embedding(src_pos, dst_pos)
        if resized is None:
            logger.warning(
                f"Skip loading {pos_key}: checkpoint shape {tuple(src_pos.shape)} != "
                f"model shape {tuple(dst_pos.shape)} and resize is not applicable"
            )
            adapted.pop(pos_key, None)
            continue

        adapted[pos_key] = resized.to(dtype=dst_pos.dtype, device=dst_pos.device)
        logger.info(
            f"Resized RDT {pos_key} from {tuple(src_pos.shape)} to {tuple(dst_pos.shape)}"
        )

    return adapted


class VA_Server:
    def __init__(self, job_config):
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.ckpt_root = job_config.ckpt_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")

        self.num_input_frames = 1
        self.history_len = max(1, int(getattr(job_config, "history_len", 1)))
        self.history_frame_stride = max(1, int(getattr(job_config, "history_frame_stride", 1)))
        self.chunk_size = int(getattr(job_config, "chunk_size", 24))
        self.rdt_horizon = self.chunk_size + 1
        self.action_chunk_exec_steps = max(1, int(getattr(job_config, "action_chunk_exec_steps", 1)))
        self.image_frame_stride = int(getattr(job_config, "image_frame_stride", 8))
        self.action_dim = int(job_config.action_dim)
        self.patch_size = tuple(getattr(job_config, "patch_size", (1, 14, 14)))
        self.multi_view_image_mode = getattr(job_config, "multi_view_image_mode", "vertical")
        self.model_arch = str(getattr(job_config, "model_arch", "actionvggt")).lower()
        self.rdt_use_language_condition = bool(getattr(job_config, "rdt_use_language_condition", False))
        self.action_representation = str(getattr(job_config, "action_representation", "relative")).lower()
        if self.action_representation not in {"relative", "absolute"}:
            raise ValueError(
                f"Unsupported action_representation `{self.action_representation}`. "
                "Expected 'relative' or 'absolute'."
            )
        self.state_condition_mode = str(getattr(job_config, "state_condition_mode", "latest")).lower()
        if self.state_condition_mode not in {"first_action", "latest", "episode_initial", "null"}:
            raise ValueError(
                f"Unsupported state_condition_mode `{self.state_condition_mode}`. "
                "Expected one of {'first_action', 'latest', 'episode_initial', 'null'}."
            )

        self.image_height = int(getattr(job_config, "image_height", job_config.height))
        self.image_width = int(getattr(job_config, "image_width", job_config.width))

        # If chunk size is not divisible by stride, keep one frame with full chunk tokens.
        if self.rdt_horizon % self.image_frame_stride == 0:
            self.pred_frames = self.rdt_horizon // self.image_frame_stride
            self.tokens_per_frame = self.image_frame_stride
        else:
            self.pred_frames = 1
            self.tokens_per_frame = self.rdt_horizon

        rdt_cfg = self.job_config.rdt
        self.train_scheduler_action = FlowMatchScheduler(
            num_inference_steps=int(getattr(rdt_cfg, "num_inference_steps", 100)),
            num_train_timesteps=int(getattr(rdt_cfg, "num_train_timesteps", 1000)),
            shift=float(getattr(rdt_cfg, "flow_match_shift", 3.0)),
            sigma_max=float(getattr(rdt_cfg, "sigma_max", 1.0)),
            sigma_min=float(getattr(rdt_cfg, "sigma_min", 0.003 / 1.002)),
            extra_one_step=bool(getattr(rdt_cfg, "extra_one_step", True)),
        )
        action_steps = int(getattr(rdt_cfg, "num_inference_steps", getattr(self.job_config, "action_num_inference_steps", 100)))
        self.train_scheduler_action.set_timesteps(action_steps)
        self.warm_start_blend = float(max(0.0, min(1.0, getattr(rdt_cfg, "warm_start_blend", 0.85))))
        self.warm_start_noise_std = float(max(0.0, getattr(rdt_cfg, "warm_start_noise_std", 0.03)))
        self.warm_start_sigma = float(max(0.0, min(1.0, getattr(rdt_cfg, "warm_start_sigma", 0.5))))
        self.action_smoothing_alpha = float(max(0.0, min(1.0, getattr(rdt_cfg, "action_smoothing_alpha", 0.35))))
        self.unexecuted_action_buffer = deque()
        self.actions_served_since_last_prediction = 0
        guard_cfg = getattr(job_config, "ee_target_guard", None)
        self.ee_target_guard_enabled = bool(getattr(guard_cfg, "enabled", False)) if guard_cfg is not None else False
        self.ee_target_guard_max_delta_xyz = float(getattr(guard_cfg, "max_delta_xyz", 0.0)) if guard_cfg is not None else 0.0
        self.ee_target_guard_left_min = (
            np.asarray(getattr(guard_cfg, "left_xyz_min", [-np.inf, -np.inf, -np.inf]), dtype=np.float32)
            if guard_cfg is not None
            else np.full(3, -np.inf, dtype=np.float32)
        )
        self.ee_target_guard_left_max = (
            np.asarray(getattr(guard_cfg, "left_xyz_max", [np.inf, np.inf, np.inf]), dtype=np.float32)
            if guard_cfg is not None
            else np.full(3, np.inf, dtype=np.float32)
        )
        self.ee_target_guard_right_min = (
            np.asarray(getattr(guard_cfg, "right_xyz_min", [-np.inf, -np.inf, -np.inf]), dtype=np.float32)
            if guard_cfg is not None
            else np.full(3, -np.inf, dtype=np.float32)
        )
        self.ee_target_guard_right_max = (
            np.asarray(getattr(guard_cfg, "right_xyz_max", [np.inf, np.inf, np.inf]), dtype=np.float32)
            if guard_cfg is not None
            else np.full(3, np.inf, dtype=np.float32)
        )

        common_kwargs = dict(
            img_height=self.image_height,
            img_width=self.image_width,
            num_image_views=get_effective_num_image_views(self.job_config),
            text_embed_dim=int(getattr(job_config, "text_embed_dim", 4096)),
            rdt_img_cond_mode=getattr(job_config, "rdt_img_cond_mode", "full"),
            rdt_img_pool_size=getattr(job_config, "rdt_img_pool_size", 1),
            rdt_img_keep_summary_tokens=getattr(job_config, "rdt_img_keep_summary_tokens", False),
            window_size=self.history_len,
            chunk_size=self.rdt_horizon,
            action_dim=self.action_dim,
            aggregator_depth=int(getattr(job_config, "actionvggt_depth", 24)),
            image_frame_stride=self.image_frame_stride,
        )
        if self.model_arch == "vga":
            self.transformer = VGA(
                enable_camera_depth_heads=bool(getattr(job_config, "enable_geometry_heads_eval", False)),
                **common_kwargs,
            )
            if getattr(job_config, "use_lora", False):
                lora_rank = int(getattr(job_config, "lora_rank", 8))
                lora_alpha = float(getattr(job_config, "lora_alpha", 16.0))
                lora_dropout = float(getattr(job_config, "lora_dropout", 0.05))
                lora_target_modules = tuple(getattr(job_config, "lora_target_modules", ("qkv", "proj", "fc1", "fc2")))
                self.transformer.enable_lora(
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    target_modules=lora_target_modules,
                )
        else:
            self.transformer = ActionVGGT(**common_kwargs)
        self.transformer.to(self.device)

        rdt_config = self.job_config.rdt
        effective_num_image_views = get_effective_num_image_views(self.job_config)
        patch_h = self.transformer.img_height // self.transformer.patch_size
        patch_w = self.transformer.img_width // self.transformer.patch_size
        pooled_patch_h = max(1, math.ceil(patch_h / self.transformer.rdt_img_pool_size))
        pooled_patch_w = max(1, math.ceil(patch_w / self.transformer.rdt_img_pool_size))
        if self.transformer.rdt_img_cond_mode == "pool":
            pooled_tokens_per_view = pooled_patch_h * pooled_patch_w
            img_tokens_per_frame = pooled_tokens_per_view * effective_num_image_views
            if self.transformer.rdt_img_keep_summary_tokens:
                img_tokens_per_frame += effective_num_image_views
                rdt_img_pos_emb_config = [("image", self.num_input_frames * img_tokens_per_frame)]
            else:
                rdt_img_pos_emb_config = [
                    ("image", (self.num_input_frames * effective_num_image_views, pooled_patch_h, pooled_patch_w))
                ]
        else:
            img_tokens_per_frame = patch_h * patch_w * effective_num_image_views
            rdt_img_pos_emb_config = [
                ("image", (self.num_input_frames * effective_num_image_views, patch_h, patch_w))
            ]

        rdt_horizon = self.rdt_horizon
        rdt_x_pos_emb_config = [("act", rdt_horizon + self.job_config.rdt.num_register_tokens)]
        num_rdt_act_frames = 1
        rdt_act_pos_emb_config = [("action", (num_rdt_act_frames, rdt_horizon))]

        self.action_head = RDT(
            horizon=rdt_horizon,
            output_size=self.action_dim,
            config=rdt_config,
            x_pos_emb_config=rdt_x_pos_emb_config,
            lang_pos_emb_config=None,
            max_lang_len=0,
            img_pos_emb_config=rdt_img_pos_emb_config,
            max_img_len=self.num_input_frames * img_tokens_per_frame,
            act_pos_emb_config=rdt_act_pos_emb_config,
            max_act_len=num_rdt_act_frames * rdt_horizon,
            text_embed_dim=int(getattr(job_config, "text_embed_dim", 4096)),
            dtype=self.dtype,
        )
        self.action_head.to(self.device)

        self._load_checkpoints()
        self.transformer.eval()
        self.action_head.eval()

        self.used_action_channel_ids = list(self.job_config.used_action_channel_ids)
        self.inverse_used_action_channel_ids = list(self.job_config.inverse_used_action_channel_ids)
        self.action_mask = torch.zeros([self.action_dim], dtype=torch.bool)
        self.action_mask[self.used_action_channel_ids] = True
        self.actions_q01 = torch.tensor(self.job_config.norm_stat["q01"], dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat["q99"], dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method
        self._text_emb_cache = {}
        self._text_emb_search_files = None

        self._preload_text_embedder()
        self._reset_runtime_buffers(prompt=None)

    def _normalize_prompt_text(self, prompt: Optional[str]) -> Optional[str]:
        if prompt is None:
            return None
        return str(prompt).strip()

    def _get_dataset_root(self) -> Optional[Path]:
        dataset_path = getattr(self.job_config, "dataset_path", None)
        if dataset_path:
            root = Path(dataset_path)
            if root.exists():
                return root
        fallback = Path(__file__).resolve().parent.parent / "dataset"
        if fallback.exists():
            return fallback
        return None

    def _build_text_emb_search_files(self):
        if self._text_emb_search_files is not None:
            return
        self._text_emb_search_files = []

        dataset_root = self._get_dataset_root()
        if dataset_root is None:
            logger.warning("No dataset root found for text embedding lookup.")
            return

        cam_key = self.job_config.obs_cam_keys[0]
        repo_id = getattr(self.job_config, "single_trajectory_repo_id", None)
        repo_name = Path(str(repo_id)).name if repo_id else None
        if bool(getattr(self.job_config, "single_trajectory", False)):
            episode_index = getattr(self.job_config, "single_trajectory_episode_index", None)
            if episode_index is None:
                logger.info(
                    "single_trajectory=True for eval, but single_trajectory_episode_index is None; "
                    "text embedding lookup will scan all episodes and cache first prompt match."
                )
                pattern = f"**/latents/chunk-*/{cam_key}/episode_*.pth"
            else:
                pattern = f"**/latents/chunk-*/{cam_key}/episode_{int(episode_index):06d}_*.pth"
                logger.info(
                    f"single_trajectory=True for eval; restricting text embedding search to episode_index={int(episode_index)}"
                )
            if repo_name:
                pattern = f"{repo_name}/latents/chunk-*/{cam_key}/episode_{int(episode_index):06d}_*.pth" if episode_index is not None else f"{repo_name}/latents/chunk-*/{cam_key}/episode_*.pth"
                logger.info(
                    f"single_trajectory=True for eval; restricting text embedding search to repo_id={repo_id}"
                )
        else:
            pattern = f"**/latents/chunk-*/{cam_key}/episode_*.pth" if not repo_name else f"{repo_name}/latents/chunk-*/{cam_key}/episode_*.pth"
        self._text_emb_search_files = sorted(dataset_root.glob(pattern))
        logger.info(
            f"Prepared text embedding search index with {len(self._text_emb_search_files)} latent files "
            f"from {dataset_root}"
        )

    def _resolve_text_emb_from_dataset(self, prompt: Optional[str]) -> Optional[torch.Tensor]:
        return self._encode_prompt_text_emb(prompt)

    def _preload_text_embedder(self) -> None:
        if not bool(getattr(self.job_config, "preload_text_embedder_eval", True)):
            logger.info("Skipping eval text embedder preload because preload_text_embedder_eval=False.")
            return

        tokenizer_name = str(getattr(self.job_config, "text_tokenizer_name", "") or "").strip()
        if not tokenizer_name:
            logger.info("No eval text tokenizer configured; skipping text embedder preload.")
            return

        model_name = str(
            getattr(self.job_config, "text_model_name_or_path", None)
            or getattr(self.job_config, "model_name_or_path", None)
            or "google/embeddinggemma-300M"
        )
        start_time = time.monotonic()
        logger.info(
            f"Preloading eval text embedder `{tokenizer_name}` from `{model_name}` before websocket startup..."
        )
        try:
            embedder = get_text_embedder(self.job_config)
            if embedder is None:
                logger.warning(
                    f"Text tokenizer `{tokenizer_name}` is not supported by get_text_embedder; "
                    "eval will run without server-side text embeddings."
                )
                return
            warmup_text = str(getattr(self.job_config, "text_embedder_warmup_prompt", "warmup"))
            warmup_emb = encode_prompt(
                warmup_text,
                self.job_config,
                device=self.device,
                dtype=self.dtype,
            )
            if warmup_emb is not None:
                self._text_emb_cache[warmup_text] = warmup_emb
                logger.info(
                    f"Eval text embedder warmup complete: shape={tuple(warmup_emb.shape)}, "
                    f"dtype={warmup_emb.dtype}, device={warmup_emb.device}"
                )
        except Exception:
            logger.exception(
                "Failed to preload eval text embedder before websocket startup. "
                "Fix the text model path/cache or set preload_text_embedder_eval=False to bypass."
            )
            raise
        finally:
            logger.info(f"Eval text embedder preload took {(time.monotonic() - start_time):.2f}s.")

    def _encode_prompt_text_emb(self, prompt: Optional[str]) -> Optional[torch.Tensor]:
        prompt_norm = self._normalize_prompt_text(prompt)
        if not prompt_norm:
            return None

        if prompt_norm in self._text_emb_cache:
            return self._text_emb_cache[prompt_norm]

        text_emb = encode_prompt(prompt_norm, self.job_config, device=self.device, dtype=self.dtype)
        if text_emb is None:
            logger.warning(
                f"No text tokenizer configured for prompt: {prompt_norm!r}. "
                "Eval will fall back to no-language conditioning."
            )
        else:
            logger.info(f"Encoded prompt with p2p-compatible text tokenizer: {prompt_norm!r}")
        self._text_emb_cache[prompt_norm] = text_emb
        return text_emb

    def _normalize_text_emb_payload(self, text_emb):
        if text_emb is None:
            return None
        if not torch.is_tensor(text_emb):
            text_emb = torch.as_tensor(text_emb)
        if text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(0)
        elif text_emb.ndim != 3:
            raise ValueError(f"text_emb must be [B, D] or [B, L, D], got shape {tuple(text_emb.shape)}")
        expected_dim = int(getattr(self.job_config, "text_embed_dim", text_emb.shape[-1]))
        if int(text_emb.shape[-1]) != expected_dim:
            logger.warning(
                f"Ignoring provided text_emb with dim {int(text_emb.shape[-1])}; "
                f"expected p2p-compatible dim {expected_dim}."
            )
            return None
        return text_emb.to(dtype=self.dtype, device=self.device)

    def _load_checkpoint_state(self, path):
        if str(path).endswith(".safetensors"):
            try:
                return load_file(path, device=str(self.device))
            except Exception as exc:
                raise RuntimeError(f"Failed to load safetensors checkpoint: {path}") from exc
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        return state

    def _is_complete_checkpoint_dir(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        marker = checkpoint_dir / CHECKPOINT_SUCCESS_MARKER
        transformer_path = checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        action_head_path = checkpoint_dir / "action_head" / "diffusion_pytorch_model.safetensors"
        return marker.exists() and _is_valid_safetensors(transformer_path) and _is_valid_safetensors(action_head_path)

    def _resize_pos_embed_tensor(self, src_pos_embed, dst_pos_embed):
        if src_pos_embed.ndim != 3 or dst_pos_embed.ndim != 3:
            return None
        if src_pos_embed.shape[-1] != dst_pos_embed.shape[-1]:
            return None

        src_len = src_pos_embed.shape[1]
        dst_len = dst_pos_embed.shape[1]
        if src_len == dst_len:
            return src_pos_embed

        src_prefix = 1
        dst_prefix = 1
        src_grid_tokens = src_len - src_prefix
        dst_grid_tokens = dst_len - dst_prefix
        src_hw = int(math.sqrt(src_grid_tokens))
        dst_hw = int(math.sqrt(dst_grid_tokens))

        if src_hw * src_hw != src_grid_tokens or dst_hw * dst_hw != dst_grid_tokens:
            return None

        src_prefix_tokens = src_pos_embed[:, :src_prefix]
        src_grid = src_pos_embed[:, src_prefix:]
        src_grid = src_grid.reshape(1, src_hw, src_hw, -1).permute(0, 3, 1, 2)
        src_grid = F.interpolate(src_grid, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
        src_grid = src_grid.permute(0, 2, 3, 1).reshape(1, dst_hw * dst_hw, -1)

        return torch.cat([src_prefix_tokens, src_grid], dim=1)

    def _adapt_transformer_state_for_resolution(self, model, state):
        if not isinstance(state, dict):
            return state

        adapted = dict(state)
        model_state = model.state_dict()
        pos_key = "aggregator.patch_embed.pos_embed"

        if pos_key in adapted and pos_key in model_state:
            src_pos = adapted[pos_key]
            dst_pos = model_state[pos_key]
            if src_pos.shape != dst_pos.shape:
                resized = self._resize_pos_embed_tensor(src_pos, dst_pos)
                if resized is None:
                    logger.warning(
                        f"Skip loading {pos_key}: checkpoint shape {tuple(src_pos.shape)} "
                        f"!= model shape {tuple(dst_pos.shape)} and resize is not applicable"
                    )
                    adapted.pop(pos_key, None)
                else:
                    adapted[pos_key] = resized.to(dtype=dst_pos.dtype, device=dst_pos.device)
                    logger.info(f"Resized {pos_key} from {tuple(src_pos.shape)} to {tuple(dst_pos.shape)}")

        adapted = _adapt_transformer_state_for_depth(
            adapted,
            target_depth=model.aggregator.depth,
        )
        return adapted

    def _resolve_latest_ckpt(self, root_dir, subdir_name):
        ckpt_dir = Path(root_dir)
        pattern = re.compile(r"checkpoint_step_(\d+)$")
        candidates = []
        if ckpt_dir.exists():
            print(f"Looking for checkpoints in {ckpt_dir}")
            for p in ckpt_dir.rglob("checkpoint_step_*"):
                if not p.is_dir():
                    continue
                m = pattern.match(p.name)
                if not m:
                    continue
                candidates.append((int(m.group(1)), p))

        for _, checkpoint_dir in sorted(candidates, key=lambda x: x[0], reverse=True):
            if not self._is_complete_checkpoint_dir(checkpoint_dir):
                continue
            candidate = checkpoint_dir / subdir_name / "diffusion_pytorch_model.safetensors"
            if _is_valid_safetensors(candidate):
                return candidate
        return None

    def _load_checkpoints(self):
        transformer_path = None
        ckpt_root = getattr(self.job_config, "ckpt_root", None)
        if getattr(self.job_config, "transformer_resume", False):
            transformer_resume_from = getattr(self.job_config, "transformer_resume_from", None)
            if transformer_resume_from:
                candidate = Path(transformer_resume_from)
                if _is_valid_safetensors(candidate):
                    transformer_path = candidate
                else:
                    logger.warning(f"Configured transformer resume checkpoint is invalid/corrupt: {candidate}")
            else:
                if ckpt_root:
                    transformer_path = self._resolve_latest_ckpt(ckpt_root, "transformer")
            if transformer_path is None and ckpt_root:
                fallback = self._resolve_latest_ckpt(ckpt_root, "transformer")
                if fallback is not None:
                    logger.warning(f"Falling back to latest valid transformer checkpoint: {fallback}")
                    transformer_path = fallback
        if transformer_path is None:
            transformer_pretrained = getattr(self.job_config, "transformer_pretrained", None)
            transformer_path = Path(transformer_pretrained) if transformer_pretrained else None

        if transformer_path is None or not transformer_path.exists():
            raise FileNotFoundError("Unable to locate transformer checkpoint")

        logger.info(f"Loading transformer checkpoint from: {transformer_path}")
        transformer_state = self._load_checkpoint_state(transformer_path)
        transformer_state = self._adapt_transformer_state_for_resolution(self.transformer, transformer_state)
        logger.info(self.transformer.load_state_dict(transformer_state, strict=True))

        action_head_path = None
        if getattr(self.job_config, "action_head_resume", False):
            action_head_resume_from = getattr(self.job_config, "action_head_resume_from", None)
            if action_head_resume_from:
                candidate = Path(action_head_resume_from)
                if _is_valid_safetensors(candidate):
                    action_head_path = candidate
                else:
                    logger.warning(f"Configured action-head resume checkpoint is invalid/corrupt: {candidate}")
            else:
                if ckpt_root:
                    action_head_path = self._resolve_latest_ckpt(ckpt_root, "action_head")
            if action_head_path is None and ckpt_root:
                fallback = self._resolve_latest_ckpt(ckpt_root, "action_head")
                if fallback is not None:
                    logger.warning(f"Falling back to latest valid action-head checkpoint: {fallback}")
                    action_head_path = fallback
        if action_head_path is None:
            action_head_pretrained = getattr(self.job_config, "action_head_pretrained", None)
            action_head_path = Path(action_head_pretrained) if action_head_pretrained else None

        if action_head_path is None or not action_head_path.exists():
            raise FileNotFoundError("Unable to locate action head checkpoint for RDT")

        logger.info(f"Loading RDT checkpoint from: {action_head_path}")
        action_head_state = self._load_checkpoint_state(action_head_path)
        action_head_state = _adapt_rdt_state_for_model(
            action_head_state,
            model=self.action_head,
        )
        logger.info(self.action_head.load_state_dict(action_head_state, strict=False))

    def _reset_runtime_buffers(self, prompt=None, text_emb=None):
        self.prompt = prompt
        if text_emb is not None:
            normalized_text_emb = self._normalize_text_emb_payload(text_emb)
            self.runtime_text_emb = (
                normalized_text_emb
                if normalized_text_emb is not None
                else self._resolve_text_emb_from_dataset(prompt)
            )
        else:
            self.runtime_text_emb = self._resolve_text_emb_from_dataset(prompt)
        self.action_history = []
        self.frame_history = []
        self.pose_history = []
        self.unexecuted_action_buffer.clear()
        self.prev_absolute_action_chunk = None
        self.actions_served_since_last_prediction = 0
        self.prev_executed_absolute_action_16d = None
        self.episode_initial_state = None
        self.current_anchor_abs_state = None
        self.transformer_past_key_values = [None] * self.transformer.aggregator.depth
        self.frame_st_id = 0
        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, "real", self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        self.ee_target_log_path = os.path.join(self.exp_save_root, "ee_target_log.jsonl")
        self._last_ee_target_log_record = None

    def _get_valid_obs_state(self, current_obs):
        if current_obs is None or current_obs.get("observation.state", None) is None:
            return None
        state = np.asarray(current_obs["observation.state"], dtype=np.float32).reshape(-1)
        if state.size != len(self.used_action_channel_ids):
            logger.warning(
                f"Skip eval warm start: expected observation.state with "
                f"{len(self.used_action_channel_ids)} values, got shape {tuple(state.shape)}"
            )
            return None
        return state

    def _warm_start_runtime_from_obs(self, current_obs):
        """Seed eval buffers from the current robot state before the first policy step."""
        state_16d = self._get_valid_obs_state(current_obs)
        if state_16d is None:
            return False

        self.current_anchor_abs_state = state_16d.copy()
        self.prev_executed_absolute_action_16d = state_16d.copy()
        self.prev_absolute_action_chunk = np.repeat(
            state_16d[:, None, None],
            self.chunk_size,
            axis=1,
        ).astype(np.float32)
        self.episode_initial_state = self.preprocess_action_state(
            state_16d,
            anchor_state=state_16d,
        ).to(self.device, dtype=self.dtype)

        state_token = self.episode_initial_state[0, 0].detach().cpu()
        state_frame = state_token.unsqueeze(-1).repeat(1, self.image_frame_stride)
        max_buffer = max(1, self.history_len * self.history_frame_stride)
        self.action_history = [state_frame.clone() for _ in range(max_buffer)]
        self.pose_history = [state_16d.copy() for _ in range(max_buffer)]

        frames = self._preprocess_obs_to_frames([current_obs])
        if len(frames) > 0:
            frame_tensor = frames[-1].detach().cpu()
            self.frame_history = [
                {
                    "frame": frame_tensor.clone(),
                    "action_abs": state_16d.copy(),
                    "pose": state_16d.copy(),
                }
                for _ in range(max_buffer)
            ]

        logger.info(
            f"Eval warm start: filled {len(self.frame_history)} frame/pose entries, "
            f"{len(self.action_history)} action-history entries, and seeded first action chunk from current state."
        )
        return True

    def _merge_latest_ee_target_log(self, update_payload):
        log_path = getattr(self, "ee_target_log_path", None)
        if log_path is None or self._last_ee_target_log_record is None:
            return False
        if not os.path.exists(log_path):
            return False

        record = dict(self._last_ee_target_log_record)
        record.update(update_payload)
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return False
            lines[-1] = json.dumps(record) + "\n"
            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self._last_ee_target_log_record = record
            return True
        except Exception as exc:
            logger.warning(f"Failed to merge planner feedback into {log_path}: {exc}")
            return False

    def _trim_transformer_kv_cache(self, past_key_values):
        if not isinstance(past_key_values, (list, tuple)):
            return past_key_values

        max_frames = 1
        trimmed = []
        for block_kv in past_key_values:
            if (
                block_kv is None
                or not isinstance(block_kv, (list, tuple))
                or len(block_kv) != 2
            ):
                trimmed.append(block_kv)
                continue

            k, v = block_kv
            if not torch.is_tensor(k) or not torch.is_tensor(v):
                trimmed.append(block_kv)
                continue

            # Expected KV cache shape from global attention: [B, heads, S_cache, P, dim].
            # Keep only the latest S_cache entries within the temporal window.
            if k.ndim >= 3 and v.ndim >= 3 and k.shape[2] > max_frames and v.shape[2] > max_frames:
                k = k[:, :, -max_frames:, ...]
                v = v[:, :, -max_frames:, ...]
            trimmed.append((k, v))

        return trimmed

    def _resize_pad_frame(self, image_np):
        frame = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        _, c, h, w = frame.shape
        scale = min(self.image_height / h, self.image_width / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        frame = F.interpolate(frame, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h = self.image_height - new_h
        pad_w = self.image_width - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        frame = F.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        return frame[0]

    def _preprocess_obs_to_frames(self, obs_items):
        merged_frames = []
        for obs in obs_items:
            per_cam = [self._resize_pad_frame(obs[k]) for k in self.job_config.obs_cam_keys]
            if self.multi_view_image_mode == "vertical":
                merged = torch.cat(per_cam, dim=1)
                merged_frames.append(merged)
            elif self.multi_view_image_mode == "first":
                merged_frames.append(per_cam[0])
            elif self.multi_view_image_mode == "frame":
                merged_frames.extend(per_cam)
            else:
                raise ValueError(f"Unsupported multi_view_image_mode `{self.multi_view_image_mode}`")
        return merged_frames

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(np.asarray(action))
        action_model_input_padded = F.pad(action_model_input, [0, 0, 0, 0, 0, 1], mode="constant", value=0)
        action_model_input = action_model_input_padded[self.inverse_used_action_channel_ids]

        if self.action_norm_method == "quantiles":
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6
            ) * 2.0 - 1.0
        else:
            raise NotImplementedError

        action_model_input = action_model_input * self.action_mask.to(action_model_input.device).view(-1, 1, 1)

        return action_model_input.unsqueeze(0).unsqueeze(-1)  # [B, C, F, N, 1]

    def preprocess_state(self, state):
        state = torch.from_numpy(np.asarray(state, dtype=np.float32)).flatten()
        if state.numel() != len(self.used_action_channel_ids):
            raise ValueError(
                f"Expected state with {len(self.used_action_channel_ids)} values, got shape {tuple(state.shape)}"
            )

        state_padded = F.pad(state, [0, 1], mode="constant", value=0)
        state_aligned = state_padded[self.inverse_used_action_channel_ids]

        if self.action_norm_method == "quantiles":
            q01 = self.actions_q01[:, 0, 0]
            q99 = self.actions_q99[:, 0, 0]
            state_aligned = (state_aligned - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            raise NotImplementedError

        state_aligned = state_aligned * self.action_mask.to(state_aligned.device)

        return state_aligned.unsqueeze(0).unsqueeze(1)  # [B, 1, C]

    @staticmethod
    def _safe_quat(quat):
        quat = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / norm

    def _relative_state_16(self, state, anchor_state=None):
        current = np.asarray(state, dtype=np.float32).reshape(-1)
        if current.size != len(self.used_action_channel_ids):
            raise ValueError(
                f"Expected state with {len(self.used_action_channel_ids)} values, got shape {tuple(current.shape)}"
            )
        if anchor_state is None:
            anchor_state = self.current_anchor_abs_state
        if anchor_state is None:
            anchor_state = current
        anchor_state = np.asarray(anchor_state, dtype=np.float32).reshape(-1)
        if anchor_state.size != len(self.used_action_channel_ids):
            raise ValueError(
                f"Expected anchor state with {len(self.used_action_channel_ids)} values, got shape {tuple(anchor_state.shape)}"
            )

        left_rot = (
            R.from_quat(self._safe_quat(anchor_state[3:7])[None]).inv()
            * R.from_quat(self._safe_quat(current[3:7])[None])
        ).as_quat().reshape(-1)
        right_rot = (
            R.from_quat(self._safe_quat(anchor_state[11:15])[None]).inv()
            * R.from_quat(self._safe_quat(current[11:15])[None])
        ).as_quat().reshape(-1)
        rel_state_16 = np.concatenate(
            [
                current[:3] - anchor_state[:3],
                left_rot,
                current[7:8],
                current[8:11] - anchor_state[8:11],
                right_rot,
                current[15:16],
            ]
        ).astype(np.float32)
        return torch.from_numpy(rel_state_16)

    def preprocess_relative_state(self, state, anchor_state=None):
        rel_state_16 = self._relative_state_16(state, anchor_state=anchor_state)

        state_padded = F.pad(rel_state_16, [0, 1], mode="constant", value=0)
        state_aligned = state_padded[self.inverse_used_action_channel_ids]
        if self.action_norm_method == "quantiles":
            q01 = self.actions_q01[:, 0, 0]
            q99 = self.actions_q99[:, 0, 0]
            state_aligned = (state_aligned - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            raise NotImplementedError

        state_aligned = state_aligned * self.action_mask.to(state_aligned.device)

        return state_aligned.unsqueeze(0).unsqueeze(1)  # [B, 1, C]

    def preprocess_action_state(self, state, anchor_state=None):
        if self.action_representation == "absolute":
            return self.preprocess_state(state)
        return self.preprocess_relative_state(state, anchor_state=anchor_state)

    def _compose_relative_pose(self, relative_pose, anchor_pose):
        relative_pose = np.asarray(relative_pose, dtype=np.float32).reshape(8)
        anchor_pose = np.asarray(anchor_pose, dtype=np.float32).reshape(8)
        rel_rot = R.from_quat(self._safe_quat(relative_pose[3:7])[None])
        anchor_rot = R.from_quat(self._safe_quat(anchor_pose[3:7])[None])
        abs_rot = (anchor_rot * rel_rot).as_quat().reshape(-1)
        abs_rot = self._safe_quat(abs_rot)
        abs_trans = relative_pose[:3] + anchor_pose[:3]
        return np.concatenate([abs_trans, abs_rot, relative_pose[7:8]]).astype(np.float32)

    def _compose_relative_compact_action(self, relative_action, anchor_state):
        relative_action = np.asarray(relative_action, dtype=np.float32)
        original_shape = relative_action.shape
        if relative_action.shape[0] != len(self.used_action_channel_ids):
            raise ValueError(
                f"Expected relative action with first dim {len(self.used_action_channel_ids)}, "
                f"got shape {relative_action.shape}"
            )
        flat_actions = relative_action.reshape(len(self.used_action_channel_ids), -1).T
        anchor_state = np.asarray(anchor_state, dtype=np.float32).reshape(len(self.used_action_channel_ids))
        absolute_actions = []
        for rel_action in flat_actions:
            left = self._compose_relative_pose(rel_action[:8], anchor_state[:8])
            right = self._compose_relative_pose(rel_action[8:], anchor_state[8:])
            absolute_actions.append(np.concatenate([left, right]).astype(np.float32))
        return np.stack(absolute_actions, axis=0).T.reshape(original_shape)

    def _slerp_quat(self, start_quat, end_quat, alpha):
        start_quat = self._safe_quat(start_quat)
        end_quat = self._safe_quat(end_quat)
        dot = float(np.dot(start_quat, end_quat))
        if dot < 0.0:
            end_quat = -end_quat
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return self._safe_quat((1.0 - alpha) * start_quat + alpha * end_quat)

        theta_0 = float(np.arccos(dot))
        sin_theta_0 = float(np.sin(theta_0))
        theta = theta_0 * alpha
        s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
        s1 = float(np.sin(theta) / sin_theta_0)
        return self._safe_quat(s0 * start_quat + s1 * end_quat)

    def _smooth_absolute_pose(self, previous_pose, current_pose, alpha):
        previous_pose = np.asarray(previous_pose, dtype=np.float32).reshape(8)
        current_pose = np.asarray(current_pose, dtype=np.float32).reshape(8)
        blended_translation = (1.0 - alpha) * previous_pose[:3] + alpha * current_pose[:3]
        blended_rotation = self._slerp_quat(previous_pose[3:7], current_pose[3:7], alpha)
        blended_gripper = (1.0 - alpha) * previous_pose[7:8] + alpha * current_pose[7:8]
        return np.concatenate([blended_translation, blended_rotation, blended_gripper]).astype(np.float32)

    def _smooth_absolute_compact_action(self, previous_action_16d, current_action_16d, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        previous_action_16d = np.asarray(previous_action_16d, dtype=np.float32).reshape(16)
        current_action_16d = np.asarray(current_action_16d, dtype=np.float32).reshape(16)
        if alpha <= 0.0:
            return previous_action_16d.copy()
        if alpha >= 1.0:
            return current_action_16d.copy()
        left = self._smooth_absolute_pose(previous_action_16d[:8], current_action_16d[:8], alpha)
        right = self._smooth_absolute_pose(previous_action_16d[8:], current_action_16d[8:], alpha)
        return np.concatenate([left, right]).astype(np.float32)

    def _clip_target_xyz(self, target_xyz, current_xyz, xyz_min, xyz_max):
        target_xyz = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        clipped = target_xyz.copy()
        if current_xyz is not None and self.ee_target_guard_max_delta_xyz > 0.0:
            current_xyz = np.asarray(current_xyz, dtype=np.float32).reshape(3)
            delta = np.clip(
                clipped - current_xyz,
                -self.ee_target_guard_max_delta_xyz,
                self.ee_target_guard_max_delta_xyz,
            )
            clipped = current_xyz + delta
        return np.minimum(np.maximum(clipped, xyz_min), xyz_max)

    def _apply_ee_target_guard(self, absolute_action, current_obs=None):
        if not self.ee_target_guard_enabled:
            return absolute_action
        guarded = np.asarray(absolute_action, dtype=np.float32).copy()
        if guarded.ndim == 2:
            guarded = guarded[:, :, None]
        if guarded.ndim != 3 or guarded.shape[0] != len(self.used_action_channel_ids):
            return absolute_action

        current_state = None
        if current_obs is not None and current_obs.get("observation.state", None) is not None:
            current_state = np.asarray(current_obs["observation.state"], dtype=np.float32).reshape(-1)
            if current_state.size != len(self.used_action_channel_ids):
                current_state = None

        left_current_xyz = None if current_state is None else current_state[:3]
        right_current_xyz = None if current_state is None else current_state[8:11]
        for step_idx in range(guarded.shape[1]):
            guarded[:3, step_idx, 0] = self._clip_target_xyz(
                guarded[:3, step_idx, 0],
                left_current_xyz,
                self.ee_target_guard_left_min,
                self.ee_target_guard_left_max,
            )
            guarded[8:11, step_idx, 0] = self._clip_target_xyz(
                guarded[8:11, step_idx, 0],
                right_current_xyz,
                self.ee_target_guard_right_min,
                self.ee_target_guard_right_max,
            )
            guarded[3:7, step_idx, 0] = self._safe_quat(guarded[3:7, step_idx, 0])
            guarded[11:15, step_idx, 0] = self._safe_quat(guarded[11:15, step_idx, 0])
        return guarded.astype(np.float32)

    def _log_executed_ee_target(self, current_obs, buffered_action_16d, served_action_16d):
        current_state = None
        if current_obs is not None and current_obs.get("observation.state", None) is not None:
            current_state = np.asarray(current_obs["observation.state"], dtype=np.float32).reshape(-1)
            if current_state.size != len(self.used_action_channel_ids):
                current_state = None

        buffered_action_16d = np.asarray(buffered_action_16d, dtype=np.float32).reshape(16)
        served_action_16d = np.asarray(served_action_16d, dtype=np.float32).reshape(16)

        def pose_payload(vec):
            return {
                "xyz": vec[:3].astype(float).tolist(),
                "quat": vec[3:7].astype(float).tolist(),
                "gripper": float(vec[7]),
            }

        def delta_payload(target, current):
            if current is None:
                return None
            delta_xyz = target[:3] - current[:3]
            return {
                "xyz": delta_xyz.astype(float).tolist(),
                "xyz_norm": float(np.linalg.norm(delta_xyz)),
            }

        left_current = None if current_state is None else current_state[:8]
        right_current = None if current_state is None else current_state[8:16]
        record = {
            "server_step": int(self.frame_st_id),
            "buffer_remaining": int(len(self.unexecuted_action_buffer)),
            "actions_served_since_last_prediction": int(self.actions_served_since_last_prediction),
            "ee_target_guard_enabled": bool(self.ee_target_guard_enabled),
            "ee_target_guard_max_delta_xyz": float(self.ee_target_guard_max_delta_xyz),
            "left": {
                "current": None if left_current is None else pose_payload(left_current),
                "buffered_target": pose_payload(buffered_action_16d[:8]),
                "served_target": pose_payload(served_action_16d[:8]),
                "served_delta_from_current": delta_payload(served_action_16d[:8], left_current),
            },
            "right": {
                "current": None if right_current is None else pose_payload(right_current),
                "buffered_target": pose_payload(buffered_action_16d[8:16]),
                "served_target": pose_payload(served_action_16d[8:16]),
                "served_delta_from_current": delta_payload(served_action_16d[8:16], right_current),
            },
        }

        log_path = getattr(self, "ee_target_log_path", None)
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        self._last_ee_target_log_record = record

    def _action_chunk_to_normalized_tensor(self, absolute_action_chunk, anchor_state=None):
        absolute_action_chunk = np.asarray(absolute_action_chunk, dtype=np.float32)
        if absolute_action_chunk.ndim == 3:
            absolute_action_chunk = absolute_action_chunk[:, :, 0]
        if absolute_action_chunk.shape[0] != len(self.used_action_channel_ids):
            raise ValueError(
                f"Expected absolute action chunk with first dim {len(self.used_action_channel_ids)}, "
                f"got shape {absolute_action_chunk.shape}"
            )
        tokens = []
        for token_idx in range(absolute_action_chunk.shape[1]):
            token = self.preprocess_action_state(
                absolute_action_chunk[:, token_idx],
                anchor_state=anchor_state,
            )[0, 0]
            tokens.append(token)
        norm_chunk = torch.stack(tokens, dim=1)  # [C, T]
        return norm_chunk.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, C, T, 1, 1]

    def _absolute_action_chunk_to_relative(self, absolute_action, anchor_state):
        absolute_action = np.asarray(absolute_action, dtype=np.float32)
        original_shape = absolute_action.shape
        if absolute_action.shape[0] != len(self.used_action_channel_ids):
            raise ValueError(
                f"Expected absolute action with first dim {len(self.used_action_channel_ids)}, "
                f"got shape {absolute_action.shape}"
            )
        flat_actions = absolute_action.reshape(len(self.used_action_channel_ids), -1).T
        relative_actions = [
            self._relative_state_16(action_16d, anchor_state=anchor_state).numpy()
            for action_16d in flat_actions
        ]
        return np.stack(relative_actions, axis=0).T.reshape(original_shape).astype(np.float32)

    def _build_warm_start_action_sample(self, current_obs, dtype):
        if self.prev_absolute_action_chunk is None or self.warm_start_blend <= 0.0:
            return None, 0
        if self.current_anchor_abs_state is None:
            return None, 0
        obs_state = current_obs.get("observation.state", None)
        if obs_state is None:
            return None, 0

        prev_chunk = np.asarray(self.prev_absolute_action_chunk, dtype=np.float32)
        if prev_chunk.ndim == 2:
            prev_chunk = prev_chunk[:, :, None]
        if prev_chunk.shape[0] != len(self.used_action_channel_ids) or prev_chunk.shape[1] < 1:
            return None, 0

        # The server may execute several buffered actions before the next model call.
        # Warm start should roll by the number of environment steps already served,
        # matching p2p's one-step roll when action_chunk_exec_steps == 1.
        shift_steps = max(1, int(self.actions_served_since_last_prediction))
        shift_steps = min(shift_steps, int(prev_chunk.shape[1]))
        shifted_chunk = np.empty_like(prev_chunk)
        if shift_steps < prev_chunk.shape[1]:
            shifted_chunk[:, :-shift_steps, :] = prev_chunk[:, shift_steps:, :]
            shifted_chunk[:, -shift_steps:, :] = prev_chunk[:, -1:, :]
        else:
            shifted_chunk[...] = prev_chunk[:, -1:, :]

        future_x0 = self._action_chunk_to_normalized_tensor(
            shifted_chunk,
            anchor_state=self.current_anchor_abs_state,
        ).to(self.device, dtype=dtype)
        current_state_x0 = self.preprocess_action_state(
            obs_state,
            anchor_state=self.current_anchor_abs_state,
        ).to(self.device, dtype=dtype)
        current_state_x0 = current_state_x0.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        x0_tokens = torch.cat(
            [
                current_state_x0,
                future_x0,
            ],
            dim=2,
        )
        if x0_tokens.shape[2] != self.rdt_horizon:
            return None, 0
        x0_tokens = x0_tokens.reshape(
            1,
            self.action_dim,
            self.pred_frames,
            self.tokens_per_frame,
            1,
        )
        x0_tokens = self._mask_action_sample_channels(x0_tokens)
        if self.warm_start_blend < 1.0:
            x0_tokens = self.warm_start_blend * x0_tokens + (1.0 - self.warm_start_blend) * torch.randn_like(x0_tokens)
        if self.warm_start_noise_std > 0.0:
            x0_tokens = x0_tokens + self.warm_start_noise_std * torch.randn_like(x0_tokens)
        x0_tokens = self._mask_action_sample_channels(x0_tokens.clamp(-1.0, 1.0))

        sigmas = self.train_scheduler_action.sigmas
        sigma_candidates = torch.nonzero(sigmas <= self.warm_start_sigma, as_tuple=False).flatten()
        if sigma_candidates.numel() == 0:
            start_idx = len(sigmas) - 1
        else:
            start_idx = int(sigma_candidates[0].item())
        sigma_start = sigmas[start_idx].to(device=self.device, dtype=dtype)
        noise = torch.randn_like(x0_tokens)
        sample = (1.0 - sigma_start) * x0_tokens + sigma_start * noise
        return self._mask_action_sample_channels(sample), start_idx

    def _mask_action_sample_channels(self, action_sample):
        """Keep eval action samples aligned with training masks.

        Training zeroes inactive/dummy action channels before RDT sees the noised
        action chunk. Do the same during online denoising so random dummy channels
        do not enter the action embedder.
        """
        action_mask = self.action_mask.to(action_sample.device)
        action_sample = action_sample.clone()
        action_sample[:, ~action_mask] = 0
        return action_sample

    def _apply_action_smoothing(self, absolute_action, relative_action, anchor_state):
        if self.prev_executed_absolute_action_16d is None or self.action_smoothing_alpha >= 1.0:
            return absolute_action, relative_action

        smoothed_action_16d = self._smooth_absolute_compact_action(
            self.prev_executed_absolute_action_16d,
            absolute_action[:, 0, 0],
            self.action_smoothing_alpha,
        )
        absolute_action = absolute_action.copy()
        relative_action = relative_action.copy()
        absolute_action[:, 0, 0] = smoothed_action_16d
        relative_action[:, 0, 0] = self._relative_state_16(
            smoothed_action_16d,
            anchor_state=anchor_state,
        ).numpy()
        return absolute_action, relative_action

    def _buffer_action_chunk(self, absolute_action_chunk):
        absolute_action_chunk = np.asarray(absolute_action_chunk, dtype=np.float32)
        if absolute_action_chunk.ndim == 2:
            absolute_action_chunk = absolute_action_chunk[:, :, None]
        if absolute_action_chunk.ndim != 3:
            raise ValueError(
                f"absolute_action_chunk must be [C, T] or [C, T, N], got {tuple(absolute_action_chunk.shape)}"
            )

        exec_steps = min(self.action_chunk_exec_steps, int(absolute_action_chunk.shape[1]))
        for step_idx in range(exec_steps):
            step = np.asarray(absolute_action_chunk[:, step_idx], dtype=np.float32).reshape(-1)
            self.unexecuted_action_buffer.append(step.copy())

    def postprocess_action(self, action):
        action = action.cpu()  # [B, C, F, N, 1]
        action = action[0, ..., 0]  # [C, F, N]
        # action = action.clamp(-1.0, 1.0)
        if self.action_norm_method == "quantiles":
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.detach().cpu().numpy()
        return action[self.used_action_channel_ids]

    def _state_token_from_action_tensor(self, action_tensor, use_latest_token=True):
        """Build state condition token with shape [1, 1, C] from [C, F, N] or [C, N]."""
        if action_tensor.ndim == 3:
            # [C, F, N] -> select one frame first.
            frame_idx = -1 if use_latest_token else 0
            frame_tokens = action_tensor[:, frame_idx, :]  # [C, N]
        elif action_tensor.ndim == 2:
            # [C, N]
            frame_tokens = action_tensor
        else:
            raise ValueError(f"action_tensor must be [C, F, N] or [C, N], got {tuple(action_tensor.shape)}")

        token_idx = -1 if use_latest_token else 0
        state_vec = frame_tokens[:, token_idx]  # [C]
        return state_vec.unsqueeze(0).unsqueeze(1)  # [1, 1, C]

    def _state_condition_from_current_obs(self, current_obs, dtype):
        obs_state = current_obs.get("observation.state", None)
        if obs_state is not None:
            return self.preprocess_action_state(
                obs_state,
                anchor_state=self.current_anchor_abs_state,
            ).to(self.device, dtype=dtype)
        return torch.zeros((1, 1, self.action_dim), device=self.device, dtype=dtype)

    def _first_action_state_condition(self, pred_frames, tokens_per_frame, dtype, current_obs=None):
        action_source = torch.zeros(
            (1, self.action_dim, pred_frames, tokens_per_frame, 1),
            device=self.device,
            dtype=dtype,
        )
        if current_obs is not None:
            action_source[:, :, 0, 0, 0] = self._state_condition_from_current_obs(current_obs, dtype)[:, 0]
        elif len(self.action_history) > 0:
            latest_action = self.action_history[-1]
            if latest_action.shape[1] != tokens_per_frame:
                latest_action = F.interpolate(
                    latest_action.unsqueeze(0),
                    size=tokens_per_frame,
                    mode="linear",
                    align_corners=False,
                )[0]
            action_source[:, :, 0, :, 0] = latest_action.to(self.device, dtype=dtype)
        return rearrange(action_source, "b c f n 1 -> b (f n) c")[:, :1]

    def _build_model_input(self, current_obs):
        current_frames = self._preprocess_obs_to_frames([current_obs])
        images = torch.stack(current_frames[-1:], dim=0).unsqueeze(0).to(self.device, dtype=self.dtype)
        image_mask = torch.ones_like(images, dtype=torch.bool)

        actions = torch.zeros(
            (1, self.action_dim, self.num_input_frames, self.image_frame_stride, 1),
            device=self.device,
            dtype=self.dtype,
        )
        actions_mask = torch.zeros_like(actions, dtype=torch.bool)

        past_actions = self.action_history[-(self.num_input_frames - 1):]
        for i, action_frame in enumerate(past_actions):
            target_idx = self.num_input_frames - 1 - len(past_actions) + i
            frame_action = action_frame
            if frame_action.shape[1] != self.image_frame_stride:
                frame_action = F.interpolate(
                    frame_action.unsqueeze(0),
                    size=self.image_frame_stride,
                    mode="linear",
                    align_corners=False,
                )[0]
            actions[:, :, target_idx, :, 0] = frame_action.to(self.device, dtype=self.dtype)
            actions_mask[:, :, target_idx, :, 0] = True

        b = images.shape[0]
        _, _, f, h, w = images.shape
        patch_f, patch_h, patch_w = self.patch_size
        image_grid_id = get_mesh_id(
            f // patch_f,
            h // patch_h,
            w // patch_w,
            t=0,
            f_w=1,
            f_shift=0,
            action=False,
        ).to(self.device)
        image_grid_id = image_grid_id[None].repeat(b, 1, 1)

        action_grid_id = get_mesh_id(
            self.num_input_frames + self.chunk_size,
            1,
            1,
            t=1,
            f_w=1,
            f_shift=0,
            action=True,
        ).to(self.device)
        action_grid_id = action_grid_id[None].repeat(b, 1, 1)

        image_dict = {
            "images": images,
            "images_mask": image_mask,
            "grid_id": image_grid_id,
            "text_emb": None,
        }
        action_dict = {
            "actions": actions,
            "actions_mask": actions_mask,
            "grid_id": action_grid_id,
            "text_emb": None,
        }
        pred_action_chunk_dict = {
            "noised_latent": torch.zeros((1, self.action_dim, self.rdt_horizon), device=self.device, dtype=self.dtype),
            "timesteps": torch.zeros((1,), device=self.device, dtype=torch.float32),
            "pred_frame_idx": torch.zeros((1,), device=self.device, dtype=torch.long),
            "latent": torch.zeros((1, self.action_dim, self.rdt_horizon), device=self.device, dtype=self.dtype),
        }

        return {
            "image_dict": image_dict,
            "action_dict": action_dict,
            "pred_action_chunk_dict": pred_action_chunk_dict,
            "chunk_size": self.rdt_horizon,
            "future_action_chunk_size": self.chunk_size,
        }

    def _select_history_indices(self, length):
        if length <= 0:
            return []
        indices = [
            length - 1 - (self.history_len - 1 - i) * self.history_frame_stride
            for i in range(self.history_len)
        ]
        return [max(0, min(length - 1, idx)) for idx in indices]

    def _update_transformer_cache_with_frame(self, frame_tensor, current_obs=None, build_conds=True):
        transformer_dtype = next(self.transformer.aggregator.patch_embed.parameters()).dtype
        frame_tensor = frame_tensor.to(self.device, dtype=transformer_dtype)
        current_pose = None
        if current_obs is not None and current_obs.get("observation.state", None) is not None:
            current_pose = np.asarray(current_obs["observation.state"], dtype=np.float32).reshape(-1)
        self.frame_history.append(
            {
                "frame": frame_tensor.detach().cpu(),
                "action_abs": None,
                "pose": None if current_pose is None else current_pose.copy(),
            }
        )
        if current_pose is not None:
            self.pose_history.append(current_pose.copy())
        elif len(self.pose_history) > 0:
            self.pose_history.append(self.pose_history[-1].copy())
        else:
            self.pose_history.append(np.zeros(len(self.used_action_channel_ids), dtype=np.float32))
        self._trim_rolling_history()

        if not build_conds:
            return None

        history_indices = self._select_history_indices(len(self.frame_history))
        if len(self.pose_history) != len(self.frame_history):
            raise ValueError(
                f"pose/frame history mismatch: poses={len(self.pose_history)}, frames={len(self.frame_history)}"
            )
        anchor_abs_state = self.pose_history[history_indices[0]].copy()
        self.current_anchor_abs_state = anchor_abs_state
        selected_frames = [
            self.frame_history[idx]["frame"].to(self.device, dtype=transformer_dtype)
            for idx in history_indices
        ]
        action_state = torch.zeros(
            (1, self.action_dim, len(selected_frames), self.image_frame_stride, 1),
            device=self.device,
            dtype=transformer_dtype,
        )
        action_mask = torch.zeros_like(action_state, dtype=torch.bool)
        current_local_idx = len(selected_frames) - 1
        for local_idx, frame_hist_idx in enumerate(history_indices):
            if local_idx == current_local_idx:
                if current_obs is not None and current_obs.get("observation.state", None) is not None:
                    current_state = self.preprocess_action_state(
                        current_obs["observation.state"],
                        anchor_state=anchor_abs_state,
                    ).to(self.device, dtype=transformer_dtype)
                    action_state[:, :, local_idx, 0, 0] = current_state[:, 0]
                    action_mask[:, :, local_idx, 0, 0] = True
                continue
            frame_action_abs = self.frame_history[frame_hist_idx].get("action_abs", None)
            if frame_action_abs is None:
                frame_action_abs = self.pose_history[frame_hist_idx]
            frame_action = self.preprocess_action_state(
                frame_action_abs,
                anchor_state=anchor_abs_state,
            )[0, 0].unsqueeze(-1)
            if frame_action is None:
                continue
            if frame_action.shape[1] != self.image_frame_stride:
                frame_action = F.interpolate(
                    frame_action.unsqueeze(0),
                    size=self.image_frame_stride,
                    mode="linear",
                    align_corners=False,
                )[0]
            action_state[:, :, local_idx, :, 0] = frame_action.to(self.device, dtype=transformer_dtype)
            action_mask[:, :, local_idx, :, 0] = True

        frame_idx = self.frame_st_id
        _, frame_h, frame_w = selected_frames[-1].shape
        image_grid_id = get_mesh_id(
            len(selected_frames),
            frame_h // self.patch_size[1],
            frame_w // self.patch_size[2],
            t=0,
            f_w=1,
            f_shift=frame_idx,
            action=False,
        ).to(self.device)[None]
        action_grid_id = get_mesh_id(
            len(selected_frames) + self.chunk_size,
            1,
            1,
            t=1,
            f_w=1,
            f_shift=frame_idx,
            action=True,
        ).to(self.device)[None]

        frame_payload = {
            "img": torch.stack(selected_frames, dim=0).unsqueeze(0),
            "actions": action_state[:, :, :, :, 0],
            "actions_mask": action_mask[:, :, :, :, 0],
            "image_grid_id": image_grid_id,
            "action_grid_id": action_grid_id,
            "text_emb": self.runtime_text_emb,
        }
        with torch.cuda.amp.autocast(enabled=False):
            transformer_out = self.transformer.inference(
                [frame_payload],
                past_key_values=None,
            )
        conds = transformer_out.ress
        return conds

    def _append_rdt_condition_history(self, conds):
        del conds

    def _trim_rolling_history(self):
        max_buffer = max(1, self.history_len * self.history_frame_stride)
        if len(self.frame_history) > max_buffer:
            self.frame_history = self.frame_history[-max_buffer:]
        if len(self.action_history) > max_buffer:
            self.action_history = self.action_history[-max_buffer:]
        if len(self.pose_history) > max_buffer:
            self.pose_history = self.pose_history[-max_buffer:]

    def _build_windowed_rdt_conds(self, frame_conds):
        conds = {
            "rdt_img_c": frame_conds["rdt_img_c"].to(self.device, dtype=self.dtype),
            "rdt_act_c": frame_conds["rdt_act_c"].to(self.device, dtype=self.dtype),
        }
        return conds

    def _get_rdt_lang_condition(self, dtype):
        if not self.rdt_use_language_condition:
            return None
        lang_c = self.runtime_text_emb
        if lang_c is None:
            return None
        if lang_c.ndim == 2:
            lang_c = lang_c.unsqueeze(1)
        elif lang_c.ndim != 3:
            raise ValueError(f"RDT language condition must be [B,D] or [B,L,D], got {tuple(lang_c.shape)}")
        return lang_c.to(device=self.device, dtype=dtype)

    def _predict_actions(self, current_obs, frame_conds=None):
        with torch.no_grad():
            if frame_conds is None:
                current_frames = self._preprocess_obs_to_frames([current_obs])
                frame_tensor = current_frames[-1].to(self.device, dtype=self.dtype)
                frame_conds = self._update_transformer_cache_with_frame(frame_tensor, current_obs=current_obs)
            conds = self._build_windowed_rdt_conds(frame_conds)
            action_head_dtype = next(self.action_head.parameters()).dtype
            conds_img_c = conds["rdt_img_c"].to(self.device, dtype=action_head_dtype)
            conds_act_c = conds["rdt_act_c"].to(self.device, dtype=action_head_dtype)
            expected_act_cond_len = self.pred_frames * self.tokens_per_frame
            if conds_act_c.shape[1] != expected_act_cond_len:
                raise ValueError(
                    f"RDT action condition length mismatch: act_c={conds_act_c.shape[1]}, "
                    f"noisy_action={expected_act_cond_len}"
                )

            obs_state = current_obs.get("observation.state", None)
            action_sample, timestep_start_idx = self._build_warm_start_action_sample(
                current_obs,
                dtype=self.dtype,
            )
            if action_sample is None:
                action_sample = torch.randn(
                    (1, self.action_dim, self.pred_frames, self.tokens_per_frame, 1),
                    device=self.device,
                    dtype=self.dtype,
                )
                timestep_start_idx = 0
            action_sample = self._mask_action_sample_channels(action_sample)

            if self.state_condition_mode == "first_action":
                # Match p2p+RDT inference: RDT uses state_actions[:, :1, :].
                # Here state_actions is the clean action-history/source tensor, not the noisy sample.
                state_c = self._first_action_state_condition(
                    pred_frames=self.pred_frames,
                    tokens_per_frame=self.tokens_per_frame,
                    dtype=self.dtype,
                    current_obs=current_obs,
                )
            elif self.state_condition_mode == "episode_initial" and obs_state is not None:
                state_c = self.preprocess_action_state(
                    obs_state,
                    anchor_state=self.current_anchor_abs_state,
                ).to(self.device, dtype=self.dtype)
            elif obs_state is not None and self.state_condition_mode == "latest":
                state_c = self.preprocess_action_state(
                    obs_state,
                    anchor_state=self.current_anchor_abs_state,
                ).to(self.device, dtype=self.dtype)
            elif len(self.action_history) > 0:
                last_state = self.action_history[-1]
                if last_state.shape[1] != self.tokens_per_frame:
                    last_state = F.interpolate(
                        last_state.unsqueeze(0),
                        size=self.tokens_per_frame,
                        mode="linear",
                        align_corners=False,
                    )[0]
                if self.state_condition_mode == "episode_initial" and self.episode_initial_state is not None:
                    state_c = self.episode_initial_state.to(self.device, dtype=self.dtype)
                elif self.state_condition_mode == "null":
                    state_c = torch.zeros((1, 1, self.action_dim), device=self.device, dtype=self.dtype)
                else:
                    # Match training "latest": use one-step state token with shape [B, 1, C].
                    state_c = self._state_token_from_action_tensor(last_state, use_latest_token=True).to(
                        self.device, dtype=self.dtype
                    )
            else:
                state_c = torch.zeros((1, 1, self.action_dim), device=self.device, dtype=self.dtype)
            state_c = state_c.to(self.device, dtype=action_head_dtype)

            timesteps = self.train_scheduler_action.timesteps.to(self.device)[timestep_start_idx:]
            for i, t in enumerate(timesteps):
                action_sample = self._mask_action_sample_channels(action_sample)
                x_in = rearrange(action_sample, "b c f n 1 -> b (f n) c").to(self.device, dtype=action_head_dtype)
                t_batch = torch.full((x_in.shape[0],), float(t.item()), device=self.device, dtype=torch.float32)
                flow_pred = self.action_head(
                    x_in,
                    t_batch,
                    lang_c=self._get_rdt_lang_condition(dtype=action_head_dtype),
                    img_c=conds_img_c,
                    act_c=conds_act_c,
                    state_c=state_c,
                    embed_input=True,
                    decode_output=True,
                )
                flow_pred = rearrange(flow_pred, "b (f n) c -> b c f n 1", f=self.pred_frames, n=self.tokens_per_frame)
                action_sample = self.train_scheduler_action.step(
                    flow_pred,
                    t,
                    action_sample,
                    prev_timestep=timesteps[i + 1] if (i + 1) < len(timesteps) else None,
                    to_final=(i + 1 == len(timesteps)),
                )
                action_sample = self._mask_action_sample_channels(action_sample)

        action_sample = self._mask_action_sample_channels(action_sample)
        action_tokens = rearrange(action_sample, "b c f n 1 -> b c (f n) 1 1")
        future_action_tokens = action_tokens[:, :, 1 : self.chunk_size + 1]
        predicted_action = self.postprocess_action(future_action_tokens)
        if self.action_representation == "relative":
            relative_action = predicted_action
            if self.current_anchor_abs_state is None:
                raise ValueError("Missing history-window anchor before composing predicted action.")
            absolute_action = self._compose_relative_compact_action(
                relative_action,
                self.current_anchor_abs_state,
            )
        else:
            absolute_action = predicted_action
            if self.current_anchor_abs_state is None:
                relative_action = predicted_action.copy()
            else:
                relative_action = self._absolute_action_chunk_to_relative(
                    absolute_action,
                    self.current_anchor_abs_state,
                )
        absolute_action = np.nan_to_num(
            absolute_action,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        ).astype(np.float32)
        absolute_action = self._apply_ee_target_guard(absolute_action, current_obs=current_obs)
        relative_action = np.nan_to_num(
            relative_action,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        ).astype(np.float32)
        return {
            "relative_action": relative_action,
            "absolute_action": absolute_action,
            "action_reference": self.current_anchor_abs_state.astype(np.float32).copy(),
        }

    @torch.no_grad()
    def infer(self, obs):
        if "planner_feedback" in obs:
            planner_feedback = obs.get("planner_feedback")
            merged = self._merge_latest_ee_target_log(
                {
                    "planner_feedback": planner_feedback,
                    "planner_success": (
                        bool(planner_feedback.get("planner_success"))
                        if isinstance(planner_feedback, dict)
                        and planner_feedback.get("planner_success") is not None
                        else None
                    ),
                }
            )
            if not merged:
                logger.warning("Received planner feedback but no EE target log record was available to update.")
            return {"planner_feedback_logged": bool(merged)}

        reset = obs.get("reset", False)
        prompt = obs.get("prompt", None)
        text_emb = obs.get("text_emb", None)
        compute_kv_cache = obs.get("compute_kv_cache", False)

        if reset:
            logger.info("******************* Reset server ******************")
            self._reset_runtime_buffers(prompt=prompt, text_emb=text_emb)
            reset_obs = obs.get("obs", None)
            if isinstance(reset_obs, list):
                reset_obs = reset_obs[-1] if len(reset_obs) > 0 else None
            if reset_obs is not None:
                self._warm_start_runtime_from_obs(reset_obs)
            return {}

        if text_emb is not None:
            normalized_text_emb = self._normalize_text_emb_payload(text_emb)
            if normalized_text_emb is not None:
                self.runtime_text_emb = normalized_text_emb
            elif prompt is not None:
                self.prompt = prompt
                self.runtime_text_emb = self._encode_prompt_text_emb(prompt)
        elif (self.runtime_text_emb is None) and prompt is not None:
            self.prompt = prompt
            self.runtime_text_emb = self._encode_prompt_text_emb(prompt)

        if compute_kv_cache:
            key_frames = obs.get("obs", [])
            if isinstance(key_frames, dict):
                key_frames = [key_frames]
            action_state = obs.get("state", None)
            action_state_norm = None
            if action_state is not None:
                action_state = np.asarray(action_state)
                if action_state.ndim == 2:
                    action_state = action_state[:, None, :]
                action_state_norm = self.preprocess_action(action_state)[0, :, :, :, 0].float()

            merged_frames = self._preprocess_obs_to_frames(key_frames) if len(key_frames) > 0 else []
            num_action_frames = 0 if action_state_norm is None else int(action_state_norm.shape[1])

            if len(merged_frames) > 0:
                frame_tensor = merged_frames[-1].to(self.device, dtype=self.dtype)
                _ = self._update_transformer_cache_with_frame(frame_tensor)
                self.frame_st_id += 1

            if action_state_norm is not None and num_action_frames > 0:
                self.action_history = [
                    action_state_norm[:, idx].detach().cpu()
                    for idx in range(num_action_frames)
                ]
                max_buffer = max(1, self.history_len * self.history_frame_stride)
                if len(self.action_history) > max_buffer:
                    self.action_history = self.action_history[-max_buffer:]
                start = max(0, len(self.frame_history) - len(self.action_history))
                for local_idx, action_tensor in enumerate(self.action_history[-len(self.frame_history):]):
                    frame_idx = start + local_idx
                    if frame_idx < len(self.frame_history):
                        self.frame_history[frame_idx]["action"] = action_tensor
                # Match training episode-initial state: actions[:, :, 0, 0, 0] -> [B, 1, C].
                self.episode_initial_state = self._state_token_from_action_tensor(
                    action_state_norm, use_latest_token=False
                ).to(
                    self.device, dtype=self.dtype
                )
            return {}

        current_obs = obs.get("obs", None)
        if current_obs is None:
            raise ValueError("obs payload must contain key `obs` for inference")
        if isinstance(current_obs, list):
            if len(current_obs) == 0:
                raise ValueError("obs list is empty")
            current_obs = current_obs[-1]

        current_frames = self._preprocess_obs_to_frames([current_obs])
        frame_tensor = current_frames[-1].to(self.device, dtype=self.dtype)
        need_new_chunk = len(self.unexecuted_action_buffer) == 0
        frame_conds = self._update_transformer_cache_with_frame(
            frame_tensor,
            current_obs=current_obs,
            build_conds=need_new_chunk,
        )

        if need_new_chunk:
            logger.info("################# Infer One Chunk (ActionVGGT + RDT) #################")
            pred = self._predict_actions(current_obs, frame_conds=frame_conds)
            action_chunk = pred["absolute_action"]
            self.prev_absolute_action_chunk = action_chunk.astype(np.float32).copy()
            self.actions_served_since_last_prediction = 0
            self._buffer_action_chunk(action_chunk)
        else:
            logger.info(
                f"################# Pop Buffered Action ({len(self.unexecuted_action_buffer)} remaining) #################"
            )

        if len(self.unexecuted_action_buffer) == 0:
            raise RuntimeError("Action buffer is empty after inference; cannot serve an action.")

        action_16d = self.unexecuted_action_buffer.popleft().astype(np.float32)
        buffered_action_16d = action_16d.copy()
        action = action_16d.reshape(len(self.used_action_channel_ids), 1, 1)
        if self.current_anchor_abs_state is None:
            relative_action = action.copy()
        else:
            relative_action = self._absolute_action_chunk_to_relative(
                action,
                self.current_anchor_abs_state,
            )
        action, relative_action = self._apply_action_smoothing(
            action,
            relative_action,
            anchor_state=self.current_anchor_abs_state,
        )
        action = self._apply_ee_target_guard(action, current_obs=current_obs)
        if self.current_anchor_abs_state is None:
            relative_action = action.copy()
        else:
            relative_action = self._absolute_action_chunk_to_relative(
                action,
                self.current_anchor_abs_state,
            )
        self._log_executed_ee_target(
            current_obs=current_obs,
            buffered_action_16d=buffered_action_16d,
            served_action_16d=action[:, 0, 0],
        )

        current_action_abs = torch.from_numpy(action[:, 0, 0]).float()
        self.action_history.append(current_action_abs.detach().cpu())
        self.prev_executed_absolute_action_16d = action[:, 0, 0].astype(np.float32).copy()
        if self.prev_absolute_action_chunk is not None:
            served_idx = int(self.actions_served_since_last_prediction)
            if 0 <= served_idx < self.prev_absolute_action_chunk.shape[1]:
                self.prev_absolute_action_chunk[:, served_idx, 0] = self.prev_executed_absolute_action_16d
        self.actions_served_since_last_prediction += 1
        if len(self.frame_history) > 0:
            self.frame_history[-1]["action_abs"] = action[:, 0, 0].astype(np.float32).copy()
        self._trim_rolling_history()
        if self.episode_initial_state is None:
            self.episode_initial_state = self.preprocess_action_state(
                action[:, 0, 0],
                anchor_state=self.current_anchor_abs_state,
            ).to(self.device, dtype=self.dtype)
        self.frame_st_id += 1
        action_reference = (
            self.current_anchor_abs_state.astype(np.float32).copy()
            if self.current_anchor_abs_state is not None
            else np.zeros(len(self.used_action_channel_ids), dtype=np.float32)
        )
        return {
            "action": action,
            "action_absolute": action,
            "action_relative": relative_action,
            "action_reference": action_reference,
        }


def run(args):
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.ckpt_root = config.save_root
        config.save_root = args.save_root

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    from distributed.util import init_distributed

    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    if args.single_trajectory:
        config.single_trajectory = True
    if args.single_trajectory_episode_index is not None:
        config.single_trajectory_episode_index = int(args.single_trajectory_episode_index)
        config.single_trajectory = True
    if args.single_trajectory_repo_id is not None:
        config.single_trajectory_repo_id = args.single_trajectory_repo_id
        config.single_trajectory = True

    model = VA_Server(config)
    run_async_server_mode(model, local_rank, config.host, port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default="robotwin_train",
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="(start) port",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help="save root",
    )
    parser.add_argument(
        "--single-trajectory",
        action="store_true",
        help="Enable single trajectory (single episode) debug mode during evaluation.",
    )
    parser.add_argument(
        "--single-trajectory-episode-index",
        type=int,
        default=None,
        help="Episode index used when --single-trajectory is enabled. If omitted, first available episode is used.",
    )
    parser.add_argument(
        "--single-trajectory-repo-id",
        type=str,
        default=None,
        help="Exact RobotWin repo folder name/path for --single-trajectory.",
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
