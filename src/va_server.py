# Copyright 2024-2026 The Robbyant Team Authors. All rights reserved.
import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file
from safetensors import safe_open

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
        self.chunk_size = int(getattr(job_config, "chunk_size", 24))
        self.image_frame_stride = int(getattr(job_config, "image_frame_stride", 8))
        self.action_dim = int(job_config.action_dim)
        self.patch_size = tuple(getattr(job_config, "patch_size", (1, 14, 14)))
        self.multi_view_image_mode = getattr(job_config, "multi_view_image_mode", "vertical")
        self.model_arch = str(getattr(job_config, "model_arch", "actionvggt")).lower()

        self.image_height = int(getattr(job_config, "image_height", job_config.height))
        self.image_width = int(getattr(job_config, "image_width", job_config.width))

        # If chunk size is not divisible by stride, keep one frame with full chunk tokens.
        if self.chunk_size % self.image_frame_stride == 0:
            self.pred_frames = self.chunk_size // self.image_frame_stride
            self.tokens_per_frame = self.image_frame_stride
        else:
            self.pred_frames = 1
            self.tokens_per_frame = self.chunk_size

        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        action_steps = int(getattr(self.job_config, "action_num_inference_steps", 50))
        self.train_scheduler_action.set_timesteps(action_steps)

        common_kwargs = dict(
            img_height=self.image_height,
            img_width=self.image_width,
            num_image_views=get_effective_num_image_views(self.job_config),
            rdt_img_cond_mode=getattr(job_config, "rdt_img_cond_mode", "full"),
            rdt_img_pool_size=getattr(job_config, "rdt_img_pool_size", 1),
            rdt_img_keep_summary_tokens=getattr(job_config, "rdt_img_keep_summary_tokens", False),
            window_size=self.num_input_frames,
            chunk_size=self.chunk_size,
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

        rdt_horizon = self.chunk_size
        rdt_x_pos_emb_config = [("act", rdt_horizon + self.job_config.rdt.num_register_tokens)]
        rdt_act_pos_emb_config = [("action", (self.num_input_frames, self.image_frame_stride))]

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
            max_act_len=self.num_input_frames * self.image_frame_stride,
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
        pattern = f"**/latents/chunk-*/{cam_key}/episode_*.pth"
        self._text_emb_search_files = sorted(dataset_root.glob(pattern))
        logger.info(
            f"Prepared text embedding search index with {len(self._text_emb_search_files)} latent files "
            f"from {dataset_root}"
        )

    def _resolve_text_emb_from_dataset(self, prompt: Optional[str]) -> Optional[torch.Tensor]:
        prompt_norm = self._normalize_prompt_text(prompt)
        if not prompt_norm:
            return None

        if prompt_norm in self._text_emb_cache:
            return self._text_emb_cache[prompt_norm]

        self._build_text_emb_search_files()
        for latent_file in self._text_emb_search_files:
            try:
                payload = torch.load(latent_file, map_location="cpu", weights_only=False)
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            text = self._normalize_prompt_text(payload.get("text", None))
            if text != prompt_norm:
                continue

            text_emb = payload.get("text_emb", None)
            if text_emb is None:
                continue
            if not torch.is_tensor(text_emb):
                text_emb = torch.as_tensor(text_emb)
            if text_emb.ndim == 2:
                text_emb = text_emb.unsqueeze(0)
            elif text_emb.ndim != 3:
                continue

            text_emb = text_emb.to(dtype=self.dtype, device=self.device)
            self._text_emb_cache[prompt_norm] = text_emb
            logger.info(f"Loaded text embedding for prompt from {latent_file}")
            return text_emb

        logger.warning(
            f"No matching dataset text embedding found for prompt: {prompt_norm!r}. "
            "Eval will fall back to no-language conditioning."
        )
        self._text_emb_cache[prompt_norm] = None
        return None

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

    def _reset_runtime_buffers(self, prompt=None):
        self.prompt = prompt
        self.runtime_text_emb = self._resolve_text_emb_from_dataset(prompt)
        self.action_history = []
        self.transformer_past_key_values = [None] * self.transformer.aggregator.depth
        self.frame_st_id = 0
        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, "real", self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)

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

        return action_model_input.unsqueeze(0).unsqueeze(-1)  # [B, C, F, N, 1]

    def postprocess_action(self, action):
        action = action.cpu()  # [B, C, F, N, 1]
        action = action[0, ..., 0]  # [C, F, N]
        if self.action_norm_method == "quantiles":
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.detach().cpu().numpy()
        return action[self.used_action_channel_ids]

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
            self.num_input_frames,
            self.image_frame_stride,
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
            "noised_latent": torch.zeros((1, self.action_dim, self.chunk_size), device=self.device, dtype=self.dtype),
            "timesteps": torch.zeros((1,), device=self.device, dtype=torch.float32),
            "pred_frame_idx": torch.zeros((1,), device=self.device, dtype=torch.long),
            "latent": torch.zeros((1, self.action_dim, self.chunk_size), device=self.device, dtype=self.dtype),
        }

        return {
            "image_dict": image_dict,
            "action_dict": action_dict,
            "pred_action_chunk_dict": pred_action_chunk_dict,
            "chunk_size": self.chunk_size,
        }

    def _update_transformer_cache_with_frame(self, frame_tensor):
        transformer_dtype = next(self.transformer.aggregator.patch_embed.parameters()).dtype
        frame_tensor = frame_tensor.to(self.device, dtype=transformer_dtype)

        action_state = torch.zeros((1, self.action_dim, 1, self.image_frame_stride, 1), device=self.device, dtype=transformer_dtype)
        if len(self.action_history) > 0:
            last_state = self.action_history[-1]
            if last_state.shape[1] != self.image_frame_stride:
                last_state = F.interpolate(
                    last_state.unsqueeze(0),
                    size=self.image_frame_stride,
                    mode="linear",
                    align_corners=False,
                )[0]
            action_state[:, :, 0, :, 0] = last_state.to(self.device, dtype=transformer_dtype)

        frame_idx = self.frame_st_id
        _, frame_h, frame_w = frame_tensor.shape
        image_grid_id = get_mesh_id(
            1,
            frame_h // self.patch_size[1],
            frame_w // self.patch_size[2],
            t=0,
            f_w=1,
            f_shift=frame_idx,
            action=False,
        ).to(self.device)[None]
        action_grid_id = get_mesh_id(
            1,
            self.image_frame_stride,
            1,
            t=1,
            f_w=1,
            f_shift=frame_idx,
            action=True,
        ).to(self.device)[None]

        frame_payload = {
            "img": frame_tensor,
            "actions": action_state[:, :, 0, :, 0],
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

    def _build_windowed_rdt_conds(self, frame_conds):
        conds = {
            "rdt_img_c": frame_conds["rdt_img_c"].to(self.device, dtype=self.dtype),
            "rdt_act_c": frame_conds["rdt_act_c"].to(self.device, dtype=self.dtype),
        }
        if "rdt_lang_c" in frame_conds and frame_conds["rdt_lang_c"] is not None:
            conds["rdt_lang_c"] = frame_conds["rdt_lang_c"].to(self.device, dtype=self.dtype)
        return conds

    def _predict_actions(self, current_obs):
        with torch.no_grad():
            current_frames = self._preprocess_obs_to_frames([current_obs])
            frame_tensor = current_frames[-1].to(self.device, dtype=self.dtype)
            frame_conds = self._update_transformer_cache_with_frame(frame_tensor)
            conds = self._build_windowed_rdt_conds(frame_conds)
            action_head_dtype = next(self.action_head.parameters()).dtype
            conds_img_c = conds["rdt_img_c"].to(self.device, dtype=action_head_dtype)
            conds_act_c = conds["rdt_act_c"].to(self.device, dtype=action_head_dtype)

            action_sample = torch.randn(
                (1, self.action_dim, self.pred_frames, self.tokens_per_frame, 1),
                device=self.device,
                dtype=self.dtype,
            )

            if len(self.action_history) > 0:
                last_state = self.action_history[-1]
                if last_state.shape[1] != self.tokens_per_frame:
                    last_state = F.interpolate(
                        last_state.unsqueeze(0),
                        size=self.tokens_per_frame,
                        mode="linear",
                        align_corners=False,
                    )[0]
                # Match training: use the latest action token as one-step robot state.
                state_c = last_state[:, -1:].unsqueeze(0).permute(0, 2, 1).to(self.device, dtype=self.dtype)
            else:
                state_c = torch.zeros((1, 1, self.action_dim), device=self.device, dtype=self.dtype)
            state_c = state_c.to(self.device, dtype=action_head_dtype)

            timesteps = self.train_scheduler_action.timesteps.to(self.device)
            for i, t in enumerate(timesteps):
                x_in = rearrange(action_sample, "b c f n 1 -> b (f n) c").to(self.device, dtype=action_head_dtype)
                t_batch = torch.full((x_in.shape[0],), float(t.item()), device=self.device, dtype=torch.float32)
                noise_pred = self.action_head(
                    x_in,
                    t_batch,
                    lang_c=conds.get("rdt_lang_c", None),
                    img_c=conds_img_c,
                    act_c=conds_act_c,
                    state_c=state_c,
                    embed_input=True,
                    decode_output=True,
                )
                noise_pred = rearrange(noise_pred, "b (f n) c -> b c f n 1", f=self.pred_frames, n=self.tokens_per_frame)
                action_sample = self.train_scheduler_action.step(noise_pred, t, action_sample)

        action_sample[:, ~self.action_mask.to(action_sample.device)] *= 0
        return self.postprocess_action(action_sample)

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get("reset", False)
        prompt = obs.get("prompt", None)
        compute_kv_cache = obs.get("compute_kv_cache", False)

        if (self.runtime_text_emb is None) and prompt is not None:
            self.prompt = prompt
            self.runtime_text_emb = self._resolve_text_emb_from_dataset(prompt)

        if reset:
            logger.info("******************* Reset server ******************")
            self._reset_runtime_buffers(prompt=prompt)
            return {}

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
                self.action_history = [action_state_norm[:, num_action_frames - 1]]
            return {}

        current_obs = obs.get("obs", None)
        if current_obs is None:
            raise ValueError("obs payload must contain key `obs` for inference")
        if isinstance(current_obs, list):
            if len(current_obs) == 0:
                raise ValueError("obs list is empty")
            current_obs = current_obs[-1]

        logger.info("################# Infer One Chunk (ActionVGGT + RDT) #################")
        action = self._predict_actions(current_obs)
        action_state_norm = self.preprocess_action(action)[0, :, :, :, 0].float()
        # Keep the newest predicted frame in history.
        self.action_history = [action_state_norm[:, -1]]
        self.frame_st_id += 1
        return {"action": action}


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
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
