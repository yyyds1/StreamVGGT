from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import PixArtAlphaTextProjection
from huggingface_hub import PyTorchModelHubMixin
from transformers.file_utils import ModelOutput

from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from vga.heads import EETargetHead
from vga.models.aggregator import VGAAggregator
from vga.utils import apply_lora_to_module


@dataclass
class VGAOutput(ModelOutput):
    ress: Optional[dict] = None
    geometry: Optional[dict] = None
    views: Optional[torch.Tensor] = None


class VGA(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_height=224,
        img_width=224,
        patch_size=14,
        embed_dim=1024,
        aggregator_depth=12,
        action_dim=30,
        window_size=4,
        chunk_size=24,
        num_image_views=1,
        image_frame_stride=8,
        text_embed_dim=4096,
        rdt_img_cond_mode="full",
        rdt_img_pool_size=1,
        rdt_img_keep_summary_tokens=False,
        rdt_condition_tokens=None,
        enable_camera_depth_heads=True,
        enable_camera_head=True,
        enable_depth_head=True,
        enable_ee_target_head=False,
        ee_target_head_num_heads=8,
        ee_target_head_trunk_depth=4,
        ee_target_head_num_iterations=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_image_views = num_image_views
        self.image_frame_stride = image_frame_stride
        self.text_embed_dim = int(text_embed_dim)

        self.rdt_img_cond_mode = rdt_img_cond_mode
        self.rdt_img_pool_size = max(int(rdt_img_pool_size), 1)
        self.rdt_img_keep_summary_tokens = bool(rdt_img_keep_summary_tokens)
        cond_cfg = dict(rdt_condition_tokens or {})
        self.rdt_cond_use_action_queries = bool(cond_cfg.get("use_action_queries", True))
        self.rdt_cond_use_image_tokens = bool(cond_cfg.get("use_image_tokens", True))
        self.rdt_cond_use_language_tokens = bool(cond_cfg.get("use_language_tokens", True))
        if not (
            self.rdt_cond_use_action_queries
            or self.rdt_cond_use_image_tokens
            or self.rdt_cond_use_language_tokens
        ):
            raise ValueError("At least one RDT condition token source must be enabled for VGA")

        self.aggregator = VGAAggregator(
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=aggregator_depth,
            action_dim=action_dim,
            action_chunk_size=chunk_size,
            num_image_views=num_image_views,
            image_frame_stride=image_frame_stride,
        )
        self.text_token_proj = PixArtAlphaTextProjection(
            self.text_embed_dim,
            embed_dim,
            act_fn="gelu_tanh",
        )

        self.enable_camera_depth_heads = bool(enable_camera_depth_heads)
        self.enable_camera_head = bool(enable_camera_head) and self.enable_camera_depth_heads
        self.enable_depth_head = bool(enable_depth_head) and self.enable_camera_depth_heads
        if self.enable_camera_head:
            self.camera_head = CameraHead(dim_in=2 * embed_dim)
        else:
            self.camera_head = None

        if self.enable_depth_head:
            self.depth_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
        else:
            self.depth_head = None

        self.enable_ee_target_head = bool(enable_ee_target_head)
        if self.enable_ee_target_head:
            self.ee_target_head = EETargetHead(
                embed_dim=embed_dim,
                num_heads=int(ee_target_head_num_heads),
                trunk_depth=int(ee_target_head_trunk_depth),
                num_iterations=int(ee_target_head_num_iterations),
            )
        else:
            self.ee_target_head = None

        self.lora_replaced_modules = []
        self.lora_config = None

    def _project_text_token(self, text_emb: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if text_emb is None:
            return None
        if text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(1)
        elif text_emb.ndim != 3:
            raise ValueError(f"text_emb must be [B,D] or [B,L,D], got shape {tuple(text_emb.shape)}")

        proj_param = next(self.text_token_proj.parameters())
        text_emb = text_emb.to(device=proj_param.device, dtype=proj_param.dtype)
        text_token = text_emb.mean(dim=1, keepdim=True)
        text_token = self.text_token_proj(text_token)
        return text_token.to(dtype=dtype)

    def enable_lora(
        self,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        target_modules: Optional[tuple[str, ...]] = None,
    ):
        """Attach LoRA adapters to the pretrained VGA backbone only."""
        target_modules = target_modules or ("qkv", "proj", "fc1", "fc2")
        self.lora_replaced_modules = apply_lora_to_module(
            self.aggregator,
            target_names=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.lora_config = {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "target_modules": tuple(target_modules),
        }
        return self.lora_replaced_modules

    def prepare_lora_training(self):
        """Freeze pretrained backbone weights and keep LoRA + new action-query tokens trainable."""
        self.aggregator.requires_grad_(False)
        for name, param in self.aggregator.named_parameters():
            if "lora_" in name or name.endswith("action_query_tokens") or name.startswith("action_embedder."):
                param.requires_grad = True
        return self

    def _build_rdt_img_tokens(self, img_tokens: torch.Tensor) -> torch.Tensor:
        if self.rdt_img_cond_mode == "full":
            return img_tokens.reshape(img_tokens.shape[0], -1, img_tokens.shape[-1])

        # VGA keeps this branch simple and robust: pool tokens per frame.
        bsz, seq_len, n_tok, dim = img_tokens.shape
        pooled = img_tokens.mean(dim=2, keepdim=True)
        if self.rdt_img_keep_summary_tokens:
            return torch.cat([pooled, img_tokens], dim=2).reshape(bsz, seq_len * (n_tok + 1), dim)
        return pooled.reshape(bsz, seq_len, dim)

    def _build_rdt_act_tokens(self, act_tokens: torch.Tensor) -> torch.Tensor:
        # RDT act condition is [current_state_token, action_query_tokens...],
        # aligned one-to-one with the RDT noised action horizon.
        tokens_per_frame = int(self.chunk_size)
        history_len = int(self.window_size)
        expected_tokens = history_len + max(tokens_per_frame - 1, 0)
        if act_tokens.shape[2] < expected_tokens:
            raise ValueError(
                f"Not enough action tokens: got {act_tokens.shape[2]}, expected at least {expected_tokens}"
            )
        current_state_token = act_tokens[:, :, history_len - 1 : history_len]
        future_query_tokens = act_tokens[:, :, history_len : history_len + tokens_per_frame - 1]
        act_tokens = torch.cat([current_state_token, future_query_tokens], dim=2)
        return act_tokens.reshape(act_tokens.shape[0], -1, act_tokens.shape[-1])

    def _extract_geometry_predictions(self, aggregated_tokens_list, images, patch_start_idx):
        geometry = {}
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            geometry["camera_pose"] = pose_enc_list[-1]
        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            geometry["depth"] = depth
            geometry["depth_conf"] = depth_conf
        return geometry

    def _compose_rdt_condition_tokens(self, img_c, act_c, lang_c):
        cond_parts = []
        if self.rdt_cond_use_action_queries:
            cond_parts.append(act_c)
        if self.rdt_cond_use_image_tokens:
            cond_parts.append(img_c)
        if self.rdt_cond_use_language_tokens:
            cond_parts.append(lang_c)
        if len(cond_parts) == 0:
            raise ValueError("No RDT condition token source is enabled")
        return torch.cat(cond_parts, dim=1)

    def forward(self, input_dict: dict, predict_geometry: bool = True):
        image_dict = input_dict["image_dict"]

        images = image_dict["images"]  # [B, C, F, H, W]
        text_emb = image_dict.get("text_emb", input_dict.get("action_dict", {}).get("text_emb", None))
        current_image_frame_count = image_dict.get("current_image_frame_count", 1)
        if torch.is_tensor(current_image_frame_count):
            current_image_frame_count = int(current_image_frame_count.reshape(-1)[0].item())
        else:
            current_image_frame_count = int(current_image_frame_count)
        current_image_frame_count = max(1, current_image_frame_count)

        image_mask = image_dict.get("images_mask", None)
        image_grid_id = image_dict.get("grid_id", image_dict.get("image_grid_id", None))
        image_time_ids = image_dict.get("image_time_ids", None)
        action_dict = input_dict.get("action_dict", {})
        actions = action_dict.get("actions", None)
        actions_mask = action_dict.get("actions_mask", action_dict.get("action_mask", None))
        action_grid_id = action_dict.get("grid_id", action_dict.get("action_grid_id", None))

        if image_mask is not None:
            images = images * image_mask

        text_token = self._project_text_token(text_emb, dtype=images.dtype)

        images = images.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        aggregated_tokens_list, token_idx = self.aggregator(
            images=images,
            actions=actions,
            actions_mask=actions_mask,
            text_emb=text_token,
            image_grid_id=image_grid_id,
            action_grid_id=action_grid_id,
            image_time_ids=image_time_ids,
            return_all_layers=True,
        )

        tokens = aggregated_tokens_list[-1]  # [B, F, P, 2C]
        # Use the last-layer global half as RDT condition tokens (no output projection).
        if tokens.shape[-1] == 2 * self.embed_dim:
            rdt_tokens = tokens[..., self.embed_dim :]
        elif tokens.shape[-1] == self.embed_dim:
            rdt_tokens = tokens
        else:
            raise ValueError(
                f"Unexpected token dim {tokens.shape[-1]} for VGA; expected {self.embed_dim} or {2 * self.embed_dim}"
            )

        img_tokens = rdt_tokens[:, :, token_idx["image"][0] : token_idx["image"][1]]
        ee_tokens = rdt_tokens[:, :, token_idx["ee_target"][0] : token_idx["ee_target"][1]]
        act_tokens = rdt_tokens[:, :, token_idx["action"][0] : token_idx["action"][1]]
        lang_tokens = rdt_tokens[:, :, token_idx["lang"][0] : token_idx["lang"][1]]
        current_image_frame_count = min(current_image_frame_count, img_tokens.shape[1])
        img_tokens = img_tokens[:, -current_image_frame_count:]
        ee_tokens = ee_tokens[:, -current_image_frame_count:]
        act_tokens = act_tokens[:, -1:]
        lang_tokens = lang_tokens[:, -1:]

        current_ee_tokens = ee_tokens.mean(dim=1)
        img_tokens = self._build_rdt_img_tokens(img_tokens)
        act_tokens = self._build_rdt_act_tokens(act_tokens)
        lang_tokens = lang_tokens.reshape(lang_tokens.shape[0], -1, lang_tokens.shape[-1])

        rdt_img_tokens = img_tokens
        rdt_act_tokens = act_tokens
        rdt_lang_tokens = lang_tokens
        rdt_cond_c = self._compose_rdt_condition_tokens(
            img_c=rdt_img_tokens,
            act_c=rdt_act_tokens,
            lang_c=rdt_lang_tokens,
        )

        geometry = None
        if predict_geometry and self.enable_camera_depth_heads:
            geometry = self._extract_geometry_predictions(
                aggregated_tokens_list=aggregated_tokens_list,
                images=images,
                patch_start_idx=token_idx["image"][0],
            )
        if self.ee_target_head is not None:
            if geometry is None:
                geometry = {}
            geometry.update(self.ee_target_head(current_ee_tokens))

        return VGAOutput(
            ress={
                "rdt_cond_c": rdt_cond_c,
                "rdt_img_c": rdt_img_tokens,
                "rdt_act_c": rdt_act_tokens,
                "rdt_lang_c": rdt_lang_tokens,
                "rdt_img_tokens": rdt_img_tokens,
                "rdt_action_query_tokens": rdt_act_tokens,
            },
            geometry=geometry,
            views=images,
        )

    def inference(self, frames, past_key_values=None):
        del past_key_values
        if len(frames) == 0:
            raise ValueError("frames must contain at least one frame")

        first_img = frames[0]["img"]
        if first_img.dim() == 3:
            images = torch.stack([frame["img"] for frame in frames], dim=0).unsqueeze(0)
        elif first_img.dim() == 4:
            images = torch.stack([frame["img"] for frame in frames], dim=1)
        elif first_img.dim() == 5:
            if len(frames) != 1:
                raise ValueError("Pre-batched 5D frame payload expects a single frame item")
            images = first_img
        else:
            raise ValueError(f"Expected frame['img'] to have 3, 4, or 5 dims, got {tuple(first_img.shape)}")

        text_emb = frames[0].get("text_emb", None)
        image_grid_id = frames[0].get("grid_id", frames[0].get("image_grid_id", None))
        action_dict = {"text_emb": text_emb}
        first_action = frames[0].get("actions", None)
        if len(frames) == 1 and first_action is not None and first_action.dim() == 4:
            mask = frames[0].get("actions_mask", torch.ones_like(first_action, dtype=torch.bool))
            action_dict.update(
                {
                    "actions": first_action.unsqueeze(-1),
                    "actions_mask": mask.unsqueeze(-1),
                }
            )
        else:
            action_frames = []
            action_mask_frames = []
            for frame in frames:
                action = frame.get("actions", None)
                if action is None:
                    continue
                if action.dim() == 2:
                    action = action.unsqueeze(0)
                action_frames.append(action)
                mask = frame.get("actions_mask", None)
                if mask is None:
                    mask = torch.ones_like(action, dtype=torch.bool)
                elif mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                action_mask_frames.append(mask)

            if len(action_frames) == len(frames):
                actions = torch.stack(action_frames, dim=2).unsqueeze(-1)
                actions_mask = torch.stack(action_mask_frames, dim=2).unsqueeze(-1)
                action_dict.update({"actions": actions, "actions_mask": actions_mask})
        action_grid_id = frames[0].get("action_grid_id", None)
        if action_grid_id is not None:
            action_dict["grid_id"] = action_grid_id

        input_dict = {
            "image_dict": {
                "images": images.permute(0, 2, 1, 3, 4),
                "text_emb": text_emb,
                "grid_id": image_grid_id,
                "current_image_frame_count": frames[0].get("current_image_frame_count", 1),
                "image_time_ids": frames[0].get("image_time_ids", None),
                "image_view_ids": frames[0].get("image_view_ids", None),
            },
            "action_dict": action_dict,
        }
        return self.forward(input_dict=input_dict, predict_geometry=False)
