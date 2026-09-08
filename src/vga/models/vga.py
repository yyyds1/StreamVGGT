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
        ee_target_head_use_image_tokens=True,
        ee_target_head_image_cross_attn_depth=1,
        ee_target_sequence_len=1,
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
        self.ee_target_sequence_len = max(1, int(ee_target_sequence_len))

        self.rdt_img_cond_mode = rdt_img_cond_mode
        self.rdt_img_pool_size = max(int(rdt_img_pool_size), 1)
        self.rdt_img_keep_summary_tokens = bool(rdt_img_keep_summary_tokens)
        cond_cfg = dict(rdt_condition_tokens or {})
        self.rdt_cond_use_action_queries = bool(cond_cfg.get("use_action_queries", True))
        self.rdt_cond_use_image_tokens = bool(cond_cfg.get("use_image_tokens", True))
        self.rdt_cond_use_language_tokens = bool(cond_cfg.get("use_language_tokens", True))
        self.rdt_cond_use_ee_target_tokens = bool(cond_cfg.get("use_ee_target_tokens", False))
        self.rdt_condition_layer_mode = str(cond_cfg.get("layer_mode", "last")).lower()
        if self.rdt_condition_layer_mode not in {"last", "selected"}:
            raise ValueError(
                f"Unsupported RDT condition layer_mode `{self.rdt_condition_layer_mode}`; "
                "expected `last` or `selected`."
            )
        self.rdt_image_layer_idx = cond_cfg.get("image_layers", None)
        self.rdt_action_layer_idx = cond_cfg.get("action_layers", None)
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
                use_image_tokens=bool(ee_target_head_use_image_tokens),
                image_cross_attn_depth=int(ee_target_head_image_cross_attn_depth),
                num_waypoints=self.ee_target_sequence_len,
            )
        else:
            self.ee_target_head = None
        if self.rdt_cond_use_ee_target_tokens:
            if self.ee_target_head is None:
                raise ValueError("rdt_condition_tokens.use_ee_target_tokens=True requires enable_ee_target_head=True")
            self.ee_target_fourier_bands = 4
            xyz_dim = 3 + 2 * self.ee_target_fourier_bands * 3
            self.ee_target_condition_num_tokens = 4 * self.ee_target_sequence_len
            self.ee_target_cond_proj = nn.ModuleDict(
                {
                    "left_xyz": nn.Linear(xyz_dim, embed_dim),
                    "left_gripper": nn.Linear(1, embed_dim),
                    "right_xyz": nn.Linear(xyz_dim, embed_dim),
                    "right_gripper": nn.Linear(1, embed_dim),
                }
            )
        else:
            self.ee_target_fourier_bands = 0
            self.ee_target_condition_num_tokens = 0
            self.ee_target_cond_proj = None

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

    def _compose_rdt_condition_tokens(self, img_c, act_c, lang_c, ee_c=None):
        cond_parts = []
        if self.rdt_cond_use_action_queries:
            cond_parts.append(act_c)
        if self.rdt_cond_use_image_tokens:
            cond_parts.append(img_c)
        if self.rdt_cond_use_language_tokens:
            cond_parts.append(lang_c)
        if ee_c is not None:
            cond_parts.append(ee_c)
        if len(cond_parts) == 0:
            raise ValueError("No RDT condition token source is enabled")
        flat_parts = [
            cond.reshape(cond.shape[0], -1, cond.shape[-1]) if cond.dim() == 4 else cond
            for cond in cond_parts
        ]
        return torch.cat(flat_parts, dim=1)

    def _fourier_encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        bands = int(getattr(self, "ee_target_fourier_bands", 0))
        if bands <= 0:
            return xyz
        freq = (2.0 ** torch.arange(bands, device=xyz.device, dtype=xyz.dtype)) * torch.pi
        angles = xyz[..., None] * freq
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=-2)
        return torch.cat([xyz, sincos], dim=-1)

    def _build_ee_target_condition_tokens(
        self,
        ee_target: torch.Tensor,
        current_ee: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.ee_target_cond_proj is None:
            raise ValueError("ee_target condition projection is disabled")
        proj_param = next(self.ee_target_cond_proj.parameters())
        ee_target = ee_target.to(device=proj_param.device, dtype=proj_param.dtype)
        if ee_target.ndim == 3:
            ee_target = ee_target[:, :, None]
        elif ee_target.ndim != 4:
            raise ValueError(
                f"ee_target condition must be [B,2,4] or [B,2,K,4], got {tuple(ee_target.shape)}"
            )
        if ee_target.shape[1] != 2 or ee_target.shape[-1] != 4:
            raise ValueError(
                f"ee_target condition must have left/right arms and 4 channels, got {tuple(ee_target.shape)}"
            )
        if current_ee is None:
            current_ee = torch.zeros_like(ee_target)
        else:
            current_ee = current_ee.to(device=proj_param.device, dtype=proj_param.dtype)
            if current_ee.ndim == 3:
                current_ee = current_ee[:, :, None].expand(-1, -1, ee_target.shape[2], -1)
            elif current_ee.ndim != 4:
                raise ValueError(
                    f"current ee-target condition must be [B,2,4] or [B,2,K,4], got {tuple(current_ee.shape)}"
                )
            if current_ee.shape[2] == 1 and ee_target.shape[2] != 1:
                current_ee = current_ee.expand(-1, -1, ee_target.shape[2], -1)
            if current_ee.shape != ee_target.shape:
                raise ValueError(
                    f"current ee-target condition shape mismatch: got {tuple(current_ee.shape)}, "
                    f"expected {tuple(ee_target.shape)}"
                )

        delta = ee_target - current_ee
        left_xyz = self._fourier_encode_xyz(delta[:, 0, :, :3])
        left_gripper = delta[:, 0, :, 3:4]
        right_xyz = self._fourier_encode_xyz(delta[:, 1, :, :3])
        right_gripper = delta[:, 1, :, 3:4]

        tokens = [
            self.ee_target_cond_proj["left_xyz"](left_xyz),
            self.ee_target_cond_proj["left_gripper"](left_gripper),
            self.ee_target_cond_proj["right_xyz"](right_xyz),
            self.ee_target_cond_proj["right_gripper"](right_gripper),
        ]
        return torch.stack(tokens, dim=2).flatten(1, 2).to(dtype=dtype)

    def _append_ee_target_condition_tokens(
        self,
        act_tokens: torch.Tensor,
        ee_target: torch.Tensor,
        current_ee: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.ee_target_cond_proj is None:
            return act_tokens
        ee_tokens = self._build_ee_target_condition_tokens(
            ee_target=ee_target,
            current_ee=current_ee,
            dtype=act_tokens.dtype,
        ).to(device=act_tokens.device)
        if act_tokens.dim() == 3:
            return torch.cat([act_tokens, ee_tokens], dim=1)
        if act_tokens.dim() == 4:
            ee_tokens = ee_tokens[:, None].expand(-1, act_tokens.shape[1], -1, -1)
            return torch.cat([act_tokens, ee_tokens], dim=2)
        raise ValueError(
            f"RDT action tokens must be [B,L,D] or [B,N_layers,L,D], got {tuple(act_tokens.shape)}"
        )

    def _mix_ee_target_condition_override(
        self,
        pred_ee_target: torch.Tensor,
        override_ee_target: Optional[torch.Tensor],
        override_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if override_ee_target is None or override_mask is None:
            return pred_ee_target
        override_ee_target = override_ee_target.to(device=pred_ee_target.device, dtype=pred_ee_target.dtype)
        override_mask = override_mask.to(device=pred_ee_target.device, dtype=torch.bool)
        if override_ee_target.shape != pred_ee_target.shape:
            raise ValueError(
                "ee_target_condition_override shape mismatch: "
                f"got {tuple(override_ee_target.shape)}, expected {tuple(pred_ee_target.shape)}"
            )
        if override_mask.ndim == 1:
            override_mask = override_mask[:, None, None, None]
        elif override_mask.ndim == 2:
            override_mask = override_mask[:, :, None, None]
        elif override_mask.ndim == 3:
            override_mask = override_mask[..., None]
        elif override_mask.ndim != 4:
            raise ValueError(
                "ee_target_condition_override_mask must be [B], [B,2], [B,2,K], or [B,2,K,1], "
                f"got {tuple(override_mask.shape)}"
            )
        override_mask = override_mask.expand_as(pred_ee_target)
        return torch.where(override_mask, override_ee_target.detach(), pred_ee_target)

    @staticmethod
    def _normalize_layer_indices(layer_indices, num_layers: int, name: str):
        if layer_indices is None:
            return [num_layers - 1]
        indices = [int(idx) for idx in layer_indices]
        if len(indices) == 0:
            raise ValueError(f"{name} must contain at least one VGA layer index")
        normalized = []
        for idx in indices:
            if idx < 0:
                idx += num_layers
            if idx < 0 or idx >= num_layers:
                raise ValueError(
                    f"{name} contains invalid layer index {idx}; "
                    f"valid range is [0, {num_layers - 1}]"
                )
            normalized.append(idx)
        return normalized

    def _extract_global_half(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] == 2 * self.embed_dim:
            return tokens[..., self.embed_dim :]
        if tokens.shape[-1] == self.embed_dim:
            return tokens
        raise ValueError(
            f"Unexpected token dim {tokens.shape[-1]} for VGA; expected {self.embed_dim} or {2 * self.embed_dim}"
        )

    def _split_rdt_tokens_from_layer(self, tokens: torch.Tensor, token_idx: dict, current_image_frame_count: int):
        rdt_tokens = self._extract_global_half(tokens)
        img_tokens = rdt_tokens[:, :, token_idx["image"][0] : token_idx["image"][1]]
        ee_tokens = rdt_tokens[:, :, token_idx["ee_target"][0] : token_idx["ee_target"][1]]
        act_tokens = rdt_tokens[:, :, token_idx["action"][0] : token_idx["action"][1]]
        lang_tokens = rdt_tokens[:, :, token_idx["lang"][0] : token_idx["lang"][1]]

        current_frame_count = min(current_image_frame_count, img_tokens.shape[1])
        img_tokens = img_tokens[:, -current_frame_count:]
        ee_tokens = ee_tokens[:, -current_frame_count:]
        act_tokens = act_tokens[:, -1:]
        lang_tokens = lang_tokens[:, -1:]

        img_tokens = self._build_rdt_img_tokens(img_tokens)
        act_tokens = self._build_rdt_act_tokens(act_tokens)
        lang_tokens = lang_tokens.reshape(lang_tokens.shape[0], -1, lang_tokens.shape[-1])
        return img_tokens, ee_tokens, act_tokens, lang_tokens

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

        img_tokens, ee_tokens, act_tokens, lang_tokens = self._split_rdt_tokens_from_layer(
            aggregated_tokens_list[-1],
            token_idx,
            current_image_frame_count,
        )

        current_ee_tokens = ee_tokens.mean(dim=1)
        ee_target_image_tokens = img_tokens
        if self.rdt_condition_layer_mode == "selected":
            num_layers = len(aggregated_tokens_list)
            img_layer_idx = self._normalize_layer_indices(
                self.rdt_image_layer_idx,
                num_layers,
                "rdt_condition_tokens.image_layers",
            )
            act_layer_idx = self._normalize_layer_indices(
                self.rdt_action_layer_idx,
                num_layers,
                "rdt_condition_tokens.action_layers",
            )
            img_tokens = torch.stack(
                [
                    self._split_rdt_tokens_from_layer(
                        aggregated_tokens_list[layer_idx],
                        token_idx,
                        current_image_frame_count,
                    )[0]
                    for layer_idx in img_layer_idx
                ],
                dim=1,
            )
            ee_target_image_tokens = img_tokens
            act_tokens = torch.stack(
                [
                    self._split_rdt_tokens_from_layer(
                        aggregated_tokens_list[layer_idx],
                        token_idx,
                        current_image_frame_count,
                    )[2]
                    for layer_idx in act_layer_idx
                ],
                dim=1,
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
            geometry.update(
                self.ee_target_head(
                    current_ee_tokens,
                    image_tokens=ee_target_image_tokens,
                )
            )

        rdt_img_tokens = img_tokens
        rdt_act_tokens = act_tokens
        rdt_lang_tokens = lang_tokens
        rdt_ee_tokens = None
        if (
            geometry is not None
            and "ee_target" in geometry
            and self.rdt_cond_use_ee_target_tokens
            and self.ee_target_cond_proj is not None
        ):
            ee_target_condition = self._mix_ee_target_condition_override(
                geometry["ee_target"],
                input_dict.get("ee_target_condition_override", None),
                input_dict.get("ee_target_condition_override_mask", None),
            )
            rdt_ee_tokens = self._build_ee_target_condition_tokens(
                ee_target=ee_target_condition,
                current_ee=None,
                dtype=rdt_act_tokens.dtype,
            )
            if rdt_act_tokens.dim() == 4:
                rdt_ee_tokens = rdt_ee_tokens[:, None].expand(-1, rdt_act_tokens.shape[1], -1, -1)
        rdt_cond_c = self._compose_rdt_condition_tokens(
            img_c=rdt_img_tokens,
            act_c=rdt_act_tokens,
            lang_c=rdt_lang_tokens,
            ee_c=rdt_ee_tokens,
        )

        return VGAOutput(
            ress={
                "rdt_cond_c": rdt_cond_c,
                "rdt_img_c": rdt_img_tokens,
                "rdt_act_c": rdt_act_tokens,
                "rdt_ee_c": rdt_ee_tokens,
                "rdt_lang_c": rdt_lang_tokens,
                "rdt_img_tokens": rdt_img_tokens,
                "rdt_action_query_tokens": rdt_act_tokens,
                "rdt_ee_target_tokens": rdt_ee_tokens,
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
            "ee_target_condition_current": frames[0].get("ee_target_condition_current", None),
        }
        return self.forward(input_dict=input_dict, predict_geometry=False)
