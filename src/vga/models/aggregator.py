import logging
from typing import Optional

import torch
import torch.nn as nn

from actionvggt.layers.rope import PositionGetter, RotaryPositionEmbedding3D
from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.vision_transformer import vit_base, vit_giant2, vit_large, vit_small

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class VGAAggregator(nn.Module):
    """VGA aggregator with modality-specific frame attention and shared global attention.

    Frame attention policy:
    1) image tokens: frame-attn per camera view
    2) action tokens: frame-attn inside action branch only
    3) language tokens: self-only (kept unchanged in frame step)
    4) merged tokens: global-attn on concatenated sequence
    """

    def __init__(
        self,
        img_height=224,
        img_width=224,
        patch_size=14,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        qk_norm=True,
        init_values=0.01,
        action_dim=30,
        action_chunk_size=24,
        num_image_views=1,
        image_frame_stride=4,
        rope_freq=100,
    ):
        super().__init__()
        self.image_height = img_height
        self.image_width = img_width
        self.num_image_views = max(1, int(num_image_views))
        self.image_frame_stride = max(1, int(image_frame_stride))
        self.patch_size = patch_size
        self.depth = int(depth)
        self.action_chunk_size = int(action_chunk_size)
        self.action_dim = int(action_dim)

        self.__build_patch_embed__(
            patch_embed,
            img_height,
            img_width,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
        )

        self.rope = RotaryPositionEmbedding3D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Learnable action query tokens [C, D], where C is action chunk size.
        self.future_action_query_count = max(self.action_chunk_size - 1, 0)
        self.action_state_query_token = nn.Parameter(torch.randn(1, embed_dim))
        self.action_query_tokens = nn.Parameter(torch.randn(self.future_action_query_count, embed_dim))
        nn.init.normal_(self.action_state_query_token, std=1e-6)
        nn.init.normal_(self.action_query_tokens, std=1e-6)
        self.action_embedder = nn.Linear(self.action_dim, embed_dim)

        self.frame_blocks_image = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )
        self.frame_blocks_action = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )

        # Two slots: first-frame token and remaining-frame token.
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.ee_target_token = nn.Parameter(torch.randn(1, 1, 2, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.ee_target_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        self.token_idx = dict(
            camera=(0, 1),
            ee_target=(1, 3),
            register=(3, 3 + num_register_tokens),
        )

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_height,
        img_width,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=(img_height, img_width),
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
            return

        vit_models = {
            "dinov2_vitl14_reg": vit_large,
            "dinov2_vitb14_reg": vit_base,
            "dinov2_vits14_reg": vit_small,
            "dinov2_vitg2_reg": vit_giant2,
        }
        self.patch_embed = vit_models[patch_embed](
            img_size=(img_height, img_width),
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
            block_chunks=block_chunks,
            init_values=init_values,
        )
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)

    def _split_views(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, S, 3, H, W] -> [B, S, V, 3, H_v, W_v]
        bsz, seq_len, c_in, h, w = images.shape
        if self.num_image_views == 1:
            return images.unsqueeze(2)

        if h % self.num_image_views == 0 and (h // self.num_image_views) == self.image_height:
            return images.reshape(bsz, seq_len, c_in, self.num_image_views, self.image_height, w).permute(0, 1, 3, 2, 4, 5)
        if w % self.num_image_views == 0 and (w // self.num_image_views) == self.image_width:
            return images.reshape(bsz, seq_len, c_in, h, self.num_image_views, self.image_width).permute(0, 1, 4, 2, 3, 5)

        raise ValueError(
            "Unable to split multi-view image tensor. "
            f"Got shape {(bsz, seq_len, c_in, h, w)} and num_image_views={self.num_image_views}."
        )

    def _grid_to_positions(
        self,
        grid_id: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if grid_id is None:
            return torch.zeros(batch_size * seq_len, token_count, 3, device=device)

        if grid_id.ndim == 2:
            grid_id = grid_id.unsqueeze(0).expand(batch_size, -1, -1)
        elif grid_id.ndim != 3:
            raise ValueError(f"grid_id must be [4,L] or [B,4,L], got {tuple(grid_id.shape)}")

        if grid_id.shape[0] != batch_size:
            raise ValueError(f"grid_id batch mismatch: got {grid_id.shape[0]}, expected {batch_size}")
        if grid_id.shape[1] < 3:
            raise ValueError(f"grid_id must contain at least 3 coordinates, got shape {tuple(grid_id.shape)}")

        grid = grid_id[:, :3].to(device=device)
        if grid.shape[-1] != seq_len * token_count:
            raise ValueError(
                "grid_id token count mismatch: "
                f"got {grid.shape[-1]}, expected {seq_len * token_count}"
            )
        grid = grid.reshape(batch_size, 3, seq_len, token_count).permute(0, 2, 3, 1).contiguous()
        return grid.reshape(batch_size * seq_len, token_count, 3)

    @staticmethod
    def _zero_positions(batch_size: int, seq_len: int, token_count: int, device: torch.device, dtype: torch.dtype):
        return torch.zeros(batch_size * seq_len, token_count, 3, device=device, dtype=dtype)

    def forward(
        self,
        images: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        actions_mask: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        image_grid_id=None,
        action_grid_id=None,
        image_time_ids=None,
        return_all_layers: bool = True,
        **kwargs,
    ):
        del kwargs

        bsz, seq_len, c_in, _, _ = images.shape
        if c_in != 3:
            raise ValueError(f"Expected 3-channel images, got {c_in}")

        images = (images - self._resnet_mean.to(device=images.device, dtype=images.dtype)) / self._resnet_std.to(
            device=images.device, dtype=images.dtype
        )
        view_images = self._split_views(images)
        _, _, num_views, _, h_view, w_view = view_images.shape

        flat_view_images = view_images.reshape(bsz * seq_len * num_views, 3, h_view, w_view)
        patch_tokens = self.patch_embed(flat_view_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        patch_tokens = patch_tokens.reshape(bsz, seq_len, num_views, patch_tokens.shape[1], patch_tokens.shape[2])
        patches_per_view = patch_tokens.shape[3]
        image_tokens = patch_tokens.reshape(bsz, seq_len, num_views * patches_per_view, patch_tokens.shape[-1])

        action_seq_len = seq_len if actions is None else int(actions.shape[2])
        action_state_query_tokens = self.action_state_query_token[None, None].expand(bsz, action_seq_len, 1, -1)
        future_action_query_tokens = self.action_query_tokens[None].expand(bsz, -1, -1)
        if actions is None:
            action_tokens = action_state_query_tokens
        else:
            if actions.ndim != 5:
                raise ValueError(
                    f"actions must be [B,C,F,N,1], got shape {tuple(actions.shape)}"
                )
            if actions.shape[0] != bsz or actions.shape[1] != self.action_dim:
                raise ValueError(
                    f"actions shape {tuple(actions.shape)} is incompatible with images "
                    f"batch={bsz} and action_dim={self.action_dim}"
                )
            action_values = actions[..., 0].permute(0, 2, 3, 1).contiguous()  # [B,S,N,C]
            action_values = action_values[:, :, :1]
            action_tokens = self.action_embedder(action_values.to(self.action_embedder.weight.dtype)).to(images.dtype)

            if actions_mask is not None:
                if actions_mask.shape != actions.shape:
                    raise ValueError(
                        f"actions_mask shape {tuple(actions_mask.shape)} must match actions shape {tuple(actions.shape)}"
                    )
                token_mask = actions_mask[..., 0].permute(0, 2, 3, 1).any(dim=-1, keepdim=True)[:, :, :1]
                action_tokens = torch.where(token_mask.to(action_tokens.device), action_tokens, action_state_query_tokens)
            else:
                current_mask = torch.ones(
                    bsz, action_seq_len, 1, 1, device=action_tokens.device, dtype=torch.bool
                )
                current_mask[:, -1:] = False
                action_tokens = torch.where(current_mask, action_tokens, action_state_query_tokens)
        state_action_tokens = action_tokens.squeeze(2)
        action_tokens = torch.cat([state_action_tokens, future_action_query_tokens], dim=1)
        n_action_tokens = action_tokens.shape[1]

        if text_emb is None:
            lang_token = torch.zeros(bsz, 1, action_tokens.shape[-1], device=images.device, dtype=images.dtype)
        else:
            if text_emb.ndim == 2:
                pooled = text_emb
            elif text_emb.ndim == 3:
                pooled = text_emb.mean(dim=1)
            else:
                raise ValueError(f"text_emb must be [B,D] or [B,L,D], got shape {tuple(text_emb.shape)}")
            lang_token = pooled.unsqueeze(1).to(images.dtype)
        lang_tokens = lang_token.unsqueeze(1).repeat(1, seq_len, 1, 1)

        camera_token = slice_expand_and_flatten(self.camera_token, bsz, seq_len).reshape(bsz, seq_len, 1, -1)
        ee_target_token = self.ee_target_token.expand(bsz, seq_len, -1, -1)
        register_token = slice_expand_and_flatten(self.register_token, bsz, seq_len).reshape(
            bsz, seq_len, self.token_idx["register"][1] - self.token_idx["register"][0], -1
        )

        image_branch_special = torch.cat([camera_token, ee_target_token, register_token], dim=2)
        image_branch_tokens = image_tokens

        special_token_count = image_branch_special.shape[2]
        special_pos = self._zero_positions(bsz, seq_len, special_token_count, images.device, images.dtype)
        image_pos = self._grid_to_positions(image_grid_id, bsz, seq_len, num_views * patches_per_view, images.device)
        action_pos = self._grid_to_positions(action_grid_id, bsz, 1, n_action_tokens, images.device)
        lang_pos = self._zero_positions(bsz, seq_len, 1, images.device, images.dtype)
        image_pos_per_view = image_pos.reshape(bsz, seq_len, num_views, patches_per_view, 3)
        action_pos_flat = action_pos.reshape(bsz, n_action_tokens, 3)
        global_pos = torch.cat(
            [
                special_pos.reshape(bsz, seq_len, special_token_count, 3),
                image_pos.reshape(bsz, seq_len, num_views * patches_per_view, 3),
                action_pos_flat.unsqueeze(1).expand(-1, seq_len, -1, -1),
                lang_pos.reshape(bsz, seq_len, 1, 3),
            ],
            dim=2,
        ).reshape(bsz, seq_len * (special_token_count + num_views * patches_per_view + n_action_tokens + 1), 3)

        # Build frame-level modality branches.
        outputs = []
        last_output = None
        for layer_idx in range(self.depth):
            updated_view_tokens = []
            updated_special_tokens = []
            for view_idx in range(num_views):
                view_patches = patch_tokens[:, :, view_idx]  # [B, S, P_v, C]
                view_pos = torch.cat(
                    [
                        special_pos,
                        image_pos_per_view[:, :, view_idx].reshape(bsz * seq_len, patches_per_view, 3),
                    ],
                    dim=1,
                )
                view_input = torch.cat([image_branch_special, view_patches], dim=2).reshape(
                    bsz * seq_len, image_branch_special.shape[2] + patches_per_view, -1
                )
                view_output = self.frame_blocks_image[layer_idx](view_input, pos=view_pos)
                view_output = view_output.reshape(bsz, seq_len, image_branch_special.shape[2] + patches_per_view, -1)
                updated_special_tokens.append(view_output[:, :, : image_branch_special.shape[2]])
                updated_view_tokens.append(view_output[:, :, image_branch_special.shape[2] :])

            image_branch_special = torch.stack(updated_special_tokens, dim=0).mean(dim=0)
            patch_tokens = torch.stack(updated_view_tokens, dim=2)
            image_branch_tokens = patch_tokens.reshape(bsz, seq_len, num_views * patches_per_view, patch_tokens.shape[-1])

            action_tokens = self.frame_blocks_action[layer_idx](action_tokens, pos=action_pos_flat)

            # language frame-attn self-only: keep tokens unchanged.
            action_tokens_per_frame = action_tokens.unsqueeze(1).expand(-1, seq_len, -1, -1)
            merged = torch.cat([image_branch_special, image_branch_tokens, action_tokens_per_frame, lang_tokens], dim=2)

            p_total = merged.shape[2]
            global_in = merged.reshape(bsz, seq_len * p_total, -1)
            if image_time_ids is None:
                causal_mask = make_frame_causal_mask(seq_len, p_total, global_in.device, global_in.dtype)
            else:
                causal_mask = make_group_causal_mask(image_time_ids, seq_len, p_total, global_in.device, global_in.dtype)
            global_out = self.global_blocks[layer_idx](global_in, pos=global_pos, attn_mask=causal_mask)
            global_out = global_out.reshape(bsz, seq_len, p_total, -1)

            # Keep StreamVGGT-compatible geometry feature format: [frame || global].
            layer_out = torch.cat([merged, global_out], dim=-1)
            if return_all_layers:
                outputs.append(layer_out)
            else:
                last_output = layer_out

        if not return_all_layers:
            outputs = [last_output]

        n_special = image_branch_special.shape[2]
        self.token_idx["camera"] = (0, 1)
        self.token_idx["ee_target"] = (1, 3)
        self.token_idx["register"] = (3, n_special)
        self.token_idx["image"] = (n_special, n_special + num_views * patches_per_view)
        self.token_idx["action"] = (
            self.token_idx["image"][1],
            self.token_idx["image"][1] + n_action_tokens,
        )
        self.token_idx["lang"] = (self.token_idx["action"][1], self.token_idx["action"][1] + 1)
        return outputs, self.token_idx


def slice_expand_and_flatten(token_tensor, bsz, seq_len):
    query = token_tensor[:, :1, ...].expand(bsz, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:seq_len, ...].expand(bsz, max(seq_len - 1, 0), *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    return combined.reshape(bsz * seq_len, *combined.shape[2:])


def make_frame_causal_mask(seq_len, tokens_per_frame, device, dtype):
    if seq_len <= 1:
        return None
    frame_ids = torch.arange(seq_len, device=device).repeat_interleave(tokens_per_frame)
    blocked = frame_ids[:, None] < frame_ids[None, :]
    mask = torch.zeros((seq_len * tokens_per_frame, seq_len * tokens_per_frame), device=device, dtype=dtype)
    return mask.masked_fill(blocked, torch.finfo(dtype).min)


def make_group_causal_mask(group_ids, seq_len, tokens_per_frame, device, dtype):
    if seq_len <= 1:
        return None
    if group_ids.ndim == 2:
        group_ids = group_ids[0]
    elif group_ids.ndim != 1:
        raise ValueError(f"group_ids must be [S] or [B,S], got {tuple(group_ids.shape)}")
    if group_ids.numel() != seq_len:
        raise ValueError(f"group_ids length mismatch: got {group_ids.numel()}, expected {seq_len}")

    frame_ids = group_ids.to(device=device).repeat_interleave(tokens_per_frame)
    blocked = frame_ids[:, None] < frame_ids[None, :]
    mask = torch.zeros((seq_len * tokens_per_frame, seq_len * tokens_per_frame), device=device, dtype=dtype)
    return mask.masked_fill(blocked, torch.finfo(dtype).min)
