import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from actionvggt.models.aggregator import Aggregator
from actionvggt.heads.camera_head import CameraHead
from actionvggt.heads.dpt_head import DPTHead
from actionvggt.heads.track_head import TrackHead
from rdt.model import RDT
from transformers.file_utils import ModelOutput
from einops import rearrange
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class ActionVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None
    # embeddings produced when the model is used as a flow‑matching
    # vision‑action network
    # action_emb: Optional[torch.Tensor] = None
    # text_emb: Optional[torch.Tensor] = None
    # temb: Optional[torch.Tensor] = None
    # timestep_proj: Optional[torch.Tensor] = None

class ActionVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_height=518,
        img_width=518,
        patch_size=14,
        embed_dim=1024,
        aggregator_depth=24,
        action_dim=30,
        text_dim=4096,
        window_size=4,
        chunk_size=4,
        num_image_views=1,
        rdt_img_cond_mode="full",
        rdt_img_pool_size=1,
        rdt_img_keep_summary_tokens=False,
    ):
        """Vision–action model inspired by WAN transformer.

        Parameters added for the flow‑matching variant:
        * ``action_dim`` – dimensionality of the raw action vector
        * ``text_dim`` – text embedding size (passed through time/text
          embedder)
        * ``time_freq_dim``/``time_proj_dim`` – controls the timestep
          embedding module.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_image_views = num_image_views
        self.rdt_img_cond_mode = rdt_img_cond_mode
        self.rdt_img_pool_size = max(int(rdt_img_pool_size), 1)
        self.rdt_img_keep_summary_tokens = rdt_img_keep_summary_tokens
        self.patch_grid_h = img_height // patch_size
        self.patch_grid_w = img_width // patch_size
        self.aggregator_depth = int(aggregator_depth)

        # original image processing backbone (DINO aggregator)
        self.aggregator = Aggregator(
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=self.aggregator_depth,
        )
        # self.camera_head = CameraHead(dim_in=2 * embed_dim)
        # self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        # self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

        self.output_image_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.output_action_proj = nn.Linear(2 * embed_dim, embed_dim)
 
    def _build_rdt_img_tokens(self, img_tokens: torch.Tensor) -> torch.Tensor:
        if self.rdt_img_cond_mode == "full":
            return img_tokens.reshape(img_tokens.shape[0], -1, img_tokens.shape[-1])

        patches_per_view = self.patch_grid_h * self.patch_grid_w
        expected_tokens = patches_per_view * self.num_image_views
        if img_tokens.shape[2] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} image tokens per frame, got {img_tokens.shape[2]}. "
                "Check num_image_views and patch-grid settings."
            )

        bsz, frames, _, dim = img_tokens.shape
        view_tokens = img_tokens.reshape(
            bsz,
            frames,
            self.num_image_views,
            self.patch_grid_h,
            self.patch_grid_w,
            dim,
        )

        pooled_h = max(1, math.ceil(self.patch_grid_h / self.rdt_img_pool_size))
        pooled_w = max(1, math.ceil(self.patch_grid_w / self.rdt_img_pool_size))

        pooled = view_tokens.permute(0, 1, 2, 5, 3, 4).reshape(
            bsz * frames * self.num_image_views,
            dim,
            self.patch_grid_h,
            self.patch_grid_w,
        )
        pooled = F.adaptive_avg_pool2d(pooled, output_size=(pooled_h, pooled_w))
        pooled = pooled.reshape(
            bsz,
            frames,
            self.num_image_views,
            dim,
            pooled_h,
            pooled_w,
        ).permute(0, 1, 2, 4, 5, 3).reshape(
            bsz,
            frames,
            self.num_image_views,
            pooled_h * pooled_w,
            dim,
        )

        if self.rdt_img_keep_summary_tokens:
            summary = view_tokens.mean(dim=(3, 4)).unsqueeze(3)
            pooled = torch.cat([summary, pooled], dim=3)

        return pooled.reshape(bsz, -1, dim)


    def forward(
        self,
        input_dict: dict,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
    ):
        """Forward pass supporting two APIs:

        1. **legacy**: positional ``views`` list as before.
          2. **dataset style**: provide ``input_dict`` with
              ``image_dict`` + ``action_dict`` that mirrors the WAN structure.
              ``image_dict`` must include raw images in
              ``image_dict['images']`` with shape [B, C, F, H, W].
        """
        # --- unpack new‑style input dict if present ---
        image_dict = input_dict['image_dict']
        action_dict = input_dict['action_dict']
        target_dict = input_dict['pred_action_chunk_dict']
        self.chunk_size = input_dict['chunk_size']
        self.window_size = input_dict['window_size']

        actions = action_dict["actions"] # [B, C_action, F, N, 1]
        action_mask = action_dict.get("actions_mask", action_dict.get("action_mask", None))  # [B, C_action, F, N, 1]
        masked_actions = actions * action_mask if action_mask is not None else actions

        images = image_dict["images"]  # [B, C, F, H, W]
        image_mask = image_dict.get("images_mask", None)  # [B, C, F, H, W]
        masked_images = images * image_mask if image_mask is not None else images

        text_emb = image_dict.get("text_emb", None)  # [B, text_dim]
        image_grid_id = image_dict.get("grid_id", None)
        action_grid_id = action_dict.get("grid_id", None)

        noised_actions = target_dict.get('noised_latent', None) # [B, C_action, chunk_size, N, 1]
        timesteps = target_dict.get('timesteps', None) # [B]
        pred_frame_idx = target_dict.get('pred_frame_idx', None) # [B]

        if text_emb is None:
            text_emb = action_dict.get('text_emb', None)

 
        # image_dict stores images as [B, C, F, H, W]; convert to [B, F, C, H, W]        
        masked_images = masked_images.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        # create views list for downstream heads/output
        B, F, C, H, W = masked_images.shape
        views = [{"img": masked_images[:, i]} for i in range(F)]

        # Call aggregator with dataset-style inputs if provided
        agg_out = self.aggregator(
            images=masked_images,
            actions=masked_actions,
            text_emb=text_emb,
            image_grid_id=image_grid_id,
            action_grid_id=action_grid_id,
            return_all_layers=False,
        )
        
        # Handle both old (2-tuple) and new (3-tuple/4-tuple with use_cache) returns
        if isinstance(agg_out, tuple):
            if len(agg_out) == 2:
                aggregated_tokens_list, token_idx = agg_out
            elif len(agg_out) == 3:
                aggregated_tokens_list, token_idx, past_key_values_agg = agg_out
            else:
                raise ValueError(f"Unexpected aggregator output length: {len(agg_out)}")
        else:
            raise ValueError(f"Aggregator output must be a tuple, got: {type(agg_out)}")

        tokens = aggregated_tokens_list[-1]  # [B, S, P, 3C]
        bsz, seq_len = tokens.shape[:2]
        if pred_frame_idx is None:
            pred_frame_idx_tensor = torch.full(
                (bsz,),
                seq_len - 1,
                dtype=torch.long,
                device=tokens.device,
            )
        elif torch.is_tensor(pred_frame_idx):
            pred_frame_idx_tensor = pred_frame_idx.to(device=tokens.device, dtype=torch.long).reshape(-1)
            if pred_frame_idx_tensor.numel() == 1:
                pred_frame_idx_tensor = pred_frame_idx_tensor.repeat(bsz)
            elif pred_frame_idx_tensor.numel() != bsz:
                raise ValueError(
                    f"pred_frame_idx has {pred_frame_idx_tensor.numel()} elements, expected {bsz}"
                )
        else:
            pred_frame_idx_tensor = torch.full(
                (bsz,),
                int(pred_frame_idx),
                dtype=torch.long,
                device=tokens.device,
            )

        pred_frame_idx_tensor = pred_frame_idx_tensor.clamp(min=0, max=seq_len - 1)

        window_tokens_list = []
        for b in range(bsz):
            pred_idx = int(pred_frame_idx_tensor[b].item())
            start = max(0, pred_idx - self.window_size + 1)
            end = pred_idx + 1
            cur_tokens = tokens[b:b + 1, start:end]
            if cur_tokens.shape[1] < self.window_size:
                pad = torch.zeros(
                    (1, self.window_size - cur_tokens.shape[1], cur_tokens.shape[2], cur_tokens.shape[3]),
                    dtype=cur_tokens.dtype,
                    device=cur_tokens.device,
                )
                cur_tokens = torch.cat([pad, cur_tokens], dim=1)
            window_tokens_list.append(cur_tokens)

        window_tokens = torch.cat(window_tokens_list, dim=0)
        token_dim = window_tokens.shape[-1]
        img_tokens = window_tokens[:, :, token_idx["image"][0]:token_idx["image"][1]]
        act_tokens = window_tokens[:, :, token_idx["action"][0]:token_idx["action"][1]]

        img_tokens = self._build_rdt_img_tokens(img_tokens)
        act_tokens = act_tokens.reshape(act_tokens.shape[0], -1, token_dim)
        rdt_img_c = self.output_image_proj(img_tokens)
        rdt_act_c = self.output_action_proj(act_tokens)

        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):

            # if self.camera_head is not None:
            #     pose_enc_list = self.camera_head(aggregated_tokens_list)
            #     predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            # if self.depth_head is not None:
            #     depth, depth_conf = self.depth_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     predictions["depth"] = depth
            #     predictions["depth_conf"] = depth_conf

            # if self.point_head is not None:
            #     pts3d, pts3d_conf = self.point_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     predictions["world_points"] = pts3d
            #     predictions["world_points_conf"] = pts3d_conf

            # if self.track_head is not None and query_points is not None:
            #     track_list, vis, conf = self.track_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            #     )
            #     predictions["track"] = track_list[-1]  # track of the last iteration
            #     predictions["vis"] = vis
            #     predictions["conf"] = conf
            # predictions["images"] = images

            B, S = masked_images.shape[:2]
            # ress = []
            # for s in range(S):
            #     res = {
            #         'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
            #         'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

            #         'depth': predictions['depth'][:, s],  # [B, H, W, 1]
            #         'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
            #         'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

            #         **({'valid_mask': views[s]["valid_mask"]}
            #         if 'valid_mask' in views[s] else {}),  # [B, H, W]

            #         **({'track': predictions['track'][:, s],  # [B, N, 2]
            #             'vis': predictions['vis'][:, s],  # [B, N]
            #             'track_conf': predictions['conf'][:, s]}
            #         if 'track' in predictions else {})
            #     }
            #     ress.append(res)
            ress = dict(
                rdt_img_c=rdt_img_c,
                rdt_act_c=rdt_act_c,
            )
            out = ActionVGGTOutput(ress=ress, views=views)
            return out
        
    def inference(self, frames, past_key_values=None):
        if len(frames) == 0:
            raise ValueError("frames must contain at least one frame")

        first_img = frames[0]["img"]
        if first_img.dim() == 3:
            images = torch.stack([frame["img"] for frame in frames], dim=0).unsqueeze(0)  # [1, S, C, H, W]
        elif first_img.dim() == 4:
            images = torch.stack([frame["img"] for frame in frames], dim=1)  # [B, S, C, H, W]
        else:
            raise ValueError(f"Expected frame['img'] to have 3 or 4 dims, got shape {tuple(first_img.shape)}")

        bsz, seq_len = images.shape[:2]

        action_items = [frame.get("actions", frame.get("action", None)) for frame in frames]
        if all(action is not None for action in action_items):
            normalized_actions = []
            for action in action_items:
                if action.dim() == 2:
                    cur = action.unsqueeze(-1).unsqueeze(-1)
                elif action.dim() == 3:
                    cur = action.unsqueeze(-1)
                elif action.dim() == 4:
                    cur = action
                else:
                    raise ValueError(
                        "Each action tensor must have shape [B, C_action], [B, C_action, N], "
                        f"or [B, C_action, N, 1], got {tuple(action.shape)}"
                    )
                normalized_actions.append(cur)
            actions = torch.stack(normalized_actions, dim=2)  # [B, C_action, S, N, 1]
        else:
            actions = images.new_zeros((bsz, self.action_dim, seq_len, 1, 1))

        text_emb = frames[0].get("text_emb", None)
        image_grid_id = frames[0].get("image_grid_id", frames[0].get("grid_id", None))
        action_grid_id = frames[0].get("action_grid_id", None)

        use_cache = past_key_values is not None
        if use_cache:
            agg_out = self.aggregator(
                images=images,
                actions=actions,
                text_emb=text_emb,
                image_grid_id=image_grid_id,
                action_grid_id=action_grid_id,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=0,
                return_all_layers=False,
            )
        else:
            agg_out = self.aggregator(
                images=images,
                actions=actions,
                text_emb=text_emb,
                image_grid_id=image_grid_id,
                action_grid_id=action_grid_id,
                return_all_layers=False,
            )

        if not isinstance(agg_out, tuple):
            raise ValueError(f"Aggregator output must be a tuple, got: {type(agg_out)}")

        if len(agg_out) == 2:
            aggregated_tokens_list, token_idx = agg_out
            updated_past_key_values = None
        elif len(agg_out) == 3:
            aggregated_tokens_list, token_idx, updated_past_key_values = agg_out
        else:
            raise ValueError(f"Unexpected aggregator output length: {len(agg_out)}")

        tokens = aggregated_tokens_list[-1]  # [B, S, P, 2C]
        token_dim = tokens.shape[-1]

        img_tokens = tokens[:, :, token_idx["image"][0]:token_idx["image"][1]]
        act_tokens = tokens[:, :, token_idx["action"][0]:token_idx["action"][1]]

        img_tokens = self._build_rdt_img_tokens(img_tokens)
        act_tokens = act_tokens.reshape(act_tokens.shape[0], -1, token_dim)

        rdt_img_c = self.output_image_proj(img_tokens)
        rdt_act_c = self.output_action_proj(act_tokens)

        ress = {
            "rdt_img_c": rdt_img_c,
            "rdt_act_c": rdt_act_c,
        }
        if updated_past_key_values is not None:
            ress["past_key_values"] = updated_past_key_values

        return ActionVGGTOutput(ress=ress, views=frames)
    

if __name__ == '__main__':
    checkpoint_path = '../ckpt/actionvggt.pth'
    model = ActionVGGT()
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    # Save the model
    torch.save(model.state_dict(), '../ckpt/action.pth')
