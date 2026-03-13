import torch
import torch.nn as nn
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
        action_dim=30,
        text_dim=4096,
        window_size=4,
        chunk_size=4,
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

        # original image processing backbone (DINO aggregator)
        self.aggregator = Aggregator(img_height=img_height, img_width=img_width, patch_size=patch_size, embed_dim=embed_dim)
        # self.camera_head = CameraHead(dim_in=2 * embed_dim)
        # self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        # self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

        self.rdt_image_proj = nn.Linear(3 * embed_dim, embed_dim)
        self.rdt_action_proj = nn.Linear(3 * embed_dim, embed_dim)
    


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
        action_mask = action_dict.get("actions_mask", None)  # [B, C_action, F, N, 1]
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

        action_grid_id = action_dict.get('grid_id', None)
        if text_emb is None:
            text_emb = action_dict.get('text_emb', None)

 
        # image_dict stores images as [B, C, F, H, W]; convert to [B, F, C, H, W]        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if images.ndim == 5:
            images = images.permute(0, 2, 1, 3, 4)
        # create views list for downstream heads/output
        B, F, C, H, W = images.shape
        views = [{"img": images[:, i]} for i in range(F)]

        # Call aggregator with dataset-style inputs if provided
        agg_out, token_idx = self.aggregator(
            images=masked_images,
            actions=masked_actions,
            text_emb=text_emb,
            image_mask=image_mask,
            action_mask=action_mask,
            image_grid_id=image_grid_id,
            action_grid_id=action_grid_id,
        )
        
        # Handle both old (2-tuple) and new (3-tuple/4-tuple with use_cache) returns
        if isinstance(agg_out, tuple):
            if len(agg_out) == 2:
                aggregated_tokens_list, patch_start_idx = agg_out
            elif len(agg_out) == 3:
                aggregated_tokens_list, patch_start_idx, past_key_values_agg = agg_out
            else:
                raise ValueError(f"Unexpected aggregator output length: {len(agg_out)}")
        else:
            raise ValueError("Aggregator output must be a tuple")

        pred_frame_idx_val = pred_frame_idx
        if torch.is_tensor(pred_frame_idx_val):
            pred_frame_idx_val = int(pred_frame_idx_val.item())

        tokens = aggregated_tokens_list[-1]  # [B, S, P, 3C]
        window_tokens_start = max(0, pred_frame_idx_val - self.window_size + 1)
        window_tokens_end = pred_frame_idx_val + 1
        window_tokens = tokens[:, window_tokens_start:window_tokens_end]
        token_dim = window_tokens.shape[-1]
        img_tokens = window_tokens[:, :, token_idx["image"][0]:token_idx["image"][1]]
        act_tokens = window_tokens[:, :, token_idx["action"][0]:token_idx["action"][1]]

        img_tokens = img_tokens.reshape(img_tokens.shape[0], -1, token_dim)
        act_tokens = act_tokens.reshape(act_tokens.shape[0], -1, token_dim)
        rdt_img_c = self.rdt_image_proj(img_tokens)
        rdt_act_c = self.rdt_action_proj(act_tokens)

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

            B, S = images.shape[:2]
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
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0) 
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) >= 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            processed_frames.append(frame)
        
        output = ActionVGGTOutput(ress=all_ress, views=processed_frames)
        return output