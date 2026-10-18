"""Independent frozen DINOv2 RGB encoders for current left/right wrist observations."""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class WristDINOv2Encoder(nn.Module):
    def __init__(self, model_name_or_path, output_dim, image_size=224, local_files_only=False):
        super().__init__()
        self.image_size = int(image_size)
        self.encoders = nn.ModuleList([
            AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
            for _ in range(2)
        ])
        patch_size = int(self.encoders[0].config.patch_size)
        if self.image_size % patch_size:
            raise ValueError("wrist image_size must be divisible by DINOv2 patch size")
        self.tokens_per_camera = (self.image_size // patch_size) ** 2
        self.projections = nn.ModuleList([
            nn.Linear(int(encoder.config.hidden_size), int(output_dim))
            for encoder in self.encoders
        ])
        self.camera_embedding = nn.Parameter(torch.zeros(2, 1, int(output_dim)))
        nn.init.normal_(self.camera_embedding, std=0.02)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.freeze_backbones()

    def freeze_backbones(self):
        for encoder in self.encoders:
            encoder.requires_grad_(False)
            encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_backbones()
        return self

    def forward(self, images):
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError(f"wrist_images must be [B,2,3,H,W], got {tuple(images.shape)}")
        outputs = []
        for camera_idx, (encoder, projection) in enumerate(zip(self.encoders, self.projections)):
            with torch.no_grad():
                pixels = F.interpolate(images[:, camera_idx].float(), size=(self.image_size, self.image_size),
                                       mode="bilinear", align_corners=False)
                # Dataset provides RGB float tensors in [0,1], no second rescale.
                pixels = (pixels - self.image_mean) / self.image_std
                pixels = pixels.to(dtype=next(encoder.parameters()).dtype)
                patch_tokens = encoder(pixel_values=pixels).last_hidden_state[:, 1:]  # discard CLS
            if patch_tokens.shape[1] != self.tokens_per_camera:
                raise ValueError(f"Unexpected DINO patch count: {patch_tokens.shape[1]}")
            tokens = projection(patch_tokens.to(dtype=projection.weight.dtype))
            outputs.append(tokens + self.camera_embedding[camera_idx].to(dtype=tokens.dtype))
        return torch.cat(outputs, dim=1)  # [B, 2*patches, output_dim]


def append_wrist_tokens(image_tokens, wrist_tokens):
    """Append on the sequence axis; repeat wrists for every selected VGGT layer."""
    if image_tokens.ndim == 4:
        wrist_tokens = wrist_tokens[:, None].expand(-1, image_tokens.shape[1], -1, -1)
    elif image_tokens.ndim != 3:
        raise ValueError(f"Unsupported image condition shape: {tuple(image_tokens.shape)}")
    return torch.cat([image_tokens, wrist_tokens.to(image_tokens)], dim=-2)
