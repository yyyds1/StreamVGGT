#!/usr/bin/env python3
"""
Random-init forward smoke test (no checkpoint). Default: ActionVGGT only.

Usage:
  source .venv/bin/activate && source .../ascend-toolkit/set_env.sh
  python scripts/verify_forward_npu.py
  python scripts/verify_forward_npu.py --cpu
  python scripts/verify_forward_npu.py --with-rdt   # also run RDT (needs state_c; train_va may differ)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="robotwin_train")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=None, help="Override image height")
    parser.add_argument("--width", type=int, default=None, help="Override image width")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override action chunk size")
    parser.add_argument("--window-size", type=int, default=None, help="Override context window size")
    parser.add_argument(
        "--with-rdt",
        action="store_true",
        help="Also run RDT head (smoke test; passes state_c=zeros — train_va wiring may omit state_c)",
    )
    args = parser.parse_args()

    from configs import VA_CONFIGS
    from utils import get_mesh_id
    from actionvggt.models.actionvggt import ActionVGGT
    cfg = VA_CONFIGS[args.config_name]
    dtype = cfg.param_dtype

    if args.cpu:
        device = torch.device("cpu")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"[forward] device={device} dtype={dtype}")

    B = args.batch_size
    F = args.frames
    H = args.height if args.height is not None else cfg.image_height
    W = args.width if args.width is not None else cfg.image_width
    chunk_size = args.chunk_size if args.chunk_size is not None else cfg.chunk_size
    window_size = args.window_size if args.window_size is not None else cfg.window_size
    action_dim = cfg.action_dim
    N = cfg.action_per_frame
    patch_f, patch_h, patch_w = cfg.patch_size
    if F % patch_f != 0:
        raise ValueError(f"--frames ({F}) must be divisible by patch_f ({patch_f})")
    if H % patch_h != 0 or W % patch_w != 0:
        raise ValueError(
            f"--height/--width ({H},{W}) must be divisible by patch size ({patch_h},{patch_w})"
        )
    if window_size > F:
        raise ValueError(f"--window-size ({window_size}) must be <= --frames ({F})")

    images = torch.randn(B, 3, F, H, W, device=device, dtype=dtype)
    actions = torch.randn(B, action_dim, F, N, 1, device=device, dtype=dtype)
    image_mask = torch.ones(B, 3, F, H, W, dtype=torch.bool, device=device)
    action_mask = torch.ones(B, action_dim, F, N, 1, dtype=torch.bool, device=device)

    image_grid_id = get_mesh_id(
        F // patch_f, H // patch_h, W // patch_w, t=0, f_w=1, f_shift=0, action=False
    ).to(device)[None].repeat(B, 1, 1)
    action_grid_id = get_mesh_id(F, N, 1, t=1, f_w=1, f_shift=0, action=True).to(device)[None].repeat(B, 1, 1)

    pred_frame_idx = torch.tensor([min(4, F - 1)], device=device, dtype=torch.long)
    noisy_latents = torch.randn(B, action_dim, chunk_size, device=device, dtype=dtype)
    timesteps = torch.rand(B, chunk_size, device=device, dtype=dtype)

    input_dict = {
        "image_dict": {
            "images": images,
            "images_mask": image_mask,
            "grid_id": image_grid_id,
        },
        "action_dict": {
            "actions": actions,
            "grid_id": action_grid_id,
            "actions_mask": action_mask,
        },
        "pred_action_chunk_dict": {
            "pred_frame_idx": pred_frame_idx,
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
        },
        "chunk_size": chunk_size,
        "window_size": window_size,
    }

    transformer = ActionVGGT(
        img_height=H,
        img_width=W,
        patch_size=patch_h,
        action_dim=action_dim,
        window_size=window_size,
        chunk_size=chunk_size,
    ).to(device=device)
    if dtype == torch.bfloat16:
        transformer = transformer.to(dtype=torch.bfloat16)

    transformer.eval()
    with torch.no_grad():
        out = transformer(input_dict)

    rdt_img_c = out.ress["rdt_img_c"]
    rdt_act_c = out.ress["rdt_act_c"]
    print(f"[forward] ActionVGGT ok: rdt_img_c {tuple(rdt_img_c.shape)}, rdt_act_c {tuple(rdt_act_c.shape)}")

    if not args.with_rdt:
        print("[forward] done (ActionVGGT only). Use --with-rdt to also run RDT.")
        return

    from rdt.model import RDT

    rdt_config = cfg.rdt
    patch_h_t = transformer.img_height // transformer.patch_size
    patch_w_t = transformer.img_width // transformer.patch_size
    img_tokens_per_frame = patch_h_t * patch_w_t
    act_tokens_per_frame = 1
    rdt_horizon = chunk_size
    rdt_x_pos_emb_config = [("act", rdt_horizon + cfg.rdt.num_register_tokens)]
    rdt_img_pos_emb_config = [("image", (window_size, patch_h_t, patch_w_t))]
    rdt_act_pos_emb_config = [("action", (window_size, act_tokens_per_frame))]

    action_head = RDT(
        horizon=rdt_horizon,
        output_size=transformer.action_dim,
        config=rdt_config,
        x_pos_emb_config=rdt_x_pos_emb_config,
        lang_pos_emb_config=None,
        max_lang_len=0,
        img_pos_emb_config=rdt_img_pos_emb_config,
        max_img_len=window_size * img_tokens_per_frame,
        act_pos_emb_config=rdt_act_pos_emb_config,
        max_act_len=window_size * act_tokens_per_frame,
        dtype=dtype,
    ).to(device=device)

    action_chunk = noisy_latents.permute(0, 2, 1)
    action_chunk_emb = action_head.action_embedder(action_chunk)
    t_step = timesteps[:, 0]

    # train_va.py currently omits state_c; RDT.forward asserts state_c — provide zeros for smoke test
    state_c = torch.zeros(B, 1, action_head.hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        h = action_head(
            action_chunk_emb,
            t_step,
            img_c=rdt_img_c.to(dtype),
            act_c=rdt_act_c.to(dtype),
            state_c=state_c,
        )
        action_pred = action_head.action_decoder(h)

    print(f"[forward] RDT ok: action_pred {tuple(action_pred.shape)}")
    print("[forward] done.")


if __name__ == "__main__":
    main()
