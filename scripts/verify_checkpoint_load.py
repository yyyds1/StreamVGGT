#!/usr/bin/env python3
"""
Try building ActionVGGT + RDT and loading pretrained weights (same logic as train_va).
Paths in config are relative to the **src/** directory when training; this script resolves them.

Usage (from repo root, with .venv + CANN set_env):
  source .venv/bin/activate
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python scripts/verify_checkpoint_load.py

Optional:
  python scripts/verify_checkpoint_load.py --transformer /path/to/actionvggt.pth --action-head /path/to/RDT.pth
  python scripts/verify_checkpoint_load.py --cpu   # force CPU only
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


def _resolve_cfg_path(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    if path.is_absolute():
        return path
    return (SRC / path).resolve()


def _load_checkpoint_state(path: Path, map_location):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path), device=str(map_location))
    state = torch.load(str(path), map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="robotwin_train")
    parser.add_argument("--transformer", default=None, help="Override transformer pretrained path")
    parser.add_argument("--action-head", default=None, help="Override action head pretrained path")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only (no NPU/CUDA)")
    args = parser.parse_args()

    from configs import VA_CONFIGS

    cfg = VA_CONFIGS[args.config_name]

    if args.cpu:
        device = torch.device("cpu")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"[verify] device: {device}")
    print(f"[verify] config: {args.config_name}")

    t_path = _resolve_cfg_path(args.transformer or getattr(cfg, "transformer_pretrained", None))
    h_path = _resolve_cfg_path(args.action_head or getattr(cfg, "action_head_pretrained", None))

    print(f"[verify] transformer checkpoint: {t_path}")
    print(f"[verify] action_head checkpoint: {h_path}")

    from actionvggt.models.actionvggt import ActionVGGT
    from rdt.model import RDT

    dtype = cfg.param_dtype
    transformer = ActionVGGT()
    rdt_config = cfg.rdt
    patch_h = transformer.img_height // transformer.patch_size
    patch_w = transformer.img_width // transformer.patch_size
    img_tokens_per_frame = patch_h * patch_w
    act_tokens_per_frame = 1
    rdt_horizon = cfg.chunk_size
    rdt_x_pos_emb_config = [("act", rdt_horizon + cfg.rdt.num_register_tokens)]
    rdt_img_pos_emb_config = [("image", (cfg.window_size, patch_h, patch_w))]
    rdt_act_pos_emb_config = [("action", (cfg.window_size, act_tokens_per_frame))]

    action_head = RDT(
        horizon=rdt_horizon,
        output_size=transformer.action_dim,
        config=rdt_config,
        x_pos_emb_config=rdt_x_pos_emb_config,
        lang_pos_emb_config=None,
        max_lang_len=0,
        img_pos_emb_config=rdt_img_pos_emb_config,
        max_img_len=cfg.window_size * img_tokens_per_frame,
        act_pos_emb_config=rdt_act_pos_emb_config,
        max_act_len=cfg.window_size * act_tokens_per_frame,
        dtype=dtype,
    )

    # Load checkpoints on CPU, then move modules to device (avoids safetensors/NPU map edge cases)
    load_dev = "cpu"

    if t_path is None or not t_path.is_file():
        print(f"[verify] SKIP ActionVGGT load: missing file: {t_path}")
    else:
        st = _load_checkpoint_state(t_path, load_dev)
        incomp = transformer.load_state_dict(st, strict=False)
        print("[verify] ActionVGGT load_state_dict ok (strict=False)")
        print(f"       missing keys (first 5): {list(incomp.missing_keys)[:5]}")
        print(f"       unexpected keys (first 5): {list(incomp.unexpected_keys)[:5]}")

    if h_path is None or not h_path.is_file():
        print(f"[verify] SKIP RDT load: missing file: {h_path}")
    else:
        st = _load_checkpoint_state(h_path, load_dev)
        incomp = action_head.load_state_dict(st, strict=False)
        print("[verify] RDT load_state_dict ok (strict=False)")
        print(f"       missing keys (first 5): {list(incomp.missing_keys)[:5]}")
        print(f"       unexpected keys (first 5): {list(incomp.unexpected_keys)[:5]}")

    transformer = transformer.to(device=device, dtype=dtype)
    action_head = action_head.to(device=device, dtype=dtype)

    n_t = sum(p.numel() for p in transformer.parameters())
    n_h = sum(p.numel() for p in action_head.parameters())
    print(f"[verify] params: ActionVGGT={n_t:,}, RDT={n_h:,}")
    print("[verify] done.")


if __name__ == "__main__":
    main()
