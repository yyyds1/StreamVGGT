#!/usr/bin/env python
"""Compute q01/q99 for absolute RobotWin compact EE actions.

This scans RobotWin LeRobot repositories, reads raw `action` and optionally
`observation.state`, maps compact 16D EE vectors into the current 30D model
layout, and prints config-ready quantile statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs import VA_CONFIGS  # noqa: E402
from dataset.lerobot_latent_dataset import LatentLeRobotDataset, recursive_find_file  # noqa: E402


POSE_CHANNELS_30D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
QUAT_CHANNELS_30D = [3, 4, 5, 6, 10, 11, 12, 13]
DUMMY_CHANNELS_30D = list(range(14, 28))


def _compact16_to_model30(compact: np.ndarray, inverse_used_action_channel_ids: list[int]) -> np.ndarray:
    compact = np.asarray(compact, dtype=np.float32)
    if compact.ndim != 2 or compact.shape[1] != 16:
        raise ValueError(f"Expected compact action shape [N, 16], got {compact.shape}")
    compact_padded = np.pad(compact, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
    return compact_padded[:, inverse_used_action_channel_ids].astype(np.float32, copy=False)


def _iter_repo_paths(dataset_path: str, single_task: str | None) -> list[Path]:
    info_files = recursive_find_file(dataset_path, "info.json")
    repos = [Path(v.split("/meta/info.json")[0]) for v in info_files]
    if single_task:
        repos = [p for p in repos if p.name.startswith(f"{single_task}-")]
    return sorted(repos)


def _as_numpy_column(batch, key: str) -> np.ndarray | None:
    value = batch.get(key, None)
    if value is None:
        return None
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value, dtype=np.float32)


def _collect_repo_actions(
    repo_path: Path,
    config,
    include_state: bool,
    batch_size: int,
) -> np.ndarray:
    dataset = LatentLeRobotDataset(repo_id=str(repo_path), config=config)
    hf_view = dataset._hf_action_view
    chunks = []
    for start in range(0, len(hf_view), batch_size):
        end = min(start + batch_size, len(hf_view))
        batch = hf_view[start:end]

        action = _as_numpy_column(batch, "action")
        if action is not None:
            if action.ndim != 2 or action.shape[1] != 16:
                raise ValueError(f"{repo_path}: expected action [N, 16], got {action.shape}")
            chunks.append(action)

        if include_state:
            state = _as_numpy_column(batch, "observation.state")
            if state is not None:
                if state.ndim != 2 or state.shape[1] != 16:
                    raise ValueError(f"{repo_path}: expected observation.state [N, 16], got {state.shape}")
                chunks.append(state)

    if len(chunks) == 0:
        return np.zeros((0, 16), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _format_python_list(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.10g}" for v in values.tolist()) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="vga_robotwin_train")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--single-task", default=None)
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--include-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-quat-range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    config = VA_CONFIGS[args.config_name]
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.single_task is not None:
        config.single_task = args.single_task

    repos = _iter_repo_paths(config.dataset_path, getattr(config, "single_task", None))
    if len(repos) == 0:
        raise FileNotFoundError(
            f"No LeRobot repositories with meta/info.json found under {config.dataset_path!r}"
        )

    compact_chunks = []
    total_rows = 0
    for repo in tqdm(repos, desc="Scanning RobotWin repos"):
        compact = _collect_repo_actions(
            repo_path=repo,
            config=config,
            include_state=args.include_state,
            batch_size=max(1, args.batch_size),
        )
        if compact.shape[0] > 0:
            compact_chunks.append(compact)
            total_rows += int(compact.shape[0])

    if len(compact_chunks) == 0:
        raise RuntimeError("No action/state rows found.")

    compact_actions = np.concatenate(compact_chunks, axis=0)
    model_actions = _compact16_to_model30(
        compact_actions,
        inverse_used_action_channel_ids=list(config.inverse_used_action_channel_ids),
    )

    q01 = np.quantile(model_actions, args.q_low, axis=0).astype(np.float32)
    q99 = np.quantile(model_actions, args.q_high, axis=0).astype(np.float32)

    q01[DUMMY_CHANNELS_30D] = 0.0
    q99[DUMMY_CHANNELS_30D] = 1.0
    if args.fixed_quat_range:
        q01[QUAT_CHANNELS_30D] = -1.0
        q99[QUAT_CHANNELS_30D] = 1.0

    result = {
        "representation": "absolute",
        "dataset_path": str(config.dataset_path),
        "single_task": getattr(config, "single_task", None),
        "num_repos": len(repos),
        "num_rows": total_rows,
        "include_state": bool(args.include_state),
        "q_low": float(args.q_low),
        "q_high": float(args.q_high),
        "fixed_quat_range": bool(args.fixed_quat_range),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }

    print("\n# Absolute-action normalization stats")
    print(f"# repos={len(repos)}, rows={total_rows}, include_state={args.include_state}")
    print("va_robotwin_cfg.action_representation = \"absolute\"")
    print("va_robotwin_cfg.norm_stat = {")
    print(f"    \"q01\": {_format_python_list(q01)},")
    print(f"    \"q99\": {_format_python_list(q99)},")
    print("}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote JSON stats to {json_path}")


if __name__ == "__main__":
    main()
