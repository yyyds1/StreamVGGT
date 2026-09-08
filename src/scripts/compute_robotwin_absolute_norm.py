#!/usr/bin/env python
"""Compute RobotWin LeRobot q01/q99 stats for all supported action modes.

The script scans RoboTwin LeRobot parquet repositories directly and emits
normalization statistics for:

* ``ee_absolute``: 30D model layout, from end-effector poses/grippers.
* ``ee_relative``: 30D model layout, relative to the sampled history anchor.
* ``joint_absolute``: 14D joint vector.
* ``joint_delta``: 14D next-joint minus current-joint vector, last row zeroed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs import VA_CONFIGS  # noqa: E402


QUAT_CHANNELS_30D = [3, 4, 5, 6, 10, 11, 12, 13]
DUMMY_CHANNELS_30D = list(range(14, 28))


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


ACTION_COLUMNS = [
    "endpose.left_endpose",
    "endpose.left_gripper",
    "endpose.right_endpose",
    "endpose.right_gripper",
    "joint_action.vector",
]


def _load_parquet_table(path: Path, columns: list[str] | None = None):
    try:
        import pandas as pd

        return pd.read_parquet(path, columns=columns)
    except ImportError:
        pass

    try:
        import pyarrow.parquet as pq

        return pq.read_table(path, columns=columns).to_pandas()
    except ImportError as exc:
        raise ImportError(
            "RobotWin norm computation requires pandas or pyarrow in this environment."
        ) from exc


def _to_numpy_column(table, key: str, dtype=np.float32) -> np.ndarray:
    if key not in table:
        raise KeyError(f"Missing parquet column `{key}`")
    values = table[key].to_numpy()
    return np.asarray([np.asarray(v, dtype=dtype) for v in values], dtype=dtype)


def _to_scalar_column(table, key: str, dtype=np.float32) -> np.ndarray:
    if key not in table:
        raise KeyError(f"Missing parquet column `{key}`")
    return np.asarray(table[key].to_numpy(), dtype=dtype).reshape(-1, 1)


def _read_compact_ee(table) -> np.ndarray:
    return np.concatenate(
        [
            _to_numpy_column(table, "endpose.left_endpose"),
            _to_scalar_column(table, "endpose.left_gripper"),
            _to_numpy_column(table, "endpose.right_endpose"),
            _to_scalar_column(table, "endpose.right_gripper"),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _read_joint(table) -> np.ndarray:
    joint = _to_numpy_column(table, "joint_action.vector")
    if joint.ndim != 2 or joint.shape[1] != 14:
        raise ValueError(f"Expected joint_action.vector shape [N, 14], got {joint.shape}")
    return joint.astype(np.float32, copy=False)


def _relative_pose(pose_7d: np.ndarray, anchor_pose_7d: np.ndarray) -> np.ndarray:
    pose_7d = np.asarray(pose_7d, dtype=np.float32).reshape(-1, 7)
    anchor_pose_7d = np.asarray(anchor_pose_7d, dtype=np.float32).reshape(7)
    rot = R.from_quat(pose_7d[:, 3:7])
    anchor_rot = R.from_quat(np.tile(anchor_pose_7d[None, 3:7], (pose_7d.shape[0], 1)))
    rel_xyz = pose_7d[:, :3] - anchor_pose_7d[None, :3]
    rel_quat = (anchor_rot.inv() * rot).as_quat()
    return np.concatenate([rel_xyz, rel_quat], axis=1).astype(np.float32, copy=False)


def _relative_compact_ee(action_16d: np.ndarray, anchor_16d: np.ndarray) -> np.ndarray:
    action_16d = np.asarray(action_16d, dtype=np.float32).reshape(-1, 16)
    anchor_16d = np.asarray(anchor_16d, dtype=np.float32).reshape(16)
    return np.concatenate(
        [
            _relative_pose(action_16d[:, :7], anchor_16d[:7]),
            action_16d[:, 7:8],
            _relative_pose(action_16d[:, 8:15], anchor_16d[8:15]),
            action_16d[:, 15:16],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _compact16_to_model30(compact: np.ndarray, inverse_used_action_channel_ids: list[int]) -> np.ndarray:
    compact = np.asarray(compact, dtype=np.float32)
    if compact.ndim != 2 or compact.shape[1] != 16:
        raise ValueError(f"Expected compact action shape [N, 16], got {compact.shape}")
    compact_padded = np.pad(compact, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
    return compact_padded[:, inverse_used_action_channel_ids].astype(np.float32, copy=False)


def _iter_repo_paths(dataset_path: str, single_task: str | None) -> list[Path]:
    dataset_root = Path(dataset_path)
    repos = sorted(path.parent.parent for path in dataset_root.rglob("meta/info.json"))
    if single_task:
        repos = [p for p in repos if p.parent.parent.name == single_task or p.name.startswith(f"{single_task}-")]
    return repos


def _iter_episodes(repos: list[Path]) -> list[dict]:
    episodes = []
    for repo in repos:
        episodes_jsonl = repo / "meta" / "episodes.jsonl"
        if not episodes_jsonl.exists():
            continue
        for record in _read_jsonl(episodes_jsonl):
            data_path = repo / record["data_path"]
            length = int(record.get("length", record.get("num_frames", 0)))
            if length <= 0:
                continue
            episodes.append(
                {
                    "repo": repo,
                    "data_path": data_path,
                    "length": length,
                    "episode_index": record.get("episode_index"),
                    "tasks": record.get("tasks", []),
                }
            )
    return episodes


def _valid_timestep_bounds(num_frames: int, history_len: int, history_stride: int, chunk_size: int) -> tuple[int, int] | None:
    min_t = (history_len - 1) * history_stride
    max_t = num_frames - chunk_size
    if max_t < min_t:
        return None
    return min_t, max_t


def _sample_relative_timesteps(
    num_frames: int,
    history_len: int,
    history_stride: int,
    chunk_size: int,
    windows_per_episode_stride: int,
    max_windows_per_episode: int,
) -> list[int]:
    bounds = _valid_timestep_bounds(num_frames, history_len, history_stride, chunk_size)
    if bounds is None:
        return []
    min_t, max_t = bounds
    num_valid = max_t - min_t + 1
    num_samples = int(np.ceil(num_valid / max(1, windows_per_episode_stride)))
    num_samples = min(max(1, max_windows_per_episode), max(1, num_samples))
    if num_samples == 1:
        return [(min_t + max_t) // 2]
    return [int(v) for v in np.linspace(min_t, max_t, num=num_samples, dtype=np.int64)]


def _format_python_list(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.10g}" for v in values.tolist()) + "]"


def _channel_quantiles(array: np.ndarray, q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray]:
    q01 = np.empty(array.shape[1], dtype=np.float32)
    q99 = np.empty(array.shape[1], dtype=np.float32)
    for channel in range(array.shape[1]):
        q01[channel] = np.quantile(array[:, channel], q_low)
        q99[channel] = np.quantile(array[:, channel], q_high)
    return q01, q99


def _finalize_ee_stats(q01: np.ndarray, q99: np.ndarray, fixed_quat_range: bool) -> tuple[np.ndarray, np.ndarray]:
    q01 = np.asarray(q01, dtype=np.float32).copy()
    q99 = np.asarray(q99, dtype=np.float32).copy()
    q01[DUMMY_CHANNELS_30D] = 0.0
    q99[DUMMY_CHANNELS_30D] = 1.0
    if fixed_quat_range:
        q01[QUAT_CHANNELS_30D] = -1.0
        q99[QUAT_CHANNELS_30D] = 1.0
    return q01, q99


def _write_mode_block(name: str, stats: dict) -> None:
    print(f"    \"{name}\": {{")
    print(f"        \"q01\": {_format_python_list(np.asarray(stats['q01'], dtype=np.float32))},")
    print(f"        \"q99\": {_format_python_list(np.asarray(stats['q99'], dtype=np.float32))},")
    print("    },")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="vga_robotwin_train")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--single-task", default=None)
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--history-frame-stride", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--windows-per-episode-stride", type=int, default=None)
    parser.add_argument("--max-windows-per-episode", type=int, default=None)
    parser.add_argument("--fixed-quat-range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    config = VA_CONFIGS[args.config_name]
    dataset_path = args.dataset_path if args.dataset_path is not None else config.dataset_path
    single_task = args.single_task if args.single_task is not None else getattr(config, "single_task", None)
    history_len = int(args.history_len or getattr(config, "history_len", 4))
    history_stride = int(args.history_frame_stride or getattr(config, "history_frame_stride", 1))
    chunk_size = int(args.chunk_size or getattr(config, "chunk_size", 8))
    windows_stride = int(
        args.windows_per_episode_stride
        or getattr(config, "robotwin_lerobot_windows_per_episode_stride", 32)
    )
    max_windows = int(
        args.max_windows_per_episode
        or getattr(config, "robotwin_lerobot_max_windows_per_episode", 8)
    )
    inverse_ids = list(config.inverse_used_action_channel_ids)

    repos = _iter_repo_paths(dataset_path, single_task)
    if len(repos) == 0:
        raise FileNotFoundError(f"No LeRobot repositories with meta/info.json found under {dataset_path!r}")
    episodes = _iter_episodes(repos)
    if len(episodes) == 0:
        raise RuntimeError("No valid RobotWin LeRobot episodes were found.")

    total_rows = int(sum(ep["length"] for ep in episodes))
    relative_rows = 0
    relative_timestep_cache = []
    for ep in episodes:
        timesteps = _sample_relative_timesteps(
            ep["length"],
            history_len,
            history_stride,
            chunk_size,
            windows_stride,
            max_windows,
        )
        relative_timestep_cache.append(timesteps)
        relative_rows += len(timesteps) * (history_len + chunk_size)
    if relative_rows == 0:
        raise RuntimeError("No valid windows available for relative EE statistics.")

    temp_root = Path(tempfile.mkdtemp(prefix="robotwin_norm_"))
    try:
        ee_abs_16 = np.memmap(temp_root / "ee_abs_16.dat", mode="w+", dtype=np.float32, shape=(total_rows, 16))
        ee_rel_16 = np.memmap(temp_root / "ee_rel_16.dat", mode="w+", dtype=np.float32, shape=(relative_rows, 16))
        joint_abs = np.memmap(temp_root / "joint_abs_14.dat", mode="w+", dtype=np.float32, shape=(total_rows, 14))
        joint_delta = np.memmap(temp_root / "joint_delta_14.dat", mode="w+", dtype=np.float32, shape=(total_rows, 14))

        abs_offset = 0
        rel_offset = 0
        for ep, rel_timesteps in tqdm(list(zip(episodes, relative_timestep_cache)), desc="Scanning RobotWin parquet"):
            table = _load_parquet_table(ep["data_path"], columns=ACTION_COLUMNS)
            ee = _read_compact_ee(table)
            joint = _read_joint(table)
            n = min(int(ep["length"]), ee.shape[0], joint.shape[0])
            if n <= 0:
                continue
            ee = ee[:n]
            joint = joint[:n]

            ee_abs_16[abs_offset : abs_offset + n] = ee
            joint_abs[abs_offset : abs_offset + n] = joint
            delta = np.zeros_like(joint, dtype=np.float32)
            delta[:-1] = joint[1:] - joint[:-1]
            joint_delta[abs_offset : abs_offset + n] = delta
            abs_offset += n

            for timestep in rel_timesteps:
                if timestep >= n:
                    continue
                history_indices = [
                    timestep - (history_len - 1 - i) * history_stride
                    for i in range(history_len)
                ]
                future_indices = list(range(timestep, min(timestep + chunk_size, n)))
                if len(future_indices) != chunk_size or min(history_indices) < 0:
                    continue
                selected_indices = history_indices + future_indices
                anchor = ee[history_indices[0]]
                rel = _relative_compact_ee(ee[selected_indices], anchor)
                ee_rel_16[rel_offset : rel_offset + rel.shape[0]] = rel
                rel_offset += rel.shape[0]

        ee_abs_16.flush()
        ee_rel_16.flush()
        joint_abs.flush()
        joint_delta.flush()

        ee_abs_30 = _compact16_to_model30(ee_abs_16[:abs_offset], inverse_ids)
        ee_rel_30 = _compact16_to_model30(ee_rel_16[:rel_offset], inverse_ids)

        ee_abs_q01, ee_abs_q99 = _finalize_ee_stats(
            *_channel_quantiles(ee_abs_30, args.q_low, args.q_high),
            fixed_quat_range=args.fixed_quat_range,
        )
        ee_rel_q01, ee_rel_q99 = _finalize_ee_stats(
            *_channel_quantiles(ee_rel_30, args.q_low, args.q_high),
            fixed_quat_range=args.fixed_quat_range,
        )
        joint_abs_q01, joint_abs_q99 = _channel_quantiles(joint_abs[:abs_offset], args.q_low, args.q_high)
        joint_delta_q01, joint_delta_q99 = _channel_quantiles(joint_delta[:abs_offset], args.q_low, args.q_high)

        stats_by_mode = {
            "ee_absolute": {"q01": ee_abs_q01.tolist(), "q99": ee_abs_q99.tolist()},
            "ee_relative": {"q01": ee_rel_q01.tolist(), "q99": ee_rel_q99.tolist()},
            "joint_absolute": {"q01": joint_abs_q01.tolist(), "q99": joint_abs_q99.tolist()},
            "joint_delta": {"q01": joint_delta_q01.tolist(), "q99": joint_delta_q99.tolist()},
        }
        result = {
            "dataset_path": str(dataset_path),
            "single_task": single_task,
            "num_repos": len(repos),
            "num_episodes": len(episodes),
            "num_rows": int(abs_offset),
            "relative_num_rows": int(rel_offset),
            "q_low": float(args.q_low),
            "q_high": float(args.q_high),
            "fixed_quat_range": bool(args.fixed_quat_range),
            "relative_sampling": {
                "history_len": history_len,
                "history_frame_stride": history_stride,
                "chunk_size": chunk_size,
                "windows_per_episode_stride": windows_stride,
                "max_windows_per_episode": max_windows,
            },
            "norm_stats_by_action_mode": stats_by_mode,
        }

        print("\n# RobotWin normalization stats")
        print(f"# repos={len(repos)}, episodes={len(episodes)}, rows={abs_offset}, relative_rows={rel_offset}")
        print("va_robotwin_cfg.norm_stats_by_action_mode = {")
        for key in ("ee_absolute", "ee_relative", "joint_absolute", "joint_delta"):
            _write_mode_block(key, stats_by_mode[key])
        print("}")

        if args.json_out:
            json_path = Path(args.json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\nWrote JSON stats to {json_path}")
    finally:
        if args.keep_temp:
            print(f"Kept temporary memmap directory: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
