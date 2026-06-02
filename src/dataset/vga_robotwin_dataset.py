# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from dataset.lerobot_latent_dataset import get_relative_compact_action
from utils import logger
from utils.text_embedding import encode_prompt


DEFAULT_CAMERA_KEYS = (
    "front_camera",
    "head_camera",
    "left_camera",
    "right_camera",
    "side_camera",
)


def _episode_index_from_name(path: Path) -> int:
    match = re.search(r"episode(\d+)\.hdf5$", path.name)
    if match is None:
        raise ValueError(f"Unable to parse episode index from `{path.name}`")
    return int(match.group(1))


class MultiVGARobotwinDataset(Dataset):
    """Local HDF5 robotwin dataset for VGA training.

    Each item returns the same sample dict shape as ``MultiLatentLeRobotDataset``:
    ``images``, ``actions``, ``actions_mask``, ``action_chunk``,
    ``action_chunk_mask``, ``state``, ``pred_frame_idx``, ``num_frames``,
    ``text``, and optionally ``text_emb``.
    """

    def __init__(self, config, num_init_worker=None):
        del num_init_worker

        self.config = config
        self.dataset_path = Path(getattr(config, "dataset_path"))
        self.data_dir = self.dataset_path / "data"
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Missing HDF5 data directory: {self.data_dir}")

        self.image_height = int(getattr(config, "image_height", 224))
        self.image_width = int(getattr(config, "image_width", 224))
        self.history_len = max(1, int(getattr(config, "history_len", 1)))
        self.history_stride = max(1, int(getattr(config, "history_frame_stride", 1)))
        self.chunk_size = max(1, int(getattr(config, "chunk_size", 1)))
        self.multi_view_image_mode = str(getattr(config, "multi_view_image_mode", "vertical"))
        self.action_representation = str(getattr(config, "action_representation", "absolute")).lower()
        self.rdt_action_target_source = str(getattr(config, "rdt_action_target_source", "endpose")).lower()
        if self.rdt_action_target_source not in {"endpose", "expert_target"}:
            raise ValueError(
                f"Unsupported rdt_action_target_source `{self.rdt_action_target_source}`. "
                "Expected 'endpose' or 'expert_target'."
            )
        self.encode_text_in_dataloader = bool(getattr(config, "encode_text_in_dataloader", False))
        self.cfg_prob = float(getattr(config, "cfg_prob", 0.0))
        self.use_marked_rgb = bool(getattr(config, "use_expert_marked_rgb", False))

        self.obs_cam_keys = list(getattr(config, "obs_cam_keys", list(DEFAULT_CAMERA_KEYS)))
        self.separate_history_current_obs_views = bool(
            getattr(config, "separate_history_current_obs_views", False)
        )
        self.current_obs_cam_keys = list(getattr(config, "current_obs_cam_keys", self.obs_cam_keys))
        self.history_obs_cam_keys = list(getattr(config, "history_obs_cam_keys", self.current_obs_cam_keys))
        self.required_obs_cam_keys = list(dict.fromkeys(self.history_obs_cam_keys + self.current_obs_cam_keys))
        self.view_position_cam_keys = list(
            dict.fromkeys(list(getattr(config, "view_position_cam_keys", self.obs_cam_keys)) + self.required_obs_cam_keys)
        )
        self.available_cam_keys = list(DEFAULT_CAMERA_KEYS)

        norm_stat = getattr(config, "norm_stat", None)
        if not norm_stat or "q01" not in norm_stat or "q99" not in norm_stat:
            raise ValueError("config.norm_stat with q01/q99 is required for VGA robotwin dataset")
        self.q01 = np.array(norm_stat["q01"], dtype=np.float32)[None]
        self.q99 = np.array(norm_stat["q99"], dtype=np.float32)[None]

        self._text_embedding_cache = {}
        self._empty_text_embedding = None
        self._file_cache = {}
        self._episodes = self._scan_episodes()
        self._samples = self._build_sample_index()

        logger.info(
            f"Loaded VGA robotwin HDF5 dataset from {self.data_dir} "
            f"with {len(self._episodes)} episodes and {len(self._samples)} samples."
        )

    def _scan_episodes(self):
        episode_paths = sorted(self.data_dir.glob("episode*.hdf5"), key=_episode_index_from_name)
        if len(episode_paths) == 0:
            raise FileNotFoundError(f"No episode*.hdf5 files found under {self.data_dir}")

        requested_episode = getattr(self.config, "single_trajectory_episode_index", None)
        if requested_episode is None:
            requested_episode = getattr(self.config, "single_trajectory_repo_id", None)
        requested_episode_index = None
        if requested_episode is not None:
            requested_str = str(requested_episode)
            if requested_str.isdigit():
                requested_episode_index = int(requested_str)
            else:
                match = re.search(r"episode(\d+)", Path(requested_str).name)
                if match is not None:
                    requested_episode_index = int(match.group(1))

        episodes = []
        for path in episode_paths:
            episode_index = _episode_index_from_name(path)
            if requested_episode_index is not None and episode_index != requested_episode_index:
                continue

            with h5py.File(path, "r") as f:
                num_frames = int(f["joint_action"]["vector"].shape[0])
                obs_keys = list(f["observation"].keys())
                required_obs_cam_keys = self.required_obs_cam_keys if self.separate_history_current_obs_views else self.obs_cam_keys
                if not set(required_obs_cam_keys).issubset(set(obs_keys)):
                    if set(DEFAULT_CAMERA_KEYS).issubset(set(obs_keys)):
                        self.obs_cam_keys = list(DEFAULT_CAMERA_KEYS)
                        self.current_obs_cam_keys = list(getattr(self.config, "current_obs_cam_keys", self.obs_cam_keys))
                        self.history_obs_cam_keys = list(getattr(self.config, "history_obs_cam_keys", self.current_obs_cam_keys))
                        self.required_obs_cam_keys = list(
                            dict.fromkeys(self.history_obs_cam_keys + self.current_obs_cam_keys)
                        )
                        self.view_position_cam_keys = list(
                            dict.fromkeys(
                                list(getattr(self.config, "view_position_cam_keys", self.obs_cam_keys))
                                + self.required_obs_cam_keys
                            )
                        )
                    else:
                        raise KeyError(
                            f"Episode {path} is missing expected camera keys. "
                            f"Found={obs_keys}, expected={required_obs_cam_keys}"
                        )
                valid = True
                if "expert_target" in f and "left" in f["expert_target"]:
                    valid = bool(f["expert_target"]["left"]["valid"][:].all()) and bool(
                        f["expert_target"]["right"]["valid"][:].all()
                    )
            episodes.append(
                {
                    "path": path,
                    "episode_index": episode_index,
                    "num_frames": num_frames,
                    "has_all_valid_targets": valid,
                }
            )

        if len(episodes) == 0:
            raise RuntimeError("No HDF5 episodes matched the current filter.")
        return episodes

    def _build_sample_index(self):
        samples = []
        required_frames_for_chunk = max(1, (self.chunk_size + 1 - 1) // 1)
        min_t = (self.history_len - 1) * self.history_stride

        for episode_id, episode in enumerate(self._episodes):
            num_frames = int(episode["num_frames"])
            max_t = num_frames - required_frames_for_chunk
            if max_t < min_t:
                continue
            for timestep in range(min_t, max_t + 1):
                samples.append((episode_id, timestep))
        if len(samples) == 0:
            raise RuntimeError("No valid training windows were found in the HDF5 dataset.")
        return samples

    def _get_file(self, path: Path):
        cached = self._file_cache.get(str(path), None)
        if cached is not None:
            return cached
        handle = h5py.File(path, "r")
        self._file_cache[str(path)] = handle
        return handle

    @staticmethod
    def _decode_image_bytes(blob):
        if not isinstance(blob, (bytes, bytearray)):
            blob = bytes(blob)
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _resize_pad_frames(self, frames: torch.Tensor):
        # frames: [F, 3, H, W]
        _, _, h, w = frames.shape
        scale = min(self.image_height / h, self.image_width / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_h = self.image_height - new_h
        pad_w = self.image_width - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

    def _merge_multi_view_images(self, image_lst):
        if self.multi_view_image_mode == "vertical":
            return torch.cat(image_lst, dim=2)
        if self.multi_view_image_mode == "frame":
            return rearrange(torch.stack(image_lst, dim=1), "f v c h w -> (f v) c h w")
        if self.multi_view_image_mode == "first":
            return image_lst[0]
        raise ValueError(
            f"Unsupported multi_view_image_mode `{self.multi_view_image_mode}`. "
            "Expected one of ['vertical', 'frame', 'first']."
        )

    def _align_actions_with_multi_view_mode(self, actions, actions_mask):
        if self.separate_history_current_obs_views:
            return actions, actions_mask
        if self.multi_view_image_mode != "frame":
            return actions, actions_mask
        num_views = len(self.obs_cam_keys)
        actions = actions.repeat_interleave(num_views, dim=1)
        actions_mask = actions_mask.repeat_interleave(num_views, dim=1)
        return actions, actions_mask

    def _align_state_with_multi_view_mode(self, state):
        if state is None or self.separate_history_current_obs_views or self.multi_view_image_mode != "frame":
            return state
        num_views = len(self.obs_cam_keys)
        return state.repeat_interleave(num_views, dim=1)

    def _read_compact_sequence(self, f, group_name: str):
        if group_name == "endpose":
            left_pose = np.asarray(f[f"{group_name}/left_endpose"][...], dtype=np.float32)
            left_gripper = np.asarray(f[f"{group_name}/left_gripper"][...], dtype=np.float32)[..., None]
            right_pose = np.asarray(f[f"{group_name}/right_endpose"][...], dtype=np.float32)
            right_gripper = np.asarray(f[f"{group_name}/right_gripper"][...], dtype=np.float32)[..., None]
            valid = np.ones((left_pose.shape[0], 1), dtype=bool)
        else:
            left_pose = np.asarray(f[f"{group_name}/left/pose_7d"][...], dtype=np.float32)
            left_gripper = np.asarray(f[f"{group_name}/left/gripper"][...], dtype=np.float32)
            if left_gripper.ndim == 1:
                left_gripper = left_gripper[..., None]
            right_pose = np.asarray(f[f"{group_name}/right/pose_7d"][...], dtype=np.float32)
            right_gripper = np.asarray(f[f"{group_name}/right/gripper"][...], dtype=np.float32)
            if right_gripper.ndim == 1:
                right_gripper = right_gripper[..., None]
            valid_left = np.asarray(f[f"{group_name}/left/valid"][...], dtype=bool).reshape(-1, 1)
            valid_right = np.asarray(f[f"{group_name}/right/valid"][...], dtype=bool).reshape(-1, 1)
            valid = np.logical_and(valid_left, valid_right)

        compact = np.concatenate([left_pose, left_gripper, right_pose, right_gripper], axis=1)
        return compact, valid

    def _normalize_compact_action(self, action_16d):
        action_16d = np.asarray(action_16d, dtype=np.float32).reshape(-1, 16)
        action_mask = np.ones_like(action_16d, dtype=bool)
        action_padded = np.pad(action_16d, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode="constant", constant_values=0)

        action_aligned = action_padded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        action_aligned *= action_mask_aligned
        return (
            torch.from_numpy(action_aligned).float(),
            torch.from_numpy(action_mask_aligned).bool(),
        )

    def _encode_raw_text_embedding(self, text: str):
        cached = self._text_embedding_cache.get(text)
        if cached is not None:
            return cached
        text_embedding = encode_prompt(text, self.config, device="cpu", dtype=torch.float32)
        if text_embedding is None:
            return None
        if text_embedding.ndim != 3:
            raise ValueError(
                f"Expected encoded text embedding [B, L, D], got {tuple(text_embedding.shape)} for text `{text}`."
            )
        text_embedding = text_embedding.squeeze(0).cpu().float().contiguous()
        self._text_embedding_cache[text] = text_embedding
        return text_embedding

    def _get_empty_text_embedding(self, template_text_embedding: torch.Tensor):
        if self._empty_text_embedding is None:
            self._empty_text_embedding = self._encode_raw_text_embedding("")
        if self._empty_text_embedding is None:
            return torch.zeros_like(template_text_embedding)
        source = self._empty_text_embedding
        if source.shape == template_text_embedding.shape:
            return source.to(dtype=template_text_embedding.dtype)
        if (
            source.ndim + 1 == template_text_embedding.ndim
            and tuple(source.shape) == tuple(template_text_embedding.shape[1:])
        ):
            return source.unsqueeze(0).expand_as(template_text_embedding).to(dtype=template_text_embedding.dtype)
        return torch.zeros_like(template_text_embedding)

    def _get_text_embedding(self, text: str):
        text = str(text or "")
        if torch.rand(1).item() < self.cfg_prob:
            template = self._encode_raw_text_embedding(text)
            if template is None:
                return None
            return self._get_empty_text_embedding(template)
        return self._encode_raw_text_embedding(text)

    def _load_window_images(self, f, frame_indices):
        image_mode = "rgb_expert_marked" if self.use_marked_rgb else "rgb"
        if self.separate_history_current_obs_views:
            frames = []
            image_time_ids = []
            image_view_ids = []
            current_start = len(frame_indices) - 1
            for local_idx, frame_idx in enumerate(frame_indices):
                cam_keys = self.current_obs_cam_keys if local_idx >= current_start else self.history_obs_cam_keys
                for cam in cam_keys:
                    frame = self._decode_image_bytes(f[f"observation/{cam}/{image_mode}"][frame_idx])
                    frame = self._resize_pad_frames(frame.unsqueeze(0))[0]
                    frames.append(frame)
                    image_time_ids.append(local_idx)
                    image_view_ids.append(self.view_position_cam_keys.index(cam))
            merged = torch.stack(frames, dim=0)  # [S, 3, H, W]
            return (
                merged.permute(1, 0, 2, 3).contiguous(),
                len(self.current_obs_cam_keys),
                torch.as_tensor(image_time_ids, dtype=torch.long),
                torch.as_tensor(image_view_ids, dtype=torch.long),
            )

        image_lst = []
        for cam in self.obs_cam_keys:
            ds = f[f"observation/{cam}/{image_mode}"]
            frames = []
            for frame_idx in frame_indices:
                frame = self._decode_image_bytes(ds[frame_idx])
                frames.append(frame)
            frames = torch.stack(frames, dim=0)  # [F, 3, H, W]
            frames = self._resize_pad_frames(frames)
            image_lst.append(frames)
        merged = self._merge_multi_view_images(image_lst)
        current_frame_count = len(self.obs_cam_keys) if self.multi_view_image_mode == "frame" else 1
        return merged.permute(1, 0, 2, 3).contiguous(), current_frame_count, None, None

    def _build_sample_from_episode(self, episode, data_timestep: int):
        f = self._get_file(episode["path"])
        num_frames = int(episode["num_frames"])
        history_indices = [data_timestep - (self.history_len - 1 - i) * self.history_stride for i in range(self.history_len)]
        if min(history_indices) < 0:
            raise IndexError(f"Invalid history indices for timestep={data_timestep} in episode={episode['path']}")
        if data_timestep + self.chunk_size > num_frames:
            raise IndexError(f"Not enough future frames for timestep={data_timestep} in episode={episode['path']}")

        raw_ee_targets_16d, ee_target_valid = self._read_compact_sequence(f, "expert_target")
        raw_state_16d, _ = self._read_compact_sequence(f, "endpose")
        if self.rdt_action_target_source == "expert_target":
            raw_action_targets_16d = raw_ee_targets_16d
            action_valid = ee_target_valid
        else:
            raw_action_targets_16d = raw_state_16d
            action_valid = np.ones((raw_state_16d.shape[0], 1), dtype=bool)

        if self.action_representation == "relative":
            anchor_pose = raw_state_16d[history_indices[0]]
            action_model_flat = get_relative_compact_action(raw_action_targets_16d, anchor_pose)
            state_model_flat = get_relative_compact_action(raw_state_16d, anchor_pose)
        elif self.action_representation == "absolute":
            action_model_flat = raw_action_targets_16d
            state_model_flat = raw_state_16d
        else:
            raise ValueError(
                f"Unsupported action_representation `{self.action_representation}`. "
                "Expected 'relative' or 'absolute'."
            )

        action_norm_flat, action_mask_flat = self._normalize_compact_action(action_model_flat)
        action_valid_mask = torch.from_numpy(np.asarray(action_valid, dtype=bool)).reshape(-1, 1)
        action_valid_mask = action_valid_mask.expand(-1, action_mask_flat.shape[1])
        action_norm_flat = action_norm_flat * action_valid_mask.to(action_norm_flat.dtype)
        action_mask_flat = action_mask_flat & action_valid_mask
        state_norm_flat, state_mask_flat = self._normalize_compact_action(state_model_flat)

        if self.action_representation == "relative":
            state_model = state_model_flat[data_timestep]
        else:
            state_model = state_model_flat[data_timestep]
        state_norm, state_mask = self._normalize_compact_action(state_model[None])
        state = state_norm[0]
        state_mask = state_mask[0]

        action_window = state_norm_flat[history_indices].T.unsqueeze(-1).unsqueeze(-1)
        action_window_mask = state_mask_flat[history_indices].T.unsqueeze(-1).unsqueeze(-1)

        future_action_chunk = action_norm_flat[data_timestep : data_timestep + self.chunk_size].T
        future_action_chunk_mask = action_mask_flat[data_timestep : data_timestep + self.chunk_size].T
        if future_action_chunk.shape[-1] < self.chunk_size:
            chunk_pad = torch.zeros(
                (future_action_chunk.shape[0], self.chunk_size - future_action_chunk.shape[-1]),
                dtype=future_action_chunk.dtype,
            )
            chunk_mask_pad = torch.zeros(
                (future_action_chunk_mask.shape[0], self.chunk_size - future_action_chunk_mask.shape[-1]),
                dtype=torch.bool,
            )
            future_action_chunk = torch.cat([future_action_chunk, chunk_pad], dim=1)
            future_action_chunk_mask = torch.cat([future_action_chunk_mask, chunk_mask_pad], dim=1)

        action_chunk = torch.cat([state.unsqueeze(-1), future_action_chunk], dim=1)
        action_chunk_mask = torch.cat([state_mask.unsqueeze(-1), future_action_chunk_mask], dim=1)

        images, current_image_frame_count, image_time_ids, image_view_ids = self._load_window_images(f, history_indices)
        ee_target = torch.from_numpy(raw_ee_targets_16d[data_timestep].reshape(2, 8)).float()
        ee_target_valid = torch.from_numpy(np.asarray(ee_target_valid[data_timestep], dtype=bool)).reshape(1)
        ee_target_valid = ee_target_valid.expand(2)

        out_dict = {
            "images": images,
            "actions": action_window,
            "images_mask": torch.ones_like(images, dtype=torch.bool),
            "actions_mask": action_window_mask,
            "action_chunk": action_chunk,
            "action_chunk_mask": action_chunk_mask,
            "state": state,
            "ee_target": ee_target,
            "ee_target_valid": ee_target_valid,
            "pred_frame_idx": torch.tensor(self.history_len - 1, dtype=torch.long),
            "num_frames": torch.tensor(self.history_len, dtype=torch.long),
            "current_image_frame_count": torch.tensor(current_image_frame_count, dtype=torch.long),
            "text": "",
        }
        if image_time_ids is not None and image_view_ids is not None:
            out_dict["image_time_ids"] = image_time_ids
            out_dict["image_view_ids"] = image_view_ids

        if self.encode_text_in_dataloader:
            out_dict["text_emb"] = self._get_text_embedding(out_dict["text"])

        out_dict["actions"], out_dict["actions_mask"] = self._align_actions_with_multi_view_mode(
            out_dict["actions"],
            out_dict["actions_mask"],
        )
        return out_dict

    def __getitem__(self, idx) -> dict:
        episode_id, data_timestep = self._samples[idx]
        episode = self._episodes[episode_id]
        return self._build_sample_from_episode(episode, data_timestep)

    def __len__(self):
        return len(self._samples)

    def __del__(self):
        for handle in self._file_cache.values():
            try:
                handle.close()
            except Exception:
                pass
