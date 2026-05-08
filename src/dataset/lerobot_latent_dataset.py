# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import get_context
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

from utils import logger
from utils.text_embedding import encode_prompt
def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config,
                                      num_init_worker=128,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if config.single_task:
        repo_list = [
            v for v in repo_list
            if Path(v).name.startswith(f"{config.single_task}-")
        ]
        logger.info(f"Found {len(repo_list)} repositories with info.json in {config.dataset_path} for task {config.single_task}.")
    # repo_list = repo_list[:2]
    if num_init_worker is None or int(num_init_worker) <= 1:
        for repo in repo_list:
            datasets_out_lst.append(construct_func(repo))
        return datasets_out_lst

    mp_start_method = getattr(config, 'dataset_mp_start_method', 'spawn')
    pool_context = get_context(mp_start_method)
    with pool_context.Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
    # for repo in repo_list:
    #     datasets_out_lst.append(construct_func(repo))
                
    return datasets_out_lst

def get_relative_pose(pose, anchor_pose=None):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    if anchor_pose is None:
        anchor_pose = pose[0]
    if torch.is_tensor(anchor_pose):
        anchor_pose = anchor_pose.detach().cpu().numpy()

    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(anchor_pose[None, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - anchor_pose[None, :3]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)


def get_relative_compact_action(action_16d, anchor_pose_16d):
    if torch.is_tensor(action_16d):
        action_16d = action_16d.detach().cpu().numpy()
    if torch.is_tensor(anchor_pose_16d):
        anchor_pose_16d = anchor_pose_16d.detach().cpu().numpy()
    action_16d = np.asarray(action_16d, dtype=np.float32).reshape(-1, 16)
    anchor_pose_16d = np.asarray(anchor_pose_16d, dtype=np.float32).reshape(16)

    left_action = get_relative_pose(action_16d[:, :7], anchor_pose_16d[:7]).numpy()
    right_action = get_relative_pose(action_16d[:, 8:15], anchor_pose_16d[8:15]).numpy()
    return np.concatenate(
        [left_action, action_16d[:, 7:8], right_action, action_16d[:, 15:16]],
        axis=1,
    )

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=None,
    ):
        if num_init_worker is None:
            num_init_worker = getattr(config, 'dataset_init_worker', 8)
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        # self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.image_path = Path(repo_id) / 'videos'
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.image_height, self.image_width = config.image_height, config.image_width
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_action_view = self.hf_dataset.with_format(
            columns=['action', 'observation.state'],
            output_all_columns=False,
        )
        self._text_embedding_cache = {}
        self._empty_text_embedding = None
        self.prefer_raw_text_for_text_embeddings = bool(
            getattr(config, "prefer_raw_text_for_text_embeddings", True)
        )
        self.encode_text_in_dataloader = bool(
            getattr(config, "encode_text_in_dataloader", False)
        )
        self._raw_text_warning_emitted = False
        self.parse_meta()

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)

        if bool(getattr(self.config, "single_trajectory", False)) and len(out) > 0:
            requested_ep = getattr(self.config, "single_trajectory_episode_index", None)
            if requested_ep is None:
                target_episode = int(out[0]["episode_index"])
            else:
                target_episode = int(requested_ep)
            filtered = [m for m in out if int(m["episode_index"]) == target_episode]
            if len(filtered) == 0:
                fallback_episode = int(out[0]["episode_index"])
                logger.warning(
                    f"single_trajectory=True but episode_index={target_episode} not found in repo {self.repo_id}; "
                    f"fallback to first available episode_index={fallback_episode}."
                )
                target_episode = fallback_episode
                filtered = [m for m in out if int(m["episode_index"]) == target_episode]
            logger.info(
                f"Dataset {self.repo_id}: single-trajectory mode enabled, selected episode_index={target_episode}, "
                f"segments={len(filtered)}"
            )
            out = filtered
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_action_view[start_frame:end_frame]
        actions = batch['action']
        if isinstance(actions, torch.Tensor):
            action_tensor = actions
        else:
            action_tensor = torch.as_tensor(np.asarray(actions))

        states = batch.get('observation.state', None)
        if states is not None:
            if isinstance(states, torch.Tensor):
                state_tensor = states
            else:
                state_tensor = torch.as_tensor(np.asarray(states))
        else:
            state_tensor = None

        out = {'action': action_tensor}
        if state_tensor is not None:
            out['observation.state'] = state_tensor
        return out

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)

    def _get_range_image_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        image_path = Path(self.image_path) / f"chunk-{episode_chunk:03d}"
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = image_path / key
            image_file = (
                cur_path / f"episode_{episode_index:06d}.mp4"
            )
            assert os.path.exists(image_file)
            # Decode only the needed frame slice to avoid holding full videos in worker memory.
            try:
                import av
                frames = []
                with av.open(str(image_file)) as container:
                    for frame_idx, frame in enumerate(container.decode(video=0)):
                        if frame_idx < start_frame:
                            continue
                        if frame_idx >= end_frame:
                            break
                        frames.append(frame.to_ndarray(format='rgb24'))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to decode video {image_file}. "
                    "OpenCV and PyAV both failed."
                ) from e

            if len(frames) == 0:
                raise RuntimeError(
                    f"Empty frame slice [{start_frame}:{end_frame}] for {image_file}. "
                    "Check action segment boundaries versus available frames."
                )

            image_data = torch.from_numpy(np.stack(frames)).float() / 255.0
            num_frames, height, width, channels = image_data.shape
            
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)

            out[key] = {
                "video": image_data.reshape(num_frames * height * width, channels),
                "video_num_frames": num_frames,
                "video_height": height,
                "video_width": width,
                "frame_ids": torch.arange(start_frame, start_frame + num_frames, dtype=torch.long),
                "text_emb": latent_data.get("text_emb", None)
            }
        
        return self._flatten_latent_dict(out)

    def _resolve_raw_text(self, meta: dict) -> str | None:
        action_text = str(meta.get('action_text', '') or '').strip()
        if action_text:
            return action_text

        for task in meta.get('tasks', []) or []:
            task_text = str(task or '').strip()
            if task_text:
                return task_text
        return None

    def _pool_text_embedding_tokens(self, text_emb):
        text_emb = torch.as_tensor(text_emb)
        if text_emb.ndim == 2 and text_emb.shape[0] > 1:
            return text_emb.mean(dim=0, keepdim=True)
        if text_emb.ndim == 3 and text_emb.shape[-2] > 1:
            return text_emb.mean(dim=-2, keepdim=True)
        return text_emb

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

    def _expand_text_embedding_like(self, source_embedding, target_embedding: torch.Tensor):
        if source_embedding is None:
            return None
        source_embedding = torch.as_tensor(source_embedding)
        target_embedding = torch.as_tensor(target_embedding)
        if tuple(source_embedding.shape) == tuple(target_embedding.shape):
            return source_embedding.to(dtype=target_embedding.dtype)
        if (
            source_embedding.ndim + 1 == target_embedding.ndim
            and tuple(source_embedding.shape) == tuple(target_embedding.shape[1:])
        ):
            return source_embedding.unsqueeze(0).expand(*target_embedding.shape).contiguous().to(
                dtype=target_embedding.dtype
            )
        if (
            source_embedding.ndim == target_embedding.ndim
            and source_embedding.shape[0] == 1
            and tuple(source_embedding.shape[1:]) == tuple(target_embedding.shape[1:])
        ):
            return source_embedding.expand(*target_embedding.shape).contiguous().to(
                dtype=target_embedding.dtype
            )
        return None

    def _get_empty_text_embedding(self, template_text_embedding: torch.Tensor):
        if self._empty_text_embedding is None:
            self._empty_text_embedding = self._encode_raw_text_embedding('')
        return self._expand_text_embedding_like(self._empty_text_embedding, template_text_embedding)

    def _get_cfg_text_embedding(self, text_emb):
        text_emb = self._pool_text_embedding_tokens(text_emb).float()
        if torch.rand(1).item() >= self.cfg_prob:
            return text_emb

        matched_encoded_empty = self._get_empty_text_embedding(text_emb)
        if matched_encoded_empty is not None:
            return matched_encoded_empty.float()

        return torch.zeros_like(text_emb)

    def _get_selected_text_embedding(self, meta: dict, data_dict: dict):
        raw_text = self._resolve_raw_text(meta)
        if self.prefer_raw_text_for_text_embeddings and raw_text:
            try:
                raw_text_embedding = self._encode_raw_text_embedding(raw_text)
            except Exception as exc:
                if not self._raw_text_warning_emitted:
                    logger.warning(
                        f"Failed to encode raw text `{raw_text}` for repo {self.repo_id}; "
                        f"falling back to precomputed text_emb. Error: {exc}"
                    )
                    self._raw_text_warning_emitted = True
            else:
                if raw_text_embedding is not None:
                    return self._get_cfg_text_embedding(raw_text_embedding)

        fallback_text_embedding = data_dict.get(f"{self.used_video_keys[0]}.text_emb")
        if fallback_text_embedding is None:
            if raw_text:
                raw_text_embedding = self._encode_raw_text_embedding(raw_text)
                if raw_text_embedding is not None:
                    return self._get_cfg_text_embedding(raw_text_embedding)
            raise ValueError(
                f"No text embedding source found for repo {self.repo_id}, episode {meta.get('episode_index')}."
            )

        return self._get_cfg_text_embedding(fallback_text_embedding)

    def _get_selected_text(self, meta: dict, data_dict: dict) -> str:
        raw_text = self._resolve_raw_text(meta)
        if raw_text:
            return raw_text
        fallback_text = data_dict.get(f"{self.used_video_keys[0]}.text", "")
        return str(fallback_text or "")
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        wrist_latent = torch.cat(latent_lst[1:], dim=2)
        cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)

        out_dict = dict(
            latents = cat_latent,
            text_emb = self._get_selected_text_embedding({}, data_dict),
        )
        print(f"lactent shape: {cat_latent.shape}")
        return out_dict

    def _merge_multi_view_images(self, image_lst):
        mode = getattr(self.config, 'multi_view_image_mode', 'vertical')
        if mode == 'vertical':
            return torch.cat(image_lst, dim=2)
        if mode == 'frame':
            return rearrange(torch.stack(image_lst, dim=1), 'f v c h w -> (f v) c h w')
        if mode == 'first':
            return image_lst[0]
        raise ValueError(
            f"Unsupported multi_view_image_mode `{mode}`. Expected one of ['vertical', 'frame', 'first']."
        )

    def _align_actions_with_multi_view_mode(self, actions, actions_mask):
        if getattr(self.config, 'multi_view_image_mode', 'vertical') != 'frame':
            return actions, actions_mask

        num_views = len(self.used_video_keys)
        actions = actions.repeat_interleave(num_views, dim=1)
        actions_mask = actions_mask.repeat_interleave(num_views, dim=1)
        return actions, actions_mask
    
    def _cat_video_images(self,
                          data_dict):
        image_lst = []
        for key in self.used_video_keys:
            image = data_dict[f"{key}.video"]
            image_num_frames = data_dict[f"{key}.video_num_frames"]
            image_height = data_dict[f"{key}.video_height"]
            image_width = data_dict[f"{key}.video_width"]
            image = rearrange(image, 
                                 '(f h w) c -> f c h w ', 
                                 f= image_num_frames, 
                                 h= image_height, 
                                 w= image_width)
            # resize and padding image to self.image_height and self.image_width, Does not distort the image content
            import torch.nn.functional as F

            # Resize and pad image to self.image_height and self.image_width without distortion
            # image: [num_frames * image_height * image_width, channels] -> [num_frames, channels, image_height, image_width]
            h = image_height
            w = image_width

            # Compute scale factor to fit within target size
            scale = min(self.image_height / h, self.image_width / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            # Resize
            image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # Pad to target size
            pad_h = self.image_height - new_h
            pad_w = self.image_width - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            image_lst.append(image)

        cat_image = self._merge_multi_view_images(image_lst)

        out_dict = dict(
            images = cat_image,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, image_frame_ids, action):
        del local_end_frame
        act_shift = int(image_frame_ids[0] - local_start_frame)
        action = np.asarray(action[act_shift:], dtype=np.float32)

        image_frame_num = len(image_frame_ids)
        if action.shape[0] < image_frame_num:
            raise ValueError(
                f"Not enough actions for image frames: actions={action.shape[0]}, images={image_frame_num}"
            )
        action = action[:image_frame_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == image_frame_num

        action_abs = rearrange(action, "f c -> c f 1 1")
        action_mask = rearrange(action_mask, "f c -> c f 1 1")
        return torch.from_numpy(action_abs).float(), torch.from_numpy(action_mask).bool()

    def _state_post_process(self, local_start_frame, local_end_frame, image_frame_ids, state):
        del local_end_frame
        if state is None:
            return None

        act_shift = int(image_frame_ids[0] - local_start_frame)
        state = np.asarray(state[act_shift:], dtype=np.float32)

        image_frame_num = len(image_frame_ids)
        if state.shape[0] < image_frame_num:
            raise ValueError(
                f"Not enough states for image frames: states={state.shape[0]}, images={image_frame_num}"
            )
        state = state[:image_frame_num]

        return torch.from_numpy(state).float().transpose(0, 1)  # [16, F]

    def _align_state_with_multi_view_mode(self, states):
        if states is None or getattr(self.config, 'multi_view_image_mode', 'vertical') != 'frame':
            return states
        num_views = len(self.used_video_keys)
        return states.repeat_interleave(num_views, dim=1)

    def _normalize_relative_compact_action(self, relative_action_16d):
        relative_action_16d = np.asarray(relative_action_16d, dtype=np.float32).reshape(-1, 16)
        action_mask = np.ones_like(relative_action_16d, dtype=bool)
        action_padded = np.pad(relative_action_16d, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_padded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned *= action_mask_aligned
        return (
            torch.from_numpy(action_aligned).float(),
            torch.from_numpy(action_mask_aligned).bool(),
        )

    def _sample_window_and_chunk(self, images, actions, states=None):
        """Sample one image frame and an action chunk starting at the same timestep."""
        chunk_size = int(getattr(self.config, 'chunk_size', 1))
        history_len = max(1, int(getattr(self.config, 'history_len', 1)))
        history_stride = max(1, int(getattr(self.config, 'history_frame_stride', 1)))

        c_img, f_total, h, w = images.shape
        c_abs, f_total_action, n, one = actions.shape
        if f_total != f_total_action:
            raise ValueError(f"images/actions frame count mismatch: {f_total} vs {f_total_action}")
        if c_abs != 16:
            raise ValueError(f"Expected absolute compact 16D action sequence, got {c_abs} channels")
        if f_total <= 0:
            raise ValueError("No valid frames available after preprocessing")

        required_frames_for_chunk = max(1, (chunk_size + n - 1) // n)
        min_t = min(f_total - 1, (history_len - 1) * history_stride)
        max_t = f_total - required_frames_for_chunk

        if max_t >= min_t:
            data_timestep = int(np.random.randint(min_t, max_t + 1))
        else:
            data_timestep = max(0, f_total - 1)

        raw_indices = [
            data_timestep - (history_len - 1 - i) * history_stride
            for i in range(history_len)
        ]
        history_indices = torch.as_tensor(
            [max(0, min(f_total - 1, idx)) for idx in raw_indices],
            dtype=torch.long,
        )

        images_window = images.index_select(1, history_indices)

        anchor_idx = int(history_indices[0].item())
        if states is not None:
            anchor_pose = states[:, anchor_idx]
        else:
            anchor_pose = actions[:, anchor_idx, 0, 0]

        action_abs_flat = rearrange(actions, 'c f n 1 -> (f n) c')
        rel_action_flat = get_relative_compact_action(action_abs_flat, anchor_pose)
        rel_action_norm_flat, rel_action_mask_flat = self._normalize_relative_compact_action(rel_action_flat)
        actions = rearrange(rel_action_norm_flat, '(f n) c -> c f n 1', f=f_total, n=n)
        base_action_mask = rearrange(rel_action_mask_flat, '(f n) c -> c f n 1', f=f_total, n=n)
        c_act = actions.shape[0]
        actions_window = actions.index_select(1, history_indices).clone()

        local_data_timestep = history_len - 1
        images_mask = torch.ones_like(images_window, dtype=torch.bool)
        actions_mask = base_action_mask.index_select(1, history_indices).clone()
        actions_mask[:, local_data_timestep:] = False

        if states is not None:
            state_abs = states[:, data_timestep]
        else:
            # Fallback for legacy datasets without observation.state.
            state_abs = action_abs_flat[data_timestep * n]
        state_rel = get_relative_compact_action(state_abs[None], anchor_pose)
        state, state_mask = self._normalize_relative_compact_action(state_rel)
        state = state[0]
        state_mask = state_mask[0]
        # Current-frame VGA action branch uses the real robot state as token 0;
        # the remaining padded slots stay learned action-query tokens.
        actions_window[:, local_data_timestep, 0, 0] = state.to(actions_window.dtype)
        actions_mask[:, local_data_timestep, 0, 0] = state_mask.to(actions_mask.device)

        action_flat = rearrange(actions, 'c f n 1 -> c (f n)')
        action_mask_flat = rearrange(base_action_mask, 'c f n 1 -> c (f n)')
        chunk_start = data_timestep * n
        chunk_end = min(chunk_start + chunk_size, action_flat.shape[-1])
        future_action_chunk = action_flat[:, chunk_start:chunk_end]
        future_action_chunk_mask = action_mask_flat[:, chunk_start:chunk_end]
        if future_action_chunk.shape[-1] < chunk_size:
            chunk_pad = torch.zeros((c_act, chunk_size - future_action_chunk.shape[-1]), dtype=future_action_chunk.dtype)
            chunk_mask_pad = torch.zeros(
                (c_act, chunk_size - future_action_chunk_mask.shape[-1]),
                dtype=torch.bool,
            )
            future_action_chunk = torch.cat([future_action_chunk, chunk_pad], dim=1)
            future_action_chunk_mask = torch.cat([future_action_chunk_mask, chunk_mask_pad], dim=1)

        # RDT denoises [current_state, future_action_0, ..., future_action_{chunk_size-1}].
        action_chunk = torch.cat([state.unsqueeze(-1), future_action_chunk], dim=1)
        action_chunk_mask = torch.cat([state_mask.unsqueeze(-1), future_action_chunk_mask], dim=1)

        return {
            'images': images_window,
            'actions': actions_window,
            'images_mask': images_mask,
            'actions_mask': actions_mask,
            'action_chunk': action_chunk,
            'action_chunk_mask': action_chunk_mask,
            'state': state,
            'pred_frame_idx': torch.tensor(local_data_timestep, dtype=torch.long),
            'num_frames': torch.tensor(history_len, dtype=torch.long),
        }

    def __getitem__(self, idx) -> dict:
        """
        Arguments:
            idx: int, the index of the data item to retrieve. The dataset will be indexed
        Returns:            A dictionary containing the following keys:
                - 'images': Tensor of shape [C, F, n_view * H, W], the raw video frames.
                - 'text_emb': Tensor of shape [D], the text embedding associated with the video.
                - 'actions': Tensor of shape [C_act, F, N, 1], the processed action sequences aligned with the latent frames.
                - 'data_timestep': int, the timestep index used for training, indicating how many frames from the start are included in the input.
        """
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        # ori_latent_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)
        # cat_latent = self._cat_video_latents(ori_latent_data_dict)

        ori_data_dict = self._get_range_image_data(start_frame, end_frame, episode_index)

        image_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_images(ori_data_dict)
        out_dict['text'] = self._get_selected_text(cur_meta, ori_data_dict)
        if self.encode_text_in_dataloader:
            out_dict['text_emb'] = self._get_selected_text_embedding(cur_meta, ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(
            local_start_frame,
            local_end_frame,
            image_frame_ids,
            ori_data_dict['action'],
        )
        out_dict['state_seq'] = self._state_post_process(
            local_start_frame,
            local_end_frame,
            image_frame_ids,
            ori_data_dict.get('observation.state', None),
        )
        out_dict['actions'], out_dict['actions_mask'] = self._align_actions_with_multi_view_mode(
            out_dict['actions'],
            out_dict['actions_mask'],
        )
        out_dict['state_seq'] = self._align_state_with_multi_view_mode(out_dict.get('state_seq', None))
        out_dict['images'] = out_dict['images'].permute(1, 0, 2, 3) # [C, F, H, W]

        sampled_dict = self._sample_window_and_chunk(
            images=out_dict['images'],
            actions=out_dict['actions'],
            states=out_dict.get('state_seq', None),
        )
        sampled_dict['text'] = out_dict['text']
        if self.encode_text_in_dataloader:
            sampled_dict['text_emb'] = out_dict['text_emb']

        return sampled_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['robotwin_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
