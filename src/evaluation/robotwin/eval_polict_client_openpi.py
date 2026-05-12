import sys
import os
import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from pathlib import Path

robowin_root = Path("/home/yds/code/RoboTwin")
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))


import os
os.chdir(robowin_root)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
from evaluation.robotwin.geometry import euler2quat
import numpy as np

from description.utils.generate_episode_instructions import *
import traceback

import imageio
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import transforms3d as t3d
import json
from pathlib import Path

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy


DATASET_ROOT = Path("/home/yds/code/StreamVGGT/dataset")


def _planner_result_payload(result):
    if result is None:
        return {
            "status": None,
            "success": False,
            "num_waypoints": 0,
        }
    status = result.get("status", None) if isinstance(result, dict) else getattr(result, "status", None)
    position = result.get("position", None) if isinstance(result, dict) else None
    num_waypoints = int(position.shape[0]) if hasattr(position, "shape") and len(position.shape) > 0 else 0
    return {
        "status": None if status is None else str(status),
        "success": bool(status == "Success"),
        "num_waypoints": num_waypoints,
    }


def take_ee_action_with_planner_feedback(task_env, ee_action):
    """Run an EE action and capture per-arm mplib planner/IK status."""
    feedback = {
        "plan_success_before": bool(getattr(task_env, "plan_success", True)),
        "left": None,
        "right": None,
    }

    robot = getattr(task_env, "robot", None)
    if robot is None:
        task_env.take_action(ee_action, action_type="ee")
        feedback["plan_success_after"] = bool(getattr(task_env, "plan_success", True))
        feedback["planner_success"] = feedback["plan_success_after"]
        return feedback

    original_left_plan_path = robot.left_plan_path
    original_right_plan_path = robot.right_plan_path

    def wrapped_left_plan_path(*args, **kwargs):
        result = original_left_plan_path(*args, **kwargs)
        feedback["left"] = _planner_result_payload(result)
        return result

    def wrapped_right_plan_path(*args, **kwargs):
        result = original_right_plan_path(*args, **kwargs)
        feedback["right"] = _planner_result_payload(result)
        return result

    robot.left_plan_path = wrapped_left_plan_path
    robot.right_plan_path = wrapped_right_plan_path
    try:
        task_env.take_action(ee_action, action_type="ee")
    finally:
        robot.left_plan_path = original_left_plan_path
        robot.right_plan_path = original_right_plan_path

    feedback["plan_success_after"] = bool(getattr(task_env, "plan_success", True))
    arm_success = [
        arm_feedback["success"]
        for arm_feedback in (feedback.get("left"), feedback.get("right"))
        if arm_feedback is not None
    ]
    feedback["planner_success"] = bool(arm_success and all(arm_success) and feedback["plan_success_after"])
    feedback["take_action_cnt"] = int(getattr(task_env, "take_action_cnt", -1))
    return feedback


def configure_headless_sapien_renderer() -> None:
    """Force a non-raytracing renderer path for headless stability.

    RoboTwin base tasks unconditionally enable raytracing + OIDN, which can
    crash on headless servers. Patch those calls to safe defaults before env
    construction.
    """
    try:
        import sapien.core as sapien
    except Exception:
        return

    try:
        original_set_shader_dir = sapien.render.set_camera_shader_dir

        def safe_set_camera_shader_dir(shader_dir):
            if shader_dir == "rt":
                return original_set_shader_dir("default")
            return original_set_shader_dir(shader_dir)

        sapien.render.set_camera_shader_dir = safe_set_camera_shader_dir
        sapien.render.set_ray_tracing_samples_per_pixel = lambda *args, **kwargs: None
        sapien.render.set_ray_tracing_path_depth = lambda *args, **kwargs: None
        sapien.render.set_ray_tracing_denoiser = lambda *args, **kwargs: None
    except Exception:
        # If patching fails, continue with original behavior.
        return

def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def add_title_bar(img, text, font_scale=0.8, thickness=2):
    """Add a black title bar with text above the image"""
    h, w, _ = img.shape
    bar_height = 40
    
    # Create black background bar
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    
    # Calculate text position to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2 - 5
    
    cv2.putText(title_bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return np.vstack([title_bar, img])

def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    quat: RoboTwin [w, x, y, z] format
    Return: [roll, pitch, yaw] (radians)
    """
    quat = safe_normalize_quat(quat)
    return np.asarray(t3d.euler.quat2euler(quat, axes="sxyz"), dtype=np.float64)

def visualize_action_step(action_history, step_idx, window=50):
    """
    Plot dual-arm action curves:
    Subplot 1: Left arm XYZ Position + Gripper
    Subplot 2: Left arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    Subplot 3: Right arm XYZ Position + Gripper
    Subplot 4: Right arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    
    Input data format: [left_x, left_y, left_z, left_rx, left_ry, left_rz, left_rw, left_gripper,
                   right_x, right_y, right_z, right_rx, right_ry, right_rz, right_rw, right_gripper]
    Total 16 dimensions
    """
    # Create four subplots, sharing the X-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=100, sharex=True)
    
    # 1. Determine slice range
    start = max(0, step_idx - window)
    end = step_idx + 1
    
    # 2. Get data subset
    history_subset = np.array(action_history)[start:end]
    
    # 3. Generate X-axis based on actual data length
    actual_len = len(history_subset)
    x_axis = range(start, start + actual_len)
    
    if actual_len > 0 and history_subset.shape[1] >= 16:
        # Convert quaternions to Euler angles
        left_euler = []
        right_euler = []
        
        for action in history_subset:
            # Left arm quaternion to Euler angles
            left_quat = action[3:7]  # [rx, ry, rz, rw]
            left_rpy = quaternion_to_euler(left_quat)
            left_euler.append(left_rpy)
            
            # Right arm quaternion to Euler angles
            right_quat = action[11:15]  # [rx, ry, rz, rw]
            right_rpy = quaternion_to_euler(right_quat)
            right_euler.append(right_rpy)
        
        left_euler = np.array(left_euler)
        right_euler = np.array(right_euler)
        
        # --- Left Arm ---
        # Subplot 1: Left Arm Translation (XYZ) + Gripper
        ax1.plot(x_axis, history_subset[:, 0], label='left_x', color='r', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 1], label='left_y', color='g', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 2], label='left_z', color='b', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 7], label='left_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Position (m)')
        ax1.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Step {step_idx}: Left Arm Position & Gripper")

        # Subplot 2: Left Arm Euler Angles (Roll, Pitch, Yaw)
        ax2.plot(x_axis, left_euler[:, 0], label='left_roll', color='c', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 1], label='left_pitch', color='m', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 2], label='left_yaw', color='y', linewidth=1.5)
        ax2.set_ylabel('Rotation (rad)')
        ax2.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Left Arm Rotation (RPY from Quaternion)")

        # --- Right Arm ---
        # Subplot 3: Right Arm Translation (XYZ) + Gripper
        ax3.plot(x_axis, history_subset[:, 8], label='right_x', color='r', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 9], label='right_y', color='g', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 10], label='right_z', color='b', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 15], label='right_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax3.set_ylabel('Position (m)')
        ax3.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Right Arm Position & Gripper")

        # Subplot 4: Right Arm Euler Angles (Roll, Pitch, Yaw)
        ax4.plot(x_axis, right_euler[:, 0], label='right_roll', color='c', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 1], label='right_pitch', color='m', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 2], label='right_yaw', color='y', linewidth=1.5, linestyle='--')
        ax4.set_ylabel('Rotation (rad)')
        ax4.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Right Arm Rotation (RPY from Quaternion)")

    # Set X-axis display range to maintain sliding window effect
    ax1.set_xlim(max(0, step_idx - window), max(window, step_idx))
    ax3.set_xlabel('Step')
    ax4.set_xlabel('Step')
    
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = img[:, :, :3]
    
    # Convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        
    plt.close(fig)
    return img


def save_comparison_video(real_obs_list, imagined_video, action_history, save_path, fps=15):
    if not real_obs_list:
        return

    n_real = len(real_obs_list)
    if imagined_video is not None:
        imagined_video = np.concatenate(imagined_video, 0)
        n_imagined = len(imagined_video) 
    else:
        n_imagined = 0
    n_frames = n_real # Based on real observation frames
    
    print(f"Saving video: Real {n_real} frames, Imagined {n_imagined} frames...")

    final_frames = []

    for i in range(n_frames):
        obs = real_obs_list[i]
        cam_high = obs["observation.images.cam_high"]
        cam_left = obs["observation.images.cam_left_wrist"]
        cam_right = obs["observation.images.cam_right_wrist"]

        base_h = cam_high.shape[0]
        
        def resize_h(img, h):
            if img.shape[0] != h:
                w = int(img.shape[1] * h / img.shape[0])
                return cv2.resize(img, (w, h))
            return img

        row_real = np.hstack([
            resize_h(cam_high, base_h), 
            resize_h(cam_left, base_h), 
            resize_h(cam_right, base_h)
        ])
        
        if row_real.dtype != np.uint8:
            row_real = (row_real * 255).astype(np.uint8)

        row_real = add_title_bar(row_real, "Real Observation (High / Left / Right)")

        target_width = row_real.shape[1]

        if imagined_video is not None and i < n_imagined:
            img_frame = imagined_video[i]
            if img_frame.dtype != np.uint8 and img_frame.max() <= 1.0001:
                img_frame = (img_frame * 255).astype(np.uint8)
            elif img_frame.dtype != np.uint8:
                img_frame = img_frame.astype(np.uint8)

            h = int(img_frame.shape[0] * target_width / img_frame.shape[1])
            row_imagined = cv2.resize(img_frame, (target_width, h))
        else:
            row_imagined = np.zeros((300, target_width, 3), dtype=np.uint8)
            cv2.putText(row_imagined, "Coming soon", (target_width//2 - 100, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        row_imagined = add_title_bar(row_imagined, "Imagined Video Stream")
        full_frame = np.vstack([row_real, row_imagined])
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Combined video saved to: {save_path}")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(robowin_root, "task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    video_guidance_scale = usr_args["video_guidance_scale"]
    action_guidance_scale = usr_args["action_guidance_scale"]
    instruction_type = 'seen'
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root
    args["max_episode_steps"] = usr_args.get("max_episode_steps", None)
    args["single_trajectory"] = bool(usr_args.get("single_trajectory", False))
    if usr_args.get("single_trajectory_episode_index", None) is not None:
        args["single_trajectory_episode_index"] = int(usr_args["single_trajectory_episode_index"])
    else:
        args["single_trajectory_episode_index"] = None
    args["single_trajectory_repo_id"] = usr_args.get("single_trajectory_repo_id", None)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")
    if args.get("single_trajectory", False):
        print(
            f"[single_trajectory][eval] enabled: repo_id={args.get('single_trajectory_repo_id', None)}, "
            f"episode_index={args.get('single_trajectory_episode_index', None)}, "
            f"test_num={usr_args.get('test_num', None)}"
        )
    else:
        print("[single_trajectory][eval] disabled")
    print(f"[eval] max_episode_steps={args.get('max_episode_steps', None)}")

    print(f"Connecting to policy server at ws://{usr_args.get('host', '127.0.0.1')}:{usr_args['port']} ...")
    model = WebsocketClientPolicy(host=usr_args.get('host', '127.0.0.1'), port=usr_args['port'])
    text_emb_lookup = DatasetTextEmbLookup(
        DATASET_ROOT,
        repo_id=args.get("single_trajectory_repo_id", None),
    )

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 10000 * (1 + seed)
    suc_nums = []
    test_num = usr_args["test_num"]

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   text_emb_lookup,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   save_visualization=True,
                                   video_guidance_scale=video_guidance_scale,
                                   action_guidance_scale=action_guidance_scale)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")

def format_eef_state(observation):
    endpose = observation["endpose"]
    return np.array(
        endpose["left_endpose"]
        + [endpose["left_gripper"]]
        + endpose["right_endpose"]
        + [endpose["right_gripper"]],
        dtype=np.float32,
    )


def format_obs(observation, prompt):
    return {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                "observation.state": format_eef_state(observation),
                "task": prompt,
            }


def safe_normalize_quat(quat, eps=1e-8):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < eps:
        # RobotWin/Sapien/transforms3d use [w, x, y, z].
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm

def add_eef_pose(new_pose, init_pose):
    new_quat = safe_normalize_quat(new_pose[3:7])
    init_quat = safe_normalize_quat(init_pose[3:7])
    out_rot = t3d.quaternions.qmult(init_quat, new_quat)
    out_rot = safe_normalize_quat(out_rot)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])

def add_init_pose(new_pose, init_pose):
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])


def sanitize_ee_action(ee_action):
    ee_action = np.asarray(ee_action, dtype=np.float64).reshape(-1)
    ee_action = np.nan_to_num(ee_action, nan=0.0, posinf=1e3, neginf=-1e3)
    if ee_action.shape[0] == 16:
        ee_action[3:7] = safe_normalize_quat(ee_action[3:7])
        ee_action[11:15] = safe_normalize_quat(ee_action[11:15])
    return ee_action


class DatasetTextEmbLookup:
    def __init__(self, dataset_root: Path, repo_id: str = None):
        self.dataset_root = Path(dataset_root)
        self.repo_id = None if repo_id is None else str(repo_id)
        self._cache = {}
        self._index = {}

    @staticmethod
    def _norm(text):
        if text is None:
            return None
        return str(text).strip()

    def _build_index(self, task_name: str):
        if task_name in self._index:
            return self._index[task_name]

        repo_name = Path(self.repo_id).name if self.repo_id else None
        if repo_name:
            search_patterns = [
                f"{repo_name}/latents/chunk-*/**/episode_*.pth",
                f"{repo_name}/latents/chunk-*/*/episode_*.pth",
            ]
        else:
            search_patterns = [
                f"{task_name}-*/latents/chunk-*/**/episode_*.pth",
                f"{task_name}-*/latents/chunk-*/*/episode_*.pth",
                f"**/latents/chunk-*/**/episode_*.pth",
                f"**/latents/chunk-*/*/episode_*.pth",
            ]

        files = []
        for pattern in search_patterns:
            files = sorted(self.dataset_root.glob(pattern))
            if len(files) > 0:
                break

        if len(files) == 0:
            self._index[task_name] = []
            return self._index[task_name]

        self._index[task_name] = files
        return files

    def get(self, task_name: str, prompt: str, episode_index: int = None):
        prompt_norm = self._norm(prompt)
        if not prompt_norm:
            return None

        cache_key = (task_name, prompt_norm, episode_index)
        if cache_key in self._cache:
            return self._cache[cache_key]

        files = self._build_index(task_name)
        if episode_index is not None:
            episode_tag = f"episode_{int(episode_index):06d}_"
            files = [p for p in files if episode_tag in p.name]

        for latent_file in files:
            try:
                payload = torch.load(latent_file, map_location="cpu", weights_only=False)
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            if self._norm(payload.get("text", None)) != prompt_norm:
                continue

            text_emb = payload.get("text_emb", None)
            if text_emb is None:
                continue
            if not torch.is_tensor(text_emb):
                text_emb = torch.as_tensor(text_emb)
            if text_emb.ndim == 2:
                text_emb = text_emb.unsqueeze(0)
            elif text_emb.ndim != 3:
                continue

            self._cache[cache_key] = text_emb
            return text_emb

        self._cache[cache_key] = None
        return None

    def get_episode_entry(self, task_name: str, episode_index: int):
        """Return (text, text_emb) from the exact trajectory episode if available."""
        files = self._build_index(task_name)
        episode_tag = f"episode_{int(episode_index):06d}_"
        for latent_file in files:
            if episode_tag not in latent_file.name:
                continue
            try:
                payload = torch.load(latent_file, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            text = self._norm(payload.get("text", None))
            text_emb = payload.get("text_emb", None)
            if text_emb is None:
                continue
            if not torch.is_tensor(text_emb):
                text_emb = torch.as_tensor(text_emb)
            if text_emb.ndim == 2:
                text_emb = text_emb.unsqueeze(0)
            elif text_emb.ndim != 3:
                continue
            return text, text_emb
        return None, None

    def get_episode_debug_entry(self, task_name: str, episode_index: int):
        """Return text, text_emb, and latent file metadata for one episode if available."""
        files = self._build_index(task_name)
        episode_tag = f"episode_{int(episode_index):06d}_"
        for latent_file in files:
            if episode_tag not in latent_file.name:
                continue
            try:
                payload = torch.load(latent_file, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            text = self._norm(payload.get("text", None))
            text_emb = payload.get("text_emb", None)
            if text_emb is not None and not torch.is_tensor(text_emb):
                text_emb = torch.as_tensor(text_emb)
            if text_emb is not None:
                if text_emb.ndim == 2:
                    text_emb = text_emb.unsqueeze(0)
                elif text_emb.ndim != 3:
                    text_emb = None

            stem_match = re.search(r"episode_(\d{6})_(\d+)_(\d+)\.pth$", latent_file.name)
            start_frame = int(stem_match.group(2)) if stem_match else None
            end_frame = int(stem_match.group(3)) if stem_match else None
            repo_name = latent_file.parents[3].name if len(latent_file.parents) > 3 else None
            return {
                "text": text,
                "text_emb": text_emb,
                "latent_file": latent_file,
                "repo_name": repo_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "segment_length": None if start_frame is None or end_frame is None else end_frame - start_frame,
            }
        return None

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                text_emb_lookup,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                save_visualization=False,
                video_guidance_scale=5.0,
                action_guidance_scale=5.0):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []


    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    single_trajectory = bool(args.get("single_trajectory", False))
    single_trajectory_episode_index = args.get("single_trajectory_episode_index", None)
    if single_trajectory_episode_index is not None:
        single_trajectory_episode_index = int(single_trajectory_episode_index)
        single_trajectory = True

    if single_trajectory and single_trajectory_episode_index is None:
        single_trajectory_episode_index = 0
        print("single_trajectory=True and no episode index provided; defaulting to episode index 0.")

    if single_trajectory and test_num != 1:
        print(f"single_trajectory=True: forcing test_num=1 (was {test_num}).")
        test_num = 1

    args["eval_mode"] = True
    max_episode_steps = args.get("max_episode_steps", None)
    if max_episode_steps is not None:
        max_episode_steps = int(max_episode_steps)

    def apply_step_limit_override():
        if max_episode_steps is not None and max_episode_steps > 0:
            old_step_lim = getattr(TASK_ENV, "step_lim", None)
            TASK_ENV.step_lim = max_episode_steps
            if old_step_lim != max_episode_steps:
                print(f"[eval] override step_lim: {old_step_lim} -> {TASK_ENV.step_lim}")

    while succ_seed < test_num:
        current_episode_index = single_trajectory_episode_index if single_trajectory else now_id
        current_seed = st_seed if single_trajectory else now_seed
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=current_episode_index, seed=current_seed, is_test=True, **args)
                apply_step_limit_override()
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                if single_trajectory:
                    print(f"Single-trajectory expert check failed with UnStableError: {e}")
                    break
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                if single_trajectory:
                    print(f"Single-trajectory expert check failed with exception: {e}")
                    traceback.print_exc()
                    break
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs ! {e}")
                traceback.print_exc()
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            if single_trajectory:
                print("Single-trajectory expert check failed; aborting this run.")
                break
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=current_episode_index, seed=current_seed, is_test=True, **args)
        apply_step_limit_override()
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        if single_trajectory:
            episode_debug = text_emb_lookup.get_episode_debug_entry(
                task_name, current_episode_index
            )
            if episode_debug is not None:
                debug_file = episode_debug.get("latent_file", None)
                debug_repo = episode_debug.get("repo_name", None)
                debug_len = episode_debug.get("segment_length", None)
                dataset_instruction = episode_debug.get("text", None)
                print(
                    f"[single_trajectory][eval] task={task_name}, episode_index={current_episode_index}, "
                    f"seed={current_seed}, step_lim={getattr(TASK_ENV, 'step_lim', None)}, "
                    f"repo={debug_repo}, latent_file={str(debug_file) if debug_file is not None else None}, "
                    f"segment_len={debug_len}, env_instruction={instruction}, dataset_instruction={dataset_instruction}"
                )
                if dataset_instruction is not None and str(dataset_instruction).strip() != str(instruction).strip():
                    print(
                        "[single_trajectory][eval] warning: dataset episode text and env-generated instruction differ; "
                        "using env-generated instruction for evaluation."
                    )
            else:
                print(
                    f"[single_trajectory][eval] task={task_name}, episode_index={current_episode_index}, "
                    f"seed={current_seed}, step_lim={getattr(TASK_ENV, 'step_lim', None)}, "
                    f"env_instruction={instruction}, dataset_instruction=None"
                )
        if single_trajectory:
            print(f"[single_trajectory][eval] env_info={episode_info['info']}")
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        prompt = TASK_ENV.get_instruction()
        episode_index = int(current_episode_index)
        # StreamVGGT now follows p2p and encodes the prompt on the policy server with
        # EmbeddingGemma. Do not send legacy dataset text_emb tensors from the client.
        prompt_text_emb = None
        ret = model.infer(dict(
            reset=True,
            prompt=prompt,
            text_emb=prompt_text_emb,
            save_visualization=save_visualization,
        ))
        
        full_obs_list = []
        gen_video_list = []
        full_action_history = []

        initial_obs = TASK_ENV.get_obs() 
        inint_eef_pose = initial_obs['endpose']['left_endpose'] + \
        [initial_obs['endpose']['left_gripper']] + \
        initial_obs['endpose']['right_endpose'] + \
        [initial_obs['endpose']['right_gripper']]
        inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
        initial_formatted_obs = format_obs(initial_obs, prompt)
        full_obs_list.append(initial_formatted_obs)
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            current_obs = format_obs(observation, prompt)
            current_obs["episode_index"] = episode_index

            ret = model.infer(dict(
                obs=current_obs,
                prompt=prompt,
                text_emb=prompt_text_emb,
                save_visualization=save_visualization,
                video_guidance_scale=video_guidance_scale,
                action_guidance_scale=action_guidance_scale,
            )) #(TASK_ENV, model, observation)
            has_absolute_action = 'action_absolute' in ret
            action = ret['action_absolute'] if has_absolute_action else ret['action']
            if 'video' in ret:
                imagined_video = ret['video']
                gen_video_list.append(imagined_video)
            raw_action_step = action[:, 0, 0].flatten()
            full_action_history.append(raw_action_step)

            ee_action = action[:, 0, 0]
            if action.shape[0] == 14:
                ee_action = np.concatenate([
                    ee_action[:3],
                    euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                    ee_action[6:10],
                    euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                    ee_action[13:14]
                ])
            elif action.shape[0] == 16:
                if not has_absolute_action:
                    action_reference = ret.get('action_reference', inint_eef_pose)
                    ee_action = add_init_pose(ee_action, action_reference)
                ee_action = np.concatenate([
                    ee_action[:3],
                    safe_normalize_quat(ee_action[3:7]),
                    ee_action[7:11],
                    safe_normalize_quat(ee_action[11:15]),
                    ee_action[15:16]
                ])
            else:
                raise NotImplementedError
            ee_action = sanitize_ee_action(ee_action)
            planner_feedback = take_ee_action_with_planner_feedback(TASK_ENV, ee_action)
            try:
                model.infer(dict(
                    planner_feedback=planner_feedback,
                ))
            except Exception as exc:
                print(f"[eval] warning: failed to send planner feedback to policy server: {exc}")

            obs = format_obs(TASK_ENV.get_obs(), prompt)
            full_obs_list.append(obs)
  
            if TASK_ENV.eval_success:
                succ = True
                break
      

        vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"{TASK_ENV.test_num}_{prompt.replace(' ', '_')}_{succ}.mp4"
        out_img_file = vis_dir / video_name
        save_comparison_video(
            real_obs_list=full_obs_list,
            imagined_video=None, #gen_video_list,
            action_history=full_action_history,
            save_path=str(out_img_file),
            fps=15 # Suggest adjusting fps based on simulation step
        )
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        save_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'metrics' / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json_file = save_dir / 'res.json'
        write_json({
          "succ_num": float(TASK_ENV.suc),
          "total_num": float(TASK_ENV.test_num),
          "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
        }, out_json_file)
        
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        if not single_trajectory:
            now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--headless", action="store_true", help="run without render self-test on headless servers")
    parser.add_argument("--host", type=str, default="127.0.0.1", help='remote policy server host.')
    parser.add_argument("--port", type=int, default=8000, help='remote policy socket port.')
    parser.add_argument("--save_root", type=str, default="results/default_vis_path")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=5.0)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--single_trajectory", action="store_true")
    parser.add_argument("--single_trajectory_episode_index", type=int, default=None)
    parser.add_argument("--single_trajectory_repo_id", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    # CLI args should always take precedence over yaml defaults.
    config["headless"] = bool(args.headless)
    config["host"] = args.host
    config["port"] = args.port
    config["save_root"] = args.save_root
    config["video_guidance_scale"] = args.video_guidance_scale
    config["action_guidance_scale"] = args.action_guidance_scale
    config["test_num"] = args.test_num
    if args.max_episode_steps is not None:
        config["max_episode_steps"] = int(args.max_episode_steps)
    if args.single_trajectory:
        config["single_trajectory"] = True
    if args.single_trajectory_episode_index is not None:
        config["single_trajectory"] = True
        config["single_trajectory_episode_index"] = int(args.single_trajectory_episode_index)
    if args.single_trajectory_repo_id is not None:
        config["single_trajectory"] = True
        config["single_trajectory_repo_id"] = args.single_trajectory_repo_id

    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    if not usr_args.get("headless", False):
        from evaluation.robotwin.test_render import Sapien_TEST
        Sapien_TEST()
    else:
        print("Headless mode enabled: skipping render self-test")
        configure_headless_sapien_renderer()
    main(usr_args)
