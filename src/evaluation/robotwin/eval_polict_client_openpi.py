import sys
import os
import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from pathlib import Path

robowin_root = Path("/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled")
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))


import os
os.chdir(robowin_root)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
from copy import deepcopy
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
from envs.utils.expert_marker import draw_expert_target_sequence_on_rgb
from evaluation.ee_target_waypoints import build_linear_ee_target_transitions


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
    print("[eval_client] StreamVGGT joint-state patch: live_robot_qpos_v2")
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
    args["use_expert_marked_rgb"] = bool(usr_args.get("use_expert_marked_rgb", True))
    args["live_waypoint_labels"] = bool(usr_args.get("live_waypoint_labels", True))
    args["ee_target_sequence_len"] = int(usr_args.get("ee_target_sequence_len", 6))
    checkpoint_dir = usr_args.get("checkpoint_dir", None)
    if checkpoint_dir is not None:
        training_config_path = Path(checkpoint_dir) / "training_config.json"
        if training_config_path.is_file():
            try:
                training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
                if "norm_stats_by_action_mode" in training_config:
                    args["norm_stats_by_action_mode"] = training_config["norm_stats_by_action_mode"]
                for key in (
                    "robotwin_action_space",
                    "action_representation",
                    "joint_action_representation",
                    "use_expert_marked_rgb",
                    "ee_target_sequence_len",
                ):
                    if key in training_config and key not in args:
                        args[key] = training_config[key]
            except Exception as exc:
                print(f"[eval] warning: failed to load checkpoint training_config.json: {exc}")

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


def format_joint_state(observation):
    joint_action = observation["joint_action"]
    if "vector" in joint_action:
        joint_state = np.asarray(joint_action["vector"], dtype=np.float32).reshape(-1)
        if joint_state.size == 16:
            return np.concatenate(
                [
                    joint_state[:6],
                    joint_state[7:8],
                    joint_state[8:14],
                    joint_state[15:16],
                ]
            ).astype(np.float32)
        return joint_state
    left_arm = np.asarray(joint_action["left_arm"], dtype=np.float32).reshape(-1)
    right_arm = np.asarray(joint_action["right_arm"], dtype=np.float32).reshape(-1)
    if left_arm.size == 7 and right_arm.size == 7:
        left_arm = left_arm[:6]
        right_arm = right_arm[:6]
    return np.concatenate(
        [
            left_arm,
            np.asarray([joint_action["left_gripper"]], dtype=np.float32),
            right_arm,
            np.asarray([joint_action["right_gripper"]], dtype=np.float32),
        ]
    ).astype(np.float32)


def format_robot_joint_state(task_env):
    robot = getattr(task_env, "robot", None)
    if robot is None:
        raise RuntimeError("TASK_ENV has no robot; cannot read live joint state.")
    joint_state = np.asarray(
        robot.get_left_arm_jointState() + robot.get_right_arm_jointState(),
        dtype=np.float32,
    ).reshape(-1)
    if joint_state.size == 16:
        return np.concatenate(
            [
                joint_state[:6],
                joint_state[7:8],
                joint_state[8:14],
                joint_state[15:16],
            ]
        ).astype(np.float32)
    return joint_state


def _compact_ee_state_14(observation):
    try:
        eef_state = format_eef_state(observation).reshape(-1)
    except Exception:
        return None
    if eef_state.size == 16:
        return np.concatenate(
            [
                eef_state[:6],
                eef_state[7:8],
                eef_state[8:14],
                eef_state[15:16],
            ]
        ).astype(np.float32)
    return None


def expand_compact_joint_action_for_robotwin(action, observation):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    joint_action = observation.get("joint_action", {})
    if "vector" in joint_action:
        live_state = np.asarray(joint_action["vector"], dtype=np.float32).reshape(-1)
        if action.size == live_state.size:
            return action
        if action.size == 14 and live_state.size == 16:
            return np.concatenate(
                [
                    action[:6],
                    live_state[6:7],
                    action[6:7],
                    action[7:13],
                    live_state[14:15],
                    action[13:14],
                ]
            ).astype(np.float32)

    left_arm = np.asarray(joint_action.get("left_arm", []), dtype=np.float32).reshape(-1)
    right_arm = np.asarray(joint_action.get("right_arm", []), dtype=np.float32).reshape(-1)
    if action.size == left_arm.size + 1 + right_arm.size + 1:
        return action
    if action.size == 14 and left_arm.size == 7 and right_arm.size == 7:
        return np.concatenate(
            [
                action[:6],
                left_arm[6:7],
                action[6:7],
                action[7:13],
                right_arm[6:7],
                action[13:14],
            ]
        ).astype(np.float32)
    return action


def execute_dense_qpos_action(task_env, qpos_action, num_steps=15):
    """Execute a qpos target as dense drive targets instead of TOPP planning.

    RobotWin's take_action(qpos) treats qpos as a high-level endpoint and sends it
    through TOPP. The learned joint policy predicts dense saved-frame drive
    targets, so apply them directly for a few simulator steps.
    """
    if task_env.take_action_cnt == task_env.step_lim or task_env.eval_success:
        return {
            "planner_success": False,
            "action_type": "qpos_direct",
            "skipped": True,
            "reason": "episode_done",
            "take_action_cnt": int(getattr(task_env, "take_action_cnt", -1)),
        }

    qpos_action = np.asarray(qpos_action, dtype=np.float32).reshape(-1)
    robot = getattr(task_env, "robot", None)
    if robot is None:
        raise RuntimeError("TASK_ENV has no robot; cannot execute dense qpos action.")

    left_jointstate = robot.get_left_arm_jointState()
    right_jointstate = robot.get_right_arm_jointState()
    left_arm_dim = len(left_jointstate) - 1
    right_arm_dim = len(right_jointstate) - 1
    expected_dim = left_arm_dim + 1 + right_arm_dim + 1
    if qpos_action.size != expected_dim:
        raise ValueError(
            f"Dense qpos action dim mismatch: expected {expected_dim} "
            f"({left_arm_dim}+gripper+{right_arm_dim}+gripper), got {qpos_action.size}"
        )

    left_arm_target = qpos_action[:left_arm_dim]
    left_gripper_target = float(qpos_action[left_arm_dim])
    right_start = left_arm_dim + 1
    right_arm_target = qpos_action[right_start : right_start + right_arm_dim]
    right_gripper_target = float(qpos_action[right_start + right_arm_dim])

    eval_video_freq = 1
    if task_env.eval_video_path is not None and task_env.take_action_cnt % eval_video_freq == 0:
        task_env.eval_video_ffmpeg.stdin.write(
            task_env.now_obs["observation"]["head_camera"]["rgb"].tobytes()
        )

    task_env.take_action_cnt += 1
    print(f"step: \033[92m{task_env.take_action_cnt} / {task_env.step_lim}\033[0m", end="\r")

    zero_left_vel = np.zeros(left_arm_dim, dtype=np.float32)
    zero_right_vel = np.zeros(right_arm_dim, dtype=np.float32)
    num_steps = max(1, int(num_steps))
    for _ in range(num_steps):
        robot.set_arm_joints(left_arm_target, zero_left_vel, "left")
        robot.set_arm_joints(right_arm_target, zero_right_vel, "right")
        robot.set_gripper(left_gripper_target, "left")
        robot.set_gripper(right_gripper_target, "right")
        task_env.scene.step()
        task_env._update_render()
        if task_env.render_freq:
            task_env.viewer.render()
        if task_env.check_success():
            task_env.eval_success = True
            task_env.get_obs()
            if task_env.eval_video_path is not None:
                task_env.eval_video_ffmpeg.stdin.write(
                    task_env.now_obs["observation"]["head_camera"]["rgb"].tobytes()
                )
            break

    return {
        "planner_success": True,
        "action_type": "qpos_direct",
        "direct_control_steps": int(num_steps),
        "raw_action_dim": int(qpos_action.size),
        "executed_action_dim": int(qpos_action.size),
        "left_arm_dim": int(left_arm_dim),
        "right_arm_dim": int(right_arm_dim),
        "take_action_cnt": int(getattr(task_env, "take_action_cnt", -1)),
    }


def clip_joint_action_to_dataset_bounds(qpos_action, args):
    if not bool(args.get("clip_joint_action_to_dataset_bounds", True)):
        return np.asarray(qpos_action, dtype=np.float32).reshape(-1), False
    stats_by_mode = args.get("norm_stats_by_action_mode", None)
    if not isinstance(stats_by_mode, dict) or "joint_absolute" not in stats_by_mode:
        return np.asarray(qpos_action, dtype=np.float32).reshape(-1), False
    stats = stats_by_mode["joint_absolute"]
    q01 = np.asarray(stats.get("q01", []), dtype=np.float32).reshape(-1)
    q99 = np.asarray(stats.get("q99", []), dtype=np.float32).reshape(-1)
    action = np.asarray(qpos_action, dtype=np.float32).reshape(-1)
    if q01.size != action.size or q99.size != action.size:
        return action, False
    clipped = np.clip(action, q01, q99)
    return clipped.astype(np.float32), bool(np.any(np.abs(clipped - action) > 1e-6))


def _get_camera_rgb(observation, camera_name, use_expert_marked_rgb=True, required=True):
    cameras = observation.get("observation", {})
    if camera_name not in cameras:
        if required:
            raise KeyError(
                f"Observation is missing camera `{camera_name}`. "
                f"Available cameras={sorted(cameras.keys())}"
            )
        return None
    camera_obs = cameras[camera_name]
    if not use_expert_marked_rgb:
        return camera_obs["rgb"]
    return camera_obs.get("rgb_expert_marked", camera_obs["rgb"])


def _compact_expert_target(observation):
    expert_target = observation.get("expert_target", None)
    if not isinstance(expert_target, dict):
        return None

    compact = []
    valid = []
    command_id = []
    for arm in ("left", "right"):
        arm_target = expert_target.get(arm, {})
        try:
            pose = np.asarray(arm_target["pose_7d"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(arm_target["gripper"], dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if pose.shape[0] != 7 or gripper.shape[0] < 1:
            return None
        compact.append(np.concatenate([pose, gripper[:1]], axis=0).astype(np.float32))
        valid.append(bool(np.asarray(arm_target.get("valid", [True])).reshape(-1)[0]))
        command_id.append(int(np.asarray(arm_target.get("command_id", [-1])).reshape(-1)[0]))

    return {
        "value": np.stack(compact, axis=0).tolist(),
        "valid": valid,
        "command_id": command_id,
    }


def _live_waypoint_sequence(observation, sequence_len):
    """Build projected-pose labels from live EE state to current expert targets."""
    target = _compact_expert_target(observation)
    if target is None:
        return None
    try:
        current = format_eef_state(observation).reshape(2, 8).astype(np.float32)
        target_value = np.asarray(target["value"], dtype=np.float32).reshape(2, 8)
        transitions = build_linear_ee_target_transitions(
            current,
            target_value,
            sequence_len,
            valid=target["valid"],
        )
    except (KeyError, TypeError, ValueError):
        return None

    horizon = transitions.shape[1]
    alpha = np.arange(1, horizon + 1, dtype=np.float32) / float(horizon)
    sequence = {}
    for arm_idx, arm in enumerate(("left", "right")):
        pose = np.zeros((horizon + 1, 7), dtype=np.float64)
        gripper = np.ones((horizon + 1,), dtype=np.float64)
        pose[0] = current[arm_idx, :7]
        gripper[0] = current[arm_idx, 7]
        pose[1:, :3] = current[arm_idx, None, :3] + alpha[:, None] * (
            target_value[arm_idx, None, :3] - current[arm_idx, None, :3]
        )
        # Orientation is not predicted by the EE-target head; retain the live
        # orientation only so the projected point label has a valid pose.
        pose[1:, 3:7] = current[arm_idx, None, 3:7]
        gripper[1:] = current[arm_idx, 7] + transitions[arm_idx, :, 3]
        valid = np.concatenate([[True], np.full((horizon,), bool(target["valid"][arm_idx]))])
        sequence[arm] = {"pose_7d": pose, "gripper": gripper, "valid": valid}
    return sequence


def _get_live_marked_camera_rgb(observation, camera_name, sequence_len):
    cameras = observation.get("observation", {})
    camera_obs = cameras.get(camera_name)
    if camera_obs is None:
        raise KeyError(
            f"Observation is missing camera `{camera_name}`. "
            f"Available cameras={sorted(cameras.keys())}"
        )
    sequence = _live_waypoint_sequence(observation, sequence_len)
    if sequence is None:
        return camera_obs["rgb"]
    marked, _ = draw_expert_target_sequence_on_rgb(camera_obs["rgb"], camera_obs, sequence)
    return marked


def format_obs(
    observation,
    prompt,
    use_expert_marked_rgb=True,
    robotwin_action_space="ee",
    live_waypoint_labels=False,
    ee_target_sequence_len=6,
):
    robotwin_action_space = str(robotwin_action_space or "ee").lower()
    state = format_joint_state(observation) if robotwin_action_space == "joint" else format_eef_state(observation)
    camera_rgb = lambda name: (
        _get_live_marked_camera_rgb(observation, name, ee_target_sequence_len)
        if live_waypoint_labels
        else _get_camera_rgb(observation, name, use_expert_marked_rgb)
    )
    obs = {
        "head_camera": camera_rgb("head_camera"),
        "left_camera": camera_rgb("left_camera"),
        "right_camera": camera_rgb("right_camera"),
        "side_camera": camera_rgb("side_camera"),
        "observation.images.cam_high": camera_rgb("head_camera"),
        "observation.images.cam_left_wrist": camera_rgb("left_camera"),
        "observation.images.cam_right_wrist": camera_rgb("right_camera"),
        "observation.state": state,
        "task": prompt,
    }
    front_camera = _get_camera_rgb(
        observation,
        "front_camera",
        use_expert_marked_rgb,
        required=False,
    )
    if front_camera is not None:
        obs["front_camera"] = (
            _get_live_marked_camera_rgb(observation, "front_camera", ee_target_sequence_len)
            if live_waypoint_labels
            else front_camera
        )
    expert_target = _compact_expert_target(observation)
    if expert_target is not None:
        obs["expert_target"] = expert_target
    try:
        obs["ee_state"] = format_eef_state(observation).reshape(2, 8).tolist()
    except Exception:
        pass
    return obs


def format_policy_obs(
    task_env,
    observation,
    prompt,
    use_expert_marked_rgb=True,
    robotwin_action_space="ee",
    live_waypoint_labels=True,
    ee_target_sequence_len=6,
):
    obs = format_obs(
        observation,
        prompt,
        use_expert_marked_rgb=use_expert_marked_rgb,
        robotwin_action_space=robotwin_action_space,
        live_waypoint_labels=live_waypoint_labels,
        ee_target_sequence_len=ee_target_sequence_len,
    )
    if str(robotwin_action_space or "ee").lower() == "joint":
        obs["observation.state"] = format_robot_joint_state(task_env)
        compact_ee = _compact_ee_state_14(observation)
        if compact_ee is not None and np.allclose(obs["observation.state"], compact_ee, atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                "Joint-mode policy observation is using end-effector pose as joint state. "
                "This would create invalid joint targets. Check that the updated eval client is running "
                "and that robot.get_*_arm_jointState() returns qpos drive targets."
            )
    return obs


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
    expert_timeline = deepcopy(args.get("expert_timeline", []))
    left_joint_path = deepcopy(args.get("left_joint_path", []))
    right_joint_path = deepcopy(args.get("right_joint_path", []))
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
                expert_timeline = deepcopy(getattr(TASK_ENV, "expert_timeline", []))
                left_joint_path = deepcopy(getattr(TASK_ENV, "left_joint_path", []))
                right_joint_path = deepcopy(getattr(TASK_ENV, "right_joint_path", []))
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

        rollout_args = deepcopy(args)
        rollout_args["need_plan"] = False
        rollout_args["left_joint_path"] = left_joint_path
        rollout_args["right_joint_path"] = right_joint_path
        rollout_args["expert_timeline"] = expert_timeline
        rollout_args["expert_target_progress_mode"] = "state"

        TASK_ENV.setup_demo(now_ep_num=current_episode_index, seed=current_seed, is_test=True, **rollout_args)
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

        initial_obs = TASK_ENV.get_obs() 
        inint_eef_pose = initial_obs['endpose']['left_endpose'] + \
        [initial_obs['endpose']['left_gripper']] + \
        initial_obs['endpose']['right_endpose'] + \
        [initial_obs['endpose']['right_gripper']]
        inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
        prompt = TASK_ENV.get_instruction()
        episode_index = int(current_episode_index)
        use_expert_marked_rgb = bool(args.get("use_expert_marked_rgb", True))
        robotwin_action_space = str(args.get("robotwin_action_space", "ee")).lower()
        initial_formatted_obs = format_policy_obs(
            TASK_ENV,
        initial_obs,
        prompt,
        use_expert_marked_rgb=use_expert_marked_rgb,
        robotwin_action_space=robotwin_action_space,
        live_waypoint_labels=bool(args.get("live_waypoint_labels", True)),
        ee_target_sequence_len=int(args.get("ee_target_sequence_len", 6)),
        )

        # StreamVGGT now follows p2p and encodes the prompt on the policy server with
        # EmbeddingGemma. Do not send legacy dataset text_emb tensors from the client.
        prompt_text_emb = None
        ret = model.infer(dict(
            reset=True,
            obs=initial_formatted_obs,
            prompt=prompt,
            text_emb=prompt_text_emb,
            save_visualization=save_visualization,
        ))
        
        full_obs_list = []
        gen_video_list = []
        full_action_history = []

        full_obs_list.append(initial_formatted_obs)
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            current_obs = format_policy_obs(
                TASK_ENV,
                observation,
                prompt,
                use_expert_marked_rgb=use_expert_marked_rgb,
                robotwin_action_space=robotwin_action_space,
                live_waypoint_labels=bool(args.get("live_waypoint_labels", True)),
                ee_target_sequence_len=int(args.get("ee_target_sequence_len", 6)),
            )
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

            action_type = ret.get(
                "action_type",
                "qpos" if ret.get("robotwin_action_space", robotwin_action_space) == "joint" else "ee",
            )
            if action_type == "qpos":
                qpos_action = expand_compact_joint_action_for_robotwin(raw_action_step, observation)
                qpos_action, clipped_to_bounds = clip_joint_action_to_dataset_bounds(qpos_action, args)
                if bool(args.get("joint_use_direct_control", True)):
                    planner_feedback = execute_dense_qpos_action(
                        TASK_ENV,
                        qpos_action,
                        num_steps=int(args.get("joint_direct_control_steps", 15)),
                    )
                    planner_feedback["raw_action_dim"] = int(raw_action_step.size)
                    planner_feedback["clipped_to_dataset_bounds"] = bool(clipped_to_bounds)
                else:
                    TASK_ENV.take_action(qpos_action, action_type="qpos")
                    planner_feedback = {
                        "planner_success": True,
                        "action_type": "qpos_topp",
                        "raw_action_dim": int(raw_action_step.size),
                        "executed_action_dim": int(qpos_action.size),
                        "clipped_to_dataset_bounds": bool(clipped_to_bounds),
                        "take_action_cnt": int(getattr(TASK_ENV, "take_action_cnt", -1)),
                    }
            elif action_type == "ee":
                ee_action = action[:, 0, 0]
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
                ee_action = sanitize_ee_action(ee_action)
                planner_feedback = take_ee_action_with_planner_feedback(TASK_ENV, ee_action)
            else:
                raise NotImplementedError(f"Unsupported action_type `{action_type}`")
            try:
                model.infer(dict(
                    planner_feedback=planner_feedback,
                ))
            except Exception as exc:
                print(f"[eval] warning: failed to send planner feedback to policy server: {exc}")

            obs = format_policy_obs(
                TASK_ENV,
                TASK_ENV.get_obs(),
                prompt,
                use_expert_marked_rgb=use_expert_marked_rgb,
                robotwin_action_space=robotwin_action_space,
                live_waypoint_labels=bool(args.get("live_waypoint_labels", True)),
                ee_target_sequence_len=int(args.get("ee_target_sequence_len", 6)),
            )
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
    parser.add_argument("--use_expert_marked_rgb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--joint_use_direct_control", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--joint_direct_control_steps", type=int, default=None)
    parser.add_argument("--clip_joint_action_to_dataset_bounds", action=argparse.BooleanOptionalAction, default=None)
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
    if args.use_expert_marked_rgb is not None:
        config["use_expert_marked_rgb"] = bool(args.use_expert_marked_rgb)
    if args.joint_use_direct_control is not None:
        config["joint_use_direct_control"] = bool(args.joint_use_direct_control)
    else:
        config.setdefault("joint_use_direct_control", True)
    if args.joint_direct_control_steps is not None:
        config["joint_direct_control_steps"] = int(args.joint_direct_control_steps)
    else:
        config.setdefault("joint_direct_control_steps", 15)
    if args.clip_joint_action_to_dataset_bounds is not None:
        config["clip_joint_action_to_dataset_bounds"] = bool(args.clip_joint_action_to_dataset_bounds)
    else:
        config.setdefault("clip_joint_action_to_dataset_bounds", True)
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
