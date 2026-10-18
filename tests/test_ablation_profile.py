import numpy as np

from evaluation.robotwin.eval_polict_client_openpi import _apply_ablation_profile


def test_masks_camera_and_waypoints():
    obs = {
        "head_camera": np.ones((4, 4, 3), dtype=np.uint8),
        "observation.images.cam_high": np.ones((4, 4, 3), dtype=np.uint8),
        "sparse_waypoints": {
            "left_pose_7d": np.ones((2, 7), dtype=np.float32),
            "right_pose_7d": np.ones((2, 7), dtype=np.float32),
            "left_valid": np.ones(2, dtype=bool),
            "right_valid": np.ones(2, dtype=bool),
            "block_valid": np.ones(2, dtype=bool),
        },
    }
    got = _apply_ablation_profile(obs, {"camera_missing": ["head"], "waypoint_missing_rate": 1.0})
    assert not got["head_camera"].any() and not got["observation.images.cam_high"].any()
    assert not got["sparse_waypoints"]["left_valid"].any()
    assert not got["sparse_waypoints"]["right_valid"].any()


def test_b0_profile_is_clean():
    import json
    from pathlib import Path

    profile = json.loads((Path(__file__).parents[1] / "configs/ablation_profiles/B0_baseline.json").read_text())
    assert profile["waypoint_noise"] == {
        "position_std": 0.0,
        "rotation_std_deg": 0.0,
        "gripper_std": 0.0,
        "duration_std": 0.0,
    }


def test_all_missing_keeps_shape_and_disables_guidance():
    obs = {
        "sparse_waypoints": {
            "left_pose_7d": np.ones((2, 7), dtype=np.float32),
            "right_pose_7d": np.ones((2, 7), dtype=np.float32),
            "left_valid": np.ones(2, dtype=bool),
            "right_valid": np.ones(2, dtype=bool),
            "block_valid": np.ones(2, dtype=bool),
        },
        "send_sparse_waypoints": True,
    }
    got = _apply_ablation_profile(obs, {"waypoint_missing_rate": 1.0})
    assert got["send_sparse_waypoints"] is False
    assert got["guidance_missing"] is True
    assert got["sparse_waypoints"]["left_valid"].shape == (2,)
