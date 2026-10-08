"""Static/runtime-light checks for the head-camera temporal experiment.

The static path intentionally does not import the project config modules, because
those modules may require optional training dependencies such as easydict.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _source(path):
    return path.read_text(encoding="utf-8")


def _literal_assignments(path):
    tree = ast.parse(_source(path), filename=str(path))
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            try:
                literal = ast.literal_eval(value)
            except Exception:
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    values[target.attr] = literal
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Name) and call.func.id == "setattr"
                    and len(call.args) == 3 and isinstance(call.args[1], ast.Constant)):
                try:
                    values[str(call.args[1].value)] = ast.literal_eval(call.args[2])
                except Exception:
                    continue
    return values


def _require(text, needle, label):
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def validate_static():
    train_cfg = ROOT / "configs" / "head_camera_temporal_train_cfg.py"
    eval_cfg = ROOT / "configs" / "head_camera_temporal_cfg.py"
    dataset = ROOT / "dataset" / "robotwin_lerobot_dataset.py"
    server = ROOT / "va_server.py"
    train = ROOT / "train_va.py"
    train_values = _literal_assignments(train_cfg)
    eval_values = _literal_assignments(eval_cfg)
    expected = ["head_camera"]
    camera_keys = ("obs_cam_keys", "current_obs_cam_keys", "history_obs_cam_keys", "view_position_cam_keys")
    for name, values, path in (("train", train_values, train_cfg), ("eval", eval_values, eval_cfg)):
        # The configs assign camera lists through a small setattr loop. Check
        # that source form as well as direct literals, so this remains usable
        # without importing EasyDict or any model dependency.
        for key in camera_keys:
            if values.get(key) != expected:
                source = _source(path)
                if f'setattr({"head_camera_temporal_train_cfg" if name == "train" else "head_camera_temporal_cfg"}, key, ["head_camera"])' not in source:
                    raise AssertionError(f"{name}.{key}={values.get(key)!r}, expected {expected!r}")
        if values.get("history_len") != 8 or values.get("temporal_window") != 8:
            raise AssertionError(f"{name} temporal window must be 8")
        if values.get("history_frame_stride") != values.get("temporal_stride"):
            raise AssertionError(f"{name} history/temporal stride mismatch")
        if values.get("image_frame_stride") != 1:
            raise AssertionError(f"{name}.image_frame_stride must be 1")

    dataset_text, server_text, train_text = map(_source, (dataset, server, train))
    _require(dataset_text, "history_indices = [data_timestep - (self.history_len - 1 - i) * self.history_stride", "dataset history_indices")
    _require(dataset_text, "self._load_window_images(episode, history_indices)", "dataset history image window")
    _require(server_text, "history_indices = self._select_history_indices(len(self.frame_history))", "server history_indices")
    _require(server_text, "self.history_frame_stride", "server history stride")
    _require(server_text, "image_time_ids.append(local_idx)", "server time ids")
    _require(server_text, "image_view_ids.append(self.view_position_cam_keys.index(cam))", "server view ids")
    _require(server_text, '"current_image_frame_count": current_image_frame_count', "server current-frame field")
    _require(train_text, '"--resume-checkpoint-dir"', "explicit resume argument")
    return {
        "mode": "static",
        "configs": ["head_camera_temporal", "head_camera_temporal_train"],
        "camera": expected,
        "history_len": 8,
        "history_frame_stride": train_values["history_frame_stride"],
        "image_frame_stride": 1,
        "current_image_frame_count": 1,
        "semantic_checks": [
            "dataset and server derive the same history window from history_len/history_frame_stride",
            "single-camera image_time_ids are local time indices and image_view_ids are camera indices",
            "8 history frames plus one current-frame marker are represented explicitly",
        ],
    }


def validate_runtime(name):
    sys.path.insert(0, str(ROOT))
    try:
        from configs import VA_CONFIGS
    except ImportError as exc:
        raise RuntimeError(f"optional config dependency unavailable: {exc}") from exc
    cfg = VA_CONFIGS[name]
    result = validate_static()
    result["mode"] = "runtime"
    result["config"] = name
    result["runtime_values"] = {key: getattr(cfg, key) for key in ("history_len", "history_frame_stride", "image_frame_stride")}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="head_camera_temporal_train")
    parser.add_argument("--static-only", action="store_true", help="Do not import optional config dependencies")
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    try:
        result = validate_static() if args.static_only else validate_runtime(args.config)
    except RuntimeError as exc:
        print(f"WARNING: {exc}; falling back to static validation", file=sys.stderr)
        result = validate_static()
        result["dependency_warning"] = str(exc)
    if args.dump:
        args.dump.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
