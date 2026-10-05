from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate_variant(path: Path, expect_render: bool) -> list[str]:
    errors: list[str] = []
    for filename in ("state.npz", "metadata.json", "_SUCCESS"):
        if not (path / filename).is_file():
            errors.append(f"missing {path / filename}")
    if expect_render:
        for filename in (
            "rgb.mp4", "scene.blend", "trajectory_gt.npz", "trajectory_freefall_gt.npz"
        ):
            if not (path / filename).is_file():
                errors.append(f"missing {path / filename}")
    if not (path / "state.npz").is_file():
        return errors
    with np.load(path / "state.npz") as state:
        required = {
            "frame", "time", "position", "velocity", "acceleration",
            "acceleration_valid", "quaternion", "angular_velocity", "contact_ball_floor",
        }
        missing = required - set(state.files)
        if missing:
            errors.append(f"missing state arrays: {sorted(missing)}")
        elif not (
            len(state["frame"]) == len(state["position"]) == len(state["velocity"])
        ):
            errors.append("state arrays have inconsistent frame counts")
        elif not np.isfinite(state["position"]).all():
            errors.append("position contains non-finite values")
    for trajectory_path in (
        path / "trajectory_gt.npz",
        path / "trajectory_freefall_gt.npz",
    ):
        if not expect_render or not trajectory_path.is_file():
            continue
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            required = {"fps", "experiment", "object_ids", "object_1", "visibility"}
            missing = required - set(trajectory.files)
            if missing:
                errors.append(f"missing trajectory arrays in {trajectory_path.name}: {sorted(missing)}")
            elif trajectory["object_1"].shape != (len(trajectory["visibility"]), 3):
                errors.append(f"{trajectory_path.name} object_1 must have shape [frames, 3]")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir")
    parser.add_argument("--violation-start", type=int, required=True)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    scene = Path(args.scene_dir)
    errors = validate_variant(scene / "normal", not args.no_render)
    errors += validate_variant(scene / "violation", not args.no_render)
    if not errors:
        with np.load(scene / "normal/state.npz") as normal, np.load(scene / "violation/state.npz") as violation:
            pre = slice(0, args.violation_start)
            if not np.allclose(normal["position"][pre], violation["position"][pre], atol=1e-6):
                errors.append("paired trajectories differ before intervention")
        if not (scene / "pair_metrics.json").is_file():
            errors.append("missing pair_metrics.json")
    report = {"valid": not errors, "errors": errors, "scene_dir": str(scene)}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
