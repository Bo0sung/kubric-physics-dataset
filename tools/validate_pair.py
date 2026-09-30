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
        for filename in ("rgb.mp4", "scene.blend"):
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

