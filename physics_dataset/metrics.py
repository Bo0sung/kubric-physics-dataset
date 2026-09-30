from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def compute_pair_metrics(
    normal_npz: str | Path,
    violation_npz: str | Path,
    *,
    violation_start: int,
    threshold: float = 1e-4,
) -> dict[str, Any]:
    """Compare world-space state arrays from a paired simulation."""

    with np.load(normal_npz) as normal, np.load(violation_npz) as violation:
        if not np.array_equal(normal["frame"], violation["frame"]):
            raise ValueError("normal and violation frame indices differ")
        position_delta = violation["position"] - normal["position"]
        velocity_delta = violation["velocity"] - normal["velocity"]
        position_l2 = np.linalg.norm(position_delta, axis=-1)
        velocity_l2 = np.linalg.norm(velocity_delta, axis=-1)
        frames = normal["frame"].astype(int)

    after = frames >= int(violation_start)
    deviating = np.flatnonzero(position_l2 > threshold)
    first_deviation = int(frames[deviating[0]]) if len(deviating) else None
    metrics = {
        "violation_start_frame": int(violation_start),
        "deviation_threshold": float(threshold),
        "first_deviation_frame": first_deviation,
        "position_l2_per_frame": position_l2,
        "velocity_l2_per_frame": velocity_l2,
        "mean_position_l2_after_intervention": float(position_l2[after].mean()) if after.any() else 0.0,
        "max_position_l2_after_intervention": float(position_l2[after].max()) if after.any() else 0.0,
        "final_position_l2": float(position_l2[-1]),
        "mean_velocity_l2_after_intervention": float(velocity_l2[after].mean()) if after.any() else 0.0,
        "max_velocity_l2_after_intervention": float(velocity_l2[after].max()) if after.any() else 0.0,
    }
    return {key: _jsonable(value) for key, value in metrics.items()}


def write_pair_metrics(
    normal_npz: str | Path,
    violation_npz: str | Path,
    output_json: str | Path,
    *,
    violation_start: int,
) -> dict[str, Any]:
    metrics = compute_pair_metrics(
        normal_npz, violation_npz, violation_start=violation_start
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

