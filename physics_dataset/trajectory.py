from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _squeeze_video_layer(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"expected a [frames, height, width] layer, got {array.shape}")
    return array


def select_moving_instance(segmentation: np.ndarray) -> int:
    """Select the non-background instance with the largest centroid motion."""

    labels = _squeeze_video_layer(segmentation)
    instance_ids = [int(value) for value in np.unique(labels) if int(value) != 0]
    if not instance_ids:
        raise ValueError("segmentation contains no non-background instance")

    ranked: list[tuple[float, int]] = []
    for instance_id in instance_ids:
        centroids = []
        for frame in labels:
            y, x = np.nonzero(frame == instance_id)
            if len(y):
                centroids.append((float(y.mean()), float(x.mean())))
        if len(centroids) < 2:
            motion = 0.0
        else:
            points = np.asarray(centroids, dtype=np.float64)
            motion = float(np.linalg.norm(np.ptp(points, axis=0)))
        ranked.append((motion, instance_id))
    return max(ranked)[1]


def extract_image_trajectory(
    segmentation: np.ndarray,
    depth: np.ndarray | None = None,
    *,
    instance_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return exact rendered-mask centroids in Morpheus [Y, X, depth] order."""

    labels = _squeeze_video_layer(segmentation)
    depth_frames = _squeeze_video_layer(depth) if depth is not None else None
    if depth_frames is not None and depth_frames.shape != labels.shape:
        raise ValueError(
            f"segmentation/depth shape mismatch: {labels.shape} vs {depth_frames.shape}"
        )
    selected_id = instance_id if instance_id is not None else select_moving_instance(labels)
    trajectory = np.full((len(labels), 3), np.nan, dtype=np.float32)
    visible = np.zeros(len(labels), dtype=bool)

    for index, frame in enumerate(labels):
        mask = frame == selected_id
        y, x = np.nonzero(mask)
        if not len(y):
            continue
        visible[index] = True
        trajectory[index, 0] = float(y.mean())
        trajectory[index, 1] = float(x.mean())
        if depth_frames is None:
            trajectory[index, 2] = 0.0
        else:
            valid_depth = np.asarray(depth_frames[index][mask], dtype=np.float64)
            valid_depth = valid_depth[np.isfinite(valid_depth)]
            trajectory[index, 2] = float(np.median(valid_depth)) if len(valid_depth) else np.nan
    return trajectory, visible, int(selected_id)


def write_morpheus_trajectory(
    output_path: str | Path,
    segmentation: np.ndarray,
    depth: np.ndarray | None,
    *,
    fps: float,
    world_position: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a Morpheus ScoreKit-compatible trajectory NPZ."""

    trajectory, visible, instance_id = extract_image_trajectory(segmentation, depth)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "fps": np.array(float(fps)),
        "experiment": np.array("falling_ball"),
        "metadata_json": np.array(json.dumps(metadata or {}, ensure_ascii=False)),
        "object_ids": np.array([1], dtype=np.int32),
        "object_1": trajectory,
        "frame": np.arange(len(trajectory), dtype=np.int32),
        "visibility": visible,
        "segmentation_id": np.array(instance_id, dtype=np.int32),
    }
    if world_position is not None:
        positions = np.asarray(world_position, dtype=np.float32)
        if positions.shape != trajectory.shape:
            raise ValueError(
                f"world_position must have shape {trajectory.shape}, got {positions.shape}"
            )
        payload["world_position"] = positions
    np.savez_compressed(output, **payload)
    return {
        "path": str(output),
        "frames": len(trajectory),
        "visible_frames": int(visible.sum()),
        "segmentation_id": instance_id,
        "coordinate_order": ["row_y", "col_x", "depth"],
    }
