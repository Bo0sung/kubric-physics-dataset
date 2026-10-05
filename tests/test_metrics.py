import numpy as np

from physics_dataset.metrics import compute_pair_metrics
from physics_dataset.trajectory import extract_image_trajectory


def test_pair_metrics_detects_intervention(tmp_path):
    frames = np.arange(6)
    normal_position = np.zeros((6, 3), dtype=np.float32)
    violation_position = normal_position.copy()
    violation_position[3:, 0] = [0.5, 1.0, 1.5]
    normal_velocity = np.zeros((6, 3), dtype=np.float32)
    violation_velocity = normal_velocity.copy()
    violation_velocity[3:, 0] = 0.5
    normal = tmp_path / "normal.npz"
    violation = tmp_path / "violation.npz"
    np.savez(normal, frame=frames, position=normal_position, velocity=normal_velocity)
    np.savez(violation, frame=frames, position=violation_position, velocity=violation_velocity)

    result = compute_pair_metrics(normal, violation, violation_start=3)

    assert result["first_deviation_frame"] == 3
    assert result["max_position_l2_after_intervention"] == 1.5
    assert len(result["position_l2_per_frame"]) == 6


def test_image_trajectory_selects_moving_instance():
    segmentation = np.zeros((3, 8, 8), dtype=np.int32)
    segmentation[:, 6:, :] = 1  # Static floor.
    segmentation[0, 1:3, 2:4] = 2
    segmentation[1, 2:4, 2:4] = 2
    segmentation[2, 3:5, 2:4] = 2
    depth = np.full((3, 8, 8), 10.0, dtype=np.float32)
    depth[segmentation == 2] = 3.0

    trajectory, visible, instance_id = extract_image_trajectory(segmentation, depth)

    assert instance_id == 2
    np.testing.assert_allclose(trajectory[:, 0], [1.5, 2.5, 3.5])
    np.testing.assert_allclose(trajectory[:, 1], [2.5, 2.5, 2.5])
    np.testing.assert_allclose(trajectory[:, 2], [3.0, 3.0, 3.0])
    assert visible.tolist() == [True, True, True]
