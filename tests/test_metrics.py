import numpy as np

from physics_dataset.metrics import compute_pair_metrics


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

