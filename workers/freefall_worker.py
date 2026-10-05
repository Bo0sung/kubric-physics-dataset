from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

import kubric as kb
from kubric.renderer.blender import Blender
from kubric.simulator.pybullet import PyBullet

from physics_dataset.trajectory import write_morpheus_trajectory


LOGGER = logging.getLogger("freefall_worker")
EARTH_GRAVITY = -9.81


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one Kubric free-fall variant.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["normal", "violation"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--step-rate", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--samples-per-pixel", type=int, default=32)
    parser.add_argument("--render-device", choices=["CPU", "CUDA", "OPTIX"], default="CPU")
    parser.add_argument("--intervention", choices=["gravity_scale", "velocity_kick", "teleport", "freeze"], default="gravity_scale")
    parser.add_argument("--dose", nargs="+", type=float, default=[0.5])
    parser.add_argument("--violation-start", type=int, default=12)
    parser.add_argument("--violation-end", type=int, default=18)
    parser.add_argument("--height", type=float)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--no-render", action="store_true", help="Only simulate and write state GT.")
    return parser


def configure_cycles_device(requested: str) -> str:
    if requested == "CPU":
        try:
            import bpy

            bpy.context.scene.cycles.device = "CPU"
        except Exception:
            pass
        return "CPU"
    try:
        import bpy

        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = requested
        preferences.get_devices()
        enabled = []
        for device in preferences.devices:
            use = device.type == requested
            device.use = use
            if use:
                enabled.append(device.name)
        if not enabled:
            raise RuntimeError(f"no Blender {requested} device was detected")
        bpy.context.scene.cycles.device = "GPU"
        LOGGER.info("Blender rendering device: %s (%s)", requested, enabled)
        return requested
    except Exception as exc:  # Blender GPU availability is host/image dependent.
        LOGGER.warning("Could not enable %s; falling back to CPU: %s", requested, exc)
        return "CPU"


def simulate(
    scene: kb.Scene,
    simulator: PyBullet,
    ball: kb.Sphere,
    floor: kb.Cube,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict], dict[str, Any]]:
    """Step PyBullet once, injecting the intervention at exact frame boundaries.

    Kubric's public ``run`` method is intentionally not called in segments because
    its keyframe-transfer phase writes states back to PyBullet. Segmenting it would
    duplicate the boundary state. This loop follows the official implementation,
    then writes the collected states to Blender exactly once.
    """

    client = simulator._physics_client  # Pinned Kubric backend; needed for step hooks.
    client.setTimeStep(1.0 / scene.step_rate)
    steps_per_frame = scene.step_rate // scene.frame_rate
    if steps_per_frame * scene.frame_rate != scene.step_rate:
        raise ValueError("step-rate must be divisible by fps")
    object_ids = [client.getBodyUniqueId(i) for i in range(client.getNumBodies())]
    animation = {
        object_id: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
        for object_id in object_ids
    }
    collisions: list[dict] = []
    actual: dict[str, Any] = {"requested_dose": args.dose}
    last_frame = args.frames - 1
    start = args.violation_start
    end = min(max(args.violation_end, start), last_frame)
    if args.variant == "violation" and not 1 <= start <= last_frame:
        raise ValueError("violation-start must be in [1, frames-1]")
    saved_velocity: np.ndarray | None = None

    for frame in range(args.frames):
        if args.variant == "violation" and frame == start:
            ball_id = ball.linked_objects[simulator]
            current_position, _ = simulator.get_position_and_rotation(ball_id)
            current_velocity, _ = simulator.get_velocities(ball_id)
            if args.intervention == "gravity_scale":
                scale = float(args.dose[0])
                actual["gravity_before"] = [0.0, 0.0, EARTH_GRAVITY]
                scene.gravity = (0.0, 0.0, EARTH_GRAVITY * scale)
                actual["gravity_during"] = list(scene.gravity)
            elif args.intervention == "velocity_kick":
                dose = np.pad(np.asarray(args.dose[:3], dtype=float), (0, max(0, 3-len(args.dose[:3]))))[:3]
                ball.velocity = np.asarray(current_velocity) + dose
                actual["velocity_before"] = list(current_velocity)
                actual["velocity_after"] = np.asarray(ball.velocity).tolist()
            elif args.intervention == "teleport":
                dose = np.pad(np.asarray(args.dose[:3], dtype=float), (0, max(0, 3-len(args.dose[:3]))))[:3]
                ball.position = np.asarray(current_position) + dose
                actual["position_before"] = list(current_position)
                actual["position_after"] = np.asarray(ball.position).tolist()
            elif args.intervention == "freeze":
                saved_velocity = np.asarray(current_velocity, dtype=float)
                actual["velocity_before"] = saved_velocity.tolist()
                ball.velocity = (0.0, 0.0, 0.0)
                ball.static = True

        if args.variant == "violation" and frame == end + 1:
            if args.intervention == "gravity_scale":
                scene.gravity = (0.0, 0.0, EARTH_GRAVITY)
            elif args.intervention == "freeze":
                ball.static = False
                ball.velocity = saved_velocity if saved_velocity is not None else (0.0, 0.0, 0.0)

        for object_id in object_ids:
            position, quaternion = simulator.get_position_and_rotation(object_id)
            velocity, angular_velocity = simulator.get_velocities(object_id)
            animation[object_id]["position"].append(position)
            animation[object_id]["quaternion"].append(quaternion)
            animation[object_id]["velocity"].append(velocity)
            animation[object_id]["angular_velocity"].append(angular_velocity)

        for substep in range(steps_per_frame):
            for collision in client.getContactPoints():
                body_a, body_b = collision[1], collision[2]
                position_b, contact_normal_b = collision[6], collision[7]
                normal_force = collision[9]
                if normal_force > 1e-6:
                    collisions.append({
                        "instances": (
                            simulator._obj_idx_to_asset(body_b),
                            simulator._obj_idx_to_asset(body_a),
                        ),
                        "position": position_b,
                        "contact_normal": contact_normal_b,
                        "frame": frame + substep / steps_per_frame,
                        "force": normal_force,
                    })
            client.stepSimulation()

    assets_animation = {
        asset: animation[asset.linked_objects[simulator]]
        for asset in scene.assets
        if asset.linked_objects.get(simulator) in object_ids
    }
    # Transfer recorded simulation state to Blender only after all physics steps.
    for asset, values in assets_animation.items():
        for frame in range(args.frames):
            for key in ("position", "quaternion", "velocity", "angular_velocity"):
                setattr(asset, key, values[key][frame])
                asset.keyframe_insert(key, frame)

    ball_animation = assets_animation[ball]
    position = np.asarray(ball_animation["position"], dtype=np.float32)
    velocity = np.asarray(ball_animation["velocity"], dtype=np.float32)
    quaternion = np.asarray(ball_animation["quaternion"], dtype=np.float32)
    angular_velocity = np.asarray(ball_animation["angular_velocity"], dtype=np.float32)

    dt = 1.0 / args.fps
    acceleration = np.gradient(velocity, dt, axis=0).astype(np.float32)
    contact = np.zeros(args.frames, dtype=bool)
    for event in collisions:
        if ball in event["instances"] and floor in event["instances"]:
            index = min(max(int(np.floor(event["frame"])), 0), args.frames - 1)
            contact[index] = True
    acceleration_valid = np.ones(args.frames, dtype=bool)
    invalid_centres = list(np.flatnonzero(contact))
    if args.variant == "violation":
        invalid_centres.append(start)
        if args.intervention in ("gravity_scale", "freeze") and end < last_frame:
            invalid_centres.append(end + 1)
    for centre in invalid_centres:
        acceleration_valid[max(0, centre - 1):min(args.frames, centre + 2)] = False
    state = {
        "frame": np.arange(args.frames, dtype=np.int32),
        "time": np.arange(args.frames, dtype=np.float32) / args.fps,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "acceleration_valid": acceleration_valid,
        "quaternion": quaternion,
        "angular_velocity": angular_velocity,
        "contact_ball_floor": contact,
    }
    return state, collisions, actual


def save_video(data_stack: dict[str, np.ndarray], output_dir: Path, fps: int) -> None:
    rgba = np.asarray(data_stack["rgba"])
    rgb = rgba[..., :3]
    imageio.mimsave(output_dir / "rgb.mp4", rgb, fps=fps, macro_block_size=None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scene = kb.Scene(resolution=(args.resolution, args.resolution))
    scene.frame_start = 0
    scene.frame_end = args.frames - 1
    scene.frame_rate = args.fps
    scene.step_rate = args.step_rate
    scene.gravity = (0.0, 0.0, EARTH_GRAVITY)

    simulator = PyBullet(scene, scratch_dir=scratch_dir)
    renderer = Blender(
        scene,
        scratch_dir=scratch_dir,
        use_denoising=True,
        samples_per_pixel=args.samples_per_pixel,
    )
    actual_render_device = configure_cycles_device(args.render_device)

    floor = kb.Cube(
        name="floor", scale=(4.0, 4.0, 0.1), position=(0.0, 0.0, -0.1),
        static=True, background=True, friction=0.5, restitution=0.2,
    )
    floor.material = kb.PrincipledBSDFMaterial(color=kb.Color(0.35, 0.38, 0.42))
    scene += floor

    height = float(args.height if args.height is not None else rng.uniform(3.5, 5.0))
    color = kb.Color(float(rng.uniform(0.15, 0.95)), float(rng.uniform(0.15, 0.95)), float(rng.uniform(0.15, 0.95)))
    ball = kb.Sphere(
        name="ball", scale=args.radius, position=(0.0, 0.0, height),
        velocity=(0.0, 0.0, 0.0), mass=1.0, friction=0.4, restitution=0.65,
    )
    ball.material = kb.PrincipledBSDFMaterial(color=color, roughness=0.35)
    scene += ball

    scene += kb.DirectionalLight(
        name="sun", position=(-3.0, -4.0, 8.0), look_at=(0.0, 0.0, 2.0), intensity=2.5
    )
    scene += kb.RectAreaLight(
        name="fill", position=(4.0, -1.0, 6.0), look_at=(0.0, 0.0, 2.0),
        intensity=250.0, width=3.0, height=3.0,
    )
    scene.camera = kb.PerspectiveCamera(
        name="camera", position=(7.0, -11.0, 4.2), focal_length=45.0,
    )
    scene.camera.look_at((0.0, 0.0, 2.2))

    state, collisions, actual = simulate(scene, simulator, ball, floor, args)
    np.savez_compressed(output_dir / "state.npz", **state)

    render_keys: list[str] = []
    trajectory_gt: dict[str, Any] | None = None
    freefall_trajectory_gt: dict[str, Any] | None = None
    if not args.no_render:
        data_stack = renderer.render()
        render_keys = sorted(data_stack)
        kb.write_image_dict(data_stack, output_dir)
        save_video(data_stack, output_dir, args.fps)
        trajectory_gt = write_morpheus_trajectory(
            output_dir / "trajectory_gt.npz",
            data_stack["segmentation"],
            data_stack.get("depth"),
            fps=args.fps,
            world_position=state["position"],
            metadata={"source": "kubric_render_gt", "variant": args.variant},
        )
        contact_frames = np.flatnonzero(state["contact_ball_floor"])
        freefall_end = int(contact_frames[0]) if len(contact_frames) else args.frames
        if freefall_end >= 8:
            freefall_trajectory_gt = write_morpheus_trajectory(
                output_dir / "trajectory_freefall_gt.npz",
                data_stack["segmentation"][:freefall_end],
                data_stack.get("depth")[:freefall_end] if data_stack.get("depth") is not None else None,
                fps=args.fps,
                world_position=state["position"][:freefall_end],
                metadata={
                    "source": "kubric_render_gt",
                    "variant": args.variant,
                    "segment": "pre_contact_free_fall",
                    "end_frame_exclusive": freefall_end,
                },
            )
        renderer.save_state(output_dir / "scene.blend")

    collision_records = []
    for event in collisions:
        names = [obj.name if obj is not None else "background" for obj in event["instances"]]
        collision_records.append({
            "instances": names,
            "frame": float(event["frame"]),
            "force": float(event["force"]),
            "position": list(event["position"]),
            "contact_normal": list(event["contact_normal"]),
        })
    metadata = {
        "schema_version": 1,
        "phenomenon": "free_fall",
        "variant": args.variant,
        "seed": args.seed,
        "frames": args.frames,
        "fps": args.fps,
        "step_rate": args.step_rate,
        "resolution": [args.resolution, args.resolution],
        "object": {"name": "ball", "radius": args.radius, "mass": 1.0, "initial_height": height},
        "intervention": {
            "type": None if args.variant == "normal" else args.intervention,
            "start_frame": None if args.variant == "normal" else args.violation_start,
            "end_frame": None if args.variant == "normal" else args.violation_end,
            **actual,
        },
        "camera": kb.get_camera_info(scene.camera),
        # List order corresponds to non-zero instance IDs in the segmentation pass.
        "instances": kb.get_instance_info(scene),
        "render": {"requested_device": args.render_device, "actual_device": actual_render_device, "layers": render_keys},
        "trajectory_gt": trajectory_gt,
        "freefall_trajectory_gt": freefall_trajectory_gt,
        "collisions": collision_records,
    }
    kb.write_json(metadata, output_dir / "metadata.json")
    (output_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    LOGGER.info("Wrote %s", output_dir)


if __name__ == "__main__":
    main()
