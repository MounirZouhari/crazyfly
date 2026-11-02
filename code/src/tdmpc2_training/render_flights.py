"""Video of a trained TD-MPC2 hover agent flying test flights (MuJoCo rendering).

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/render_flights.py <run_dir> --snapshot 1000000 \\
        --starts train_area/00 held_out/00 outside/00 home --out videos/agent.mp4

Each start is one panel of the video (up to 4, in a 2 x 2 grid). The flights
are flown exactly as in the tests (hover_eval.FlightTester: deterministic
planning, same sensor noise for a given --seed), then replayed in MuJoCo's
renderer: the drone, its path so far (blue dots), the target (red) and the
10 cm success zone around it (green). --no-mpc flies the policy network alone.

Rendering needs no screen (EGL). The video is written with ffmpeg if one is
found (the imageio-ffmpeg package, $FFMPEG, or ffmpeg on the PATH), otherwise
as a GIF. Nothing is trained and no result file is written.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constants import TARGET_POS
from eval_hover import load_agent
from hover_eval import CONTROL_DT, TEST_SETS, TEST_STARTS, TOLERANCE_M, FlightTester, resolve_starts
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, place_drone

PANEL_W, PANEL_H = 480, 360
TRAIL_EVERY = 4            # one path dot every 4 steps (0.08 s)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class RecordingTester(FlightTester):
    """FlightTester that also keeps the full simulator pose (position and
    orientation) of every step, for replay in the renderer."""

    def _fly_batch(self, agent, reset_keys, starts, targets, plan_key):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            return place_drone(self.drone, state, start, target)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)
        n, horizon, action_dim = len(starts), agent.horizon, agent.model.action_dim
        plan = (jnp.zeros((n, horizon, action_dim)),
                jnp.full((n, horizon, action_dim), agent.max_plan_std))

        def control_step(carry, key):
            states, plan, alive = carry
            if self.mpc:
                action, plan = agent.act(states.obs, prev_plan=plan,
                                         deterministic=True, train=False, key=key)
            else:
                action, _ = agent.act(states.obs, mpc=False,
                                      deterministic=True, train=False, key=key)
            command = jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)
            states = step_all(states, command)
            done = states.done > 0
            record = {"qpos": states.data.qpos, "done": done, "alive": alive}
            return (states, plan, alive & ~done), record

        keys = jax.random.split(plan_key, self.flight_steps)
        _, trajectory = jax.lax.scan(control_step, (states, plan, jnp.ones(n, dtype=bool)), keys)
        return trajectory


def resolve_labels(labels):
    """'home', 'train_area/03', ... -> (label, (x, y, z)) pairs."""
    known = {label: xyz for group in TEST_SETS.values() for label, xyz in group}
    known.update(TEST_STARTS)
    return [(label, known[label]) for label in labels]


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def render_panels(model, qpos, alive, starts, target, labels, title):
    """Frames (steps, 2*H, 2*W, 3) of up to four flights replayed side by side."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, PANEL_H, PANEL_W)
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    if target_geom >= 0:
        model.geom_pos[target_geom] = target
    camera = mujoco.MjvCamera()
    font = ImageFont.truetype(FONT, 15)
    small = ImageFont.truetype(FONT, 13)
    steps, n = alive.shape
    frames = []
    for t in range(steps):
        canvas = Image.new("RGB", (2 * PANEL_W, 2 * PANEL_H + 34), (255, 255, 255))
        for i in range(n):
            flown = int(alive[:, i].sum())               # steps before the crash, if any
            shown = min(t, flown - 1) if flown > 0 else 0
            data.qpos[:] = qpos[shown, i]
            mujoco.mj_forward(model, data)
            position = data.qpos[:3].copy()
            # Frame the start and the target together.
            middle = (np.asarray(starts[i]) + target) / 2
            span = np.linalg.norm(np.asarray(starts[i]) - target)
            camera.lookat[:] = middle
            camera.distance = max(1.6, 1.4 * span + 1.0)
            camera.azimuth, camera.elevation = 135.0, -18.0
            renderer.update_scene(data, camera=camera)
            add_sphere(renderer.scene, target, TOLERANCE_M, (0.1, 0.8, 0.2, 0.18))
            for s in range(0, shown + 1, TRAIL_EVERY):
                add_sphere(renderer.scene, qpos[s, i, :3], 0.008, (0.15, 0.4, 0.9, 0.8))
            panel = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(panel)
            distance = np.linalg.norm(position - target)
            crashed = flown < steps and t >= flown - 1
            status = "CRASHED" if crashed else ("within 10 cm" if distance < TOLERANCE_M else "")
            draw.text((8, 6), labels[i], fill=(20, 20, 20), font=font)
            draw.text((8, 26), f"t = {min(t, flown - 1 if crashed else t) * CONTROL_DT + CONTROL_DT:4.2f} s"
                               f"   distance {100 * distance:5.1f} cm", fill=(20, 20, 20), font=small)
            if status:
                draw.text((8, 44), status, font=font,
                          fill=(200, 30, 30) if crashed else (20, 140, 40))
            canvas.paste(panel, ((i % 2) * PANEL_W, 34 + (i // 2) * PANEL_H))
        ImageDraw.Draw(canvas).text((10, 8), title, fill=(10, 10, 10), font=font)
        frames.append(np.asarray(canvas))
    renderer.close()
    return frames


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return os.environ.get("FFMPEG") or shutil.which("ffmpeg")


def write_video(frames, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    height, width = frames[0].shape[:2]
    if ffmpeg and out.suffix == ".mp4":
        command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{width}x{height}", "-r", str(round(1 / CONTROL_DT)), "-i", "-",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
        return out
    gif = out.with_suffix(".gif")
    images = [Image.fromarray(f).resize((width // 2, height // 2)) for f in frames[::2]]
    images[0].save(gif, save_all=True, append_images=images[1:], duration=int(2000 * CONTROL_DT), loop=0)
    return gif


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--snapshot", type=int, default=None, help="snapshot step (default: checkpoint/)")
    parser.add_argument("--starts", nargs="+", default=["train_area/00", "held_out/00", "outside/00", "home"])
    parser.add_argument("--no-mpc", action="store_true", help="policy network only")
    parser.add_argument("--seed", type=int, default=0, help="sensor noise and planner randomness")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.starts) > 4:
        parser.error("at most 4 starts (one panel each)")

    agent, step, cfg = load_agent(args.run_dir, snapshot=args.snapshot)
    tester = RecordingTester(mpc=not args.no_mpc, metric_target=bool(cfg.env.get("metric_target", False)))
    starts = resolve_labels(args.starts)
    labels, coords = resolve_starts(starts)
    targets = np.tile(np.asarray(TARGET_POS, dtype=np.float32), (len(coords), 1))
    reset_key, plan_key = jax.random.split(jax.random.PRNGKey(args.seed))
    trajectory = jax.device_get(tester._fly(agent, jax.random.split(reset_key, len(coords)),
                                            coords, targets, plan_key))

    rule = "policy network only" if args.no_mpc else "with planning"
    title = args.title or f"{args.run_dir.parent.name}/{args.run_dir.name}, {step // 1000}k steps, {rule}"
    frames = render_panels(tester.drone.mj_model, np.asarray(trajectory["qpos"]),
                           np.asarray(trajectory["alive"]), coords, np.asarray(TARGET_POS),
                           labels, title)
    print(f"Wrote {write_video(frames, args.out)} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
