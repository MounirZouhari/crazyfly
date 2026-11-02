"""Check the deployment adapter (deploy_controller.py) against the simulator. No drone is involved.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/check_deployment.py <run_dir> [<run_dir> ...]

  1. Crazyflie log units and orders are converted correctly (deg/s, quaternion order)
  2. With the simulator's sensor noise switched off, the adapter's observation
     equals the simulator's at every step of a closed-loop flight that the
     adapter itself flies (frames, units, target scaling, previous-action lag)
  3. Every command stays inside the actuator limits
  4. Decision latency after warm-up (reported, hardware dependent)
  5. The adapter refuses to produce a Crazyflie command
Run it with one Batch 1 run (target offset divided by the start distance) and
one Batch 2 run (offset in metres).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np

import crazyflie_env
from constants import TARGET_POS
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, HoverDrone, place_drone
from deploy_controller import DroneState, TDMPC2Controller, rotation_matrix

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}")
    if not ok:
        failures.append(name)


# 1. Units and orders ---------------------------------------------------------
s = np.sin(np.pi / 4)
state = DroneState.from_crazyflie_log(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.0, 0.0, s, s, 180.0, 0.0, -90.0)
check("quaternion reordered from (qx, qy, qz, qw) to (w, x, y, z)", np.allclose(state.quaternion, [s, 0, 0, s]))
check("gyro converted from deg/s to rad/s", np.allclose(state.angular_velocity, [np.pi, 0.0, -np.pi / 2]))
check("90 degree yaw turns the body x axis into world y",
      np.allclose(rotation_matrix(state.quaternion) @ [1, 0, 0], [0, 1, 0]))

# 2-4. Closed loop against the simulator, noise off -------------------------------
# The simulator draws its sensor noise in _get_obs; switch it off so the
# observations can be compared exactly. Must happen before anything is compiled.
crazyflie_env.clipped_normal = lambda rng, shape, std, std_clip=3.0: jnp.zeros(shape)
crazyflie_env.add_quat_noise = lambda rng, quat, angle_std=0.01, std_clip=3.0: (quat, rng)

for run_dir in sys.argv[1:]:
    for use_planner in (True, False):
        controller = TDMPC2Controller(run_dir, use_planner=use_planner)
        drone = HoverDrone(metric_target=controller.metric_target)
        start_flight = jax.jit(lambda key, start: place_drone(
            drone, drone.reset(key, random_reset=False, target=TARGET_POS), start, TARGET_POS))
        step = jax.jit(drone.step)
        label = f"{Path(run_dir).name}, {'planner' if use_planner else 'policy only'}, " \
                f"offset in {'metres' if controller.metric_target else 'start distances'}"
        worst, in_limits, steps = 0.0, True, 0
        for start in [(0.0, 0.0, 0.2), (0.4, -0.3, 0.6)]:
            sim = start_flight(jax.random.PRNGKey(1), jnp.asarray(start, dtype=jnp.float32))
            controller.reset(start_position=start)
            for _ in range(150):
                state = DroneState.from_simulator(sim.data.qpos, sim.data.qvel)
                worst = max(worst, float(np.abs(controller.observation(state) - np.asarray(sim.obs)).max()))
                command = controller.decide(state)
                in_limits &= bool(np.all(command >= ACTION_LOW) and np.all(command <= ACTION_HIGH))
                sim = step(sim, jnp.asarray(command, dtype=jnp.float32))
                steps += 1
        check(f"observation identical to the simulator's over {steps} closed-loop steps ({label})",
              worst < 1e-4, f"largest difference {worst:.2e}")
        check(f"commands inside the actuator limits ({label})", in_limits)

        compile_ms = controller.warm_up()
        hover_state = DroneState(np.array([0.0, 0.0, 1.0]), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3))
        for _ in range(200):
            controller.decide(hover_state)
        lat = np.array(controller.latencies_ms)
        print(f"INFO  decision time ({label}): first call {compile_ms / 1000:.1f} s (compiling), then median "
              f"{np.median(lat):.2f} ms, 99th percentile {np.percentile(lat, 99):.2f} ms, worst {lat.max():.2f} ms "
              f"on {jax.devices()[0].device_kind}")

# 5. No hardware command ----------------------------------------------------------
try:
    TDMPC2Controller.to_crazyflie_command(np.zeros(4))
    check("refuses to produce a Crazyflie command", False)
except NotImplementedError:
    check("refuses to produce a Crazyflie command", True)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("All deployment checks passed.")
