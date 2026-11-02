"""Check the Batch 2 task code before any training. Nothing is trained.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/check_batch2.py

Each check prints PASS or FAIL; the script exits with an error if any fails.
  1. training starts: inside the area, never in the held-out block, never
     closer than 0.3 m to the target, spread over the whole area
  2. with both options off, the observation is exactly the lab's original one
  3. the metric option changes only the 3 target-offset numbers, and gives the
     offset in metres, both after reset and after flying
  4. the metric option changes nothing about physics or reward
  5. the training environment draws a new start every flight, reproducibly
  6. the test starts are fixed, the right size, and where they should be
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np

from constants import SCENE_PATH_1_DRONE, TARGET_POS
from crazyflie_env import CrazyflieEnv
from crazyflie_hover import (HELD_OUT_XY_HIGH, HELD_OUT_XY_LOW, MIN_GOAL_DISTANCE, START_AREA_HIGH,
                             START_AREA_LOW, CrazyflieHoverEnv, HoverDrone, in_held_out_block,
                             place_drone, sample_training_start)
from hover_eval import CURVE_STARTS, OUTSIDE_MAX, OUTSIDE_MIN, TEST_SETS, TEST_STARTS, _make_test_sets

TARGET = np.asarray(TARGET_POS, dtype=np.float64)
OFFSET = slice(18, 21)        # target offset in body axes inside the 31 observation numbers
NOISE_TOLERANCE = 0.06        # position noise is clipped at 3 x 1 cm per axis
HOVER_THRUST = 0.26487        # constants.BASE_HOVER_THRUST
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}")
    if not ok:
        failures.append(name)


def true_start(env):
    return np.asarray(env._state.data.qpos[:3], dtype=np.float64)


# 1. The training distribution --------------------------------------------
rng = np.random.default_rng(0)
samples = np.array([sample_training_start(rng) for _ in range(20000)])
inside = np.all((samples >= START_AREA_LOW) & (samples <= START_AREA_HIGH), axis=1)
check("starts inside the training area", inside.all())
check("no start in the held-out block", not any(in_held_out_block(s) for s in samples))
distance = np.linalg.norm(samples - TARGET, axis=1)
check("every start at least 0.3 m from the target", distance.min() >= MIN_GOAL_DISTANCE,
      f"closest {distance.min():.3f} m")
# Coverage: every 0.25 m x 0.25 m column of the area outside the held-out block is used.
cells = {(int(np.floor((x + 0.75) / 0.25)), int(np.floor((y + 0.75) / 0.25))) for x, y, _ in samples}
expected = {(i, j) for i in range(6) for j in range(6) if not (i >= 4 and j >= 4)}
check("starts cover the whole area outside the block", expected <= cells,
      f"{len(expected & cells)}/{len(expected)} columns used")
check("heights spread over 0.2-1.5 m", samples[:, 2].min() < 0.25 and samples[:, 2].max() > 1.45,
      f"{samples[:, 2].min():.2f}-{samples[:, 2].max():.2f} m")

# 2. Options off = the lab's observation ---------------------------------
lab = CrazyflieEnv(scene_path=SCENE_PATH_1_DRONE, num_drones=1)
ours = HoverDrone(metric_target=False)
key = jax.random.PRNGKey(3)


def start_at(drone):
    return jax.jit(lambda k: place_drone(drone, drone.reset(k, random_reset=False, target=TARGET_POS),
                                         jnp.array([0.4, -0.3, 0.6]), TARGET_POS))


state_lab, state_ours = start_at(lab)(key), start_at(ours)(key)
same_reset = np.array_equal(np.asarray(state_lab.obs), np.asarray(state_ours.obs))
action = jnp.array([HOVER_THRUST, 0.1, -0.1, 0.0])
step_lab, step_ours = jax.jit(lab.step), jax.jit(ours.step)
for _ in range(5):
    state_lab = step_lab(state_lab, action)
    state_ours = step_ours(state_ours, action)
same_step = np.array_equal(np.asarray(state_lab.obs), np.asarray(state_ours.obs))
check("options off: observation identical to the lab's environment", same_reset and same_step)

# 3. Metric target offset --------------------------------------------------
plain = CrazyflieHoverEnv(metric_target=False)
metric = CrazyflieHoverEnv(metric_target=True)
start = np.array([0.6, -0.5, 0.3])
obs_plain, _ = plain.reset(seed=11, options={"start": start})
obs_metric, _ = metric.reset(seed=11, options={"start": start})
offset = TARGET - start
others_equal = np.array_equal(np.delete(obs_plain, range(18, 21)), np.delete(obs_metric, range(18, 21)))
check("metric option changes only the 3 target-offset numbers", others_equal)
error_m = np.abs(obs_metric[OFFSET] - offset).max()
check("metric offset equals target minus position, in metres", error_m < NOISE_TOLERANCE,
      f"largest difference {error_m:.3f} m")
scaled = obs_plain[OFFSET] * np.linalg.norm(offset)
check("without the option the offset is divided by the start distance (as in Batch 1)",
      np.abs(scaled - offset).max() < NOISE_TOLERANCE)

# 4. Physics and reward do not depend on the observation option ------------
rewards_plain, rewards_metric, heights_plain, heights_metric = [], [], [], []
for step in range(60):
    act = np.array([HOVER_THRUST * 1.05, 0.2 * np.sin(step / 7), 0.1, 0.0], dtype=np.float32)
    _, r_p, *_ , info_p = plain.step(act)
    obs_m, r_m, *_, info_m = metric.step(act)
    rewards_plain.append(r_p)
    rewards_metric.append(r_m)
    heights_plain.append(info_p["height"])
    heights_metric.append(info_m["height"])
check("identical flight and reward with and without the metric option",
      rewards_plain == rewards_metric and heights_plain == heights_metric)
position = np.asarray(metric._state.data.qpos[:3], dtype=np.float64)
# The drone is close to level, so body axes and world axes nearly agree.
error_m = np.abs(obs_m[OFFSET] - (TARGET - position)).max()
check("metric offset still in metres after flying 60 steps", error_m < 0.1, f"largest difference {error_m:.3f} m")

# 5. Training environment draws a new start every flight -------------------
default_env = CrazyflieHoverEnv()
default_env.reset(seed=0)
check("default environment still starts at (0, 0, 0.2)",
      np.allclose(true_start(default_env), [0.0, 0.0, 0.2]))
train_env = CrazyflieHoverEnv(random_starts=True, metric_target=True)
train_env.reset(seed=5)
starts = [true_start(train_env)]
for _ in range(99):
    train_env.reset()
    starts.append(true_start(train_env))
starts = np.array(starts)
check("random-start environment: 100 different starts", len(np.unique(starts.round(6), axis=0)) == 100)
check("random-start environment: all inside the area, none in the block",
      all(np.all(s >= START_AREA_LOW - 1e-6) and np.all(s <= START_AREA_HIGH + 1e-6)
          and not in_held_out_block(s) for s in starts))
again = CrazyflieHoverEnv(random_starts=True, metric_target=True)
again.reset(seed=5)
repeat = [true_start(again)]
for _ in range(9):
    again.reset()
    repeat.append(true_start(again))
check("same seed gives the same sequence of starts", np.allclose(np.array(repeat), starts[:10]))

# 6. Test starts ----------------------------------------------------------
rebuilt = _make_test_sets()
check("test starts are fixed (rebuilt identically)", rebuilt == TEST_SETS)
sizes = {name: len(entries) for name, entries in TEST_SETS.items()}
check("test set sizes", sizes == {"batch1": 6, "train_area": 50, "held_out": 30, "outside": 30}, str(sizes))
xyz = {name: np.array([p for _, p in entries]) for name, entries in TEST_SETS.items()}
check("train_area tests: same rules as training starts",
      all(np.all(p >= START_AREA_LOW) and np.all(p <= START_AREA_HIGH) and not in_held_out_block(p)
          and np.linalg.norm(p - TARGET) >= MIN_GOAL_DISTANCE for p in xyz["train_area"]))
check("held_out tests: all inside the held-out block",
      all(in_held_out_block(p) and START_AREA_LOW[2] <= p[2] <= START_AREA_HIGH[2] for p in xyz["held_out"]))
edge = np.max(np.abs(xyz["outside"][:, :2]), axis=1)
check("outside tests: 0.25-0.75 m beyond the area edge",
      edge.min() >= OUTSIDE_MIN and edge.max() <= OUTSIDE_MAX, f"{edge.min():.2f}-{edge.max():.2f} m from centre")
check("Batch 1 starts unchanged", dict(TEST_SETS["batch1"]) == TEST_STARTS)
labels = [label for entries in TEST_SETS.values() for label, _ in entries]
check("every test start has a unique label", len(labels) == len(set(labels)))
check("learning-curve starts: 21, from all groups", len(CURVE_STARTS) == 21
      and {label.split('/')[0] for label, _ in CURVE_STARTS[1:]} == {"train_area", "held_out", "outside"})

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("All Batch 2 checks passed.")
