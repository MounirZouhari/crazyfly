"""Train the lab's Brax PPO on exactly the TD-MPC2 hover task (fair comparison).

Same task as the TD-MPC2 runs: one drone, start (0, 0, 0.2), target (0, 0, 1),
no physics randomization, 500-step episodes, the existing reward and 31-number
observation, and the same mapping of [-1, 1] actions to drone units. PPO's own
settings are the lab's (gpu_training/ppo_training.py: default_ppo_params),
except the episode length. The lab's original run instead trained on random
starts/targets with physics randomization, 1500-step episodes and raw actions.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/ppo_hover.py --seed 0 --steps 20000000 \
        --out outputs/ppo_fixed_start/seed_00

Writes learning_curve.csv, evaluation/ and ppo_params.pkl, the same layout as a
TD-MPC2 run, so analyze_seeds.py works on PPO batches too. To only test a saved
policy (e.g. the lab's existing checkpoint, which uses raw actions):

    .venv/bin/python code/src/tdmpc2_training/ppo_hover.py --load code/rl_models/gpu/PPO_brax_model_checkpoint.pkl \
        --raw-actions --out outputs/ppo_existing_checkpoint
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
from brax.io import model as brax_model
from mujoco_playground import wrapper

from constants import SCENE_PATH_1_DRONE, TARGET_POS
from crazyflie_env import CrazyflieEnv
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, place_drone
from hover_eval import TEST_STARTS, FlightTester, append_learning_curve, write_summary, write_trajectories
from ppo_training import build_ppo_train_fn, default_ppo_params, load_ppo_policy

EPISODE_LENGTH = 500


def to_drone_units(action):
    """Same mapping as Gymnasium's RescaleAction in the TD-MPC2 runs."""
    low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)
    return jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)


class FixedHoverEnv(CrazyflieEnv):
    """The lab's one-drone environment with the TD-MPC2 task setup."""

    def __init__(self, rescale_actions=True):
        super().__init__(scene_path=SCENE_PATH_1_DRONE, num_drones=1)
        self.rescale_actions = rescale_actions

    def reset(self, rng, random_reset=False, target=TARGET_POS):
        # Brax calls reset(rng) only, so every episode is the fixed setup.
        return super().reset(rng, random_reset=False, target=target)

    def step(self, state, action):
        if self.rescale_actions:
            action = to_drone_units(action)
        return super().step(state, action)   # the base step clips to the drone's limits


class PolicyFlightTester(FlightTester):
    """Same test flights as for TD-MPC2, flown by a PPO policy instead of the planner."""

    def __init__(self, make_policy, rescale_actions=True, flight_steps=EPISODE_LENGTH):
        self.make_policy = make_policy
        self.rescale_actions = rescale_actions
        super().__init__(flight_steps=flight_steps)

    def _fly_batch(self, params, reset_keys, starts, targets, plan_key):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            return place_drone(self.drone, state, start, target)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        policy = self.make_policy(params, deterministic=True)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)

        def control_step(carry, key):
            states, alive = carry
            action, _ = policy(states.obs, key)
            command = to_drone_units(action) if self.rescale_actions else jnp.clip(action, low, high)
            states = step_all(states, command)
            done = states.done > 0
            record = {"position": states.data.qpos[:, :3], "command": command,
                      "reward": states.reward, "done": done, "alive": alive}
            return (states, alive & ~done), record

        keys = jax.random.split(plan_key, self.flight_steps)
        _, trajectory = jax.lax.scan(control_step, (states, jnp.ones(len(starts), dtype=bool)), keys)
        return trajectory


def final_test(make_policy, params, seed, out_dir, rescale_actions, flights_per_start=3):
    """The same 6 starts x 3 flights as eval_hover.py, with the same noise seed rule."""
    tester = PolicyFlightTester(make_policy, rescale_actions)
    names = [name for name in TEST_STARTS for _ in range(flights_per_start)]
    flights = tester.fly(params, names, seed=seed + 2000)
    os.makedirs(out_dir / "evaluation", exist_ok=True)
    rows = write_summary(out_dir / "evaluation" / "summary.csv", flights, seed=seed)
    write_trajectories(out_dir / "evaluation" / "trajectories.csv", flights)
    for name in TEST_STARTS:
        mine = [r for r in rows if r["start"] == name]
        print(f"{name:<8} late error {sum(r['late_error_m'] for r in mine) / len(mine):.3f} m, "
              f"crashes {sum(r['crashed'] for r in mine)}/{len(mine)}")


def train(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    curve_path = out_dir / "learning_curve.csv"
    if curve_path.exists():
        curve_path.unlink()   # a rerun starts a fresh curve

    ppo_params = default_ppo_params(
        num_timesteps=args.steps, episode_length=EPISODE_LENGTH,
        num_evals=args.curve_points, seed=args.seed,
    )
    clock = {"start": None, "testing": 0.0, "tester": None}

    def record_learning_curve(env_steps, make_policy, params):
        now = time.perf_counter()
        if clock["start"] is None:   # first call: training (and compiling) starts now
            clock["start"] = now
        if clock["tester"] is None:
            clock["tester"] = PolicyFlightTester(make_policy)
        flights = clock["tester"].fly(params, ["home"] * 3, seed=args.seed + 1000)
        rows = append_learning_curve(
            curve_path, flights, env_steps=int(env_steps),
            train_seconds=round(now - clock["start"] - clock["testing"], 1),
        )
        late = sum(r["late_error_m"] for r in rows) / len(rows)
        print(f"{int(env_steps):>10} steps: late error {late:.3f} m", flush=True)
        clock["testing"] += time.perf_counter() - now

    train_fn = build_ppo_train_fn(ppo_params)
    make_policy, params, _ = train_fn(
        environment=FixedHoverEnv(),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        policy_params_fn=record_learning_curve,
    )
    brax_model.save_params(str(out_dir / "ppo_params.pkl"), params)
    print(f"Training took {(time.perf_counter() - clock['start']) / 60:.1f} min")
    final_test(make_policy, params, args.seed, out_dir, rescale_actions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000_000, help="environment steps to train")
    # Brax trains at least one ~1M-step batch between two curve points, so a
    # short run needs few points (e.g. --steps 2000000 --curve-points 3).
    parser.add_argument("--curve-points", type=int, default=21,
                        help="test-flight points on the learning curve, including the start")
    parser.add_argument("--load", type=Path, help="only test this saved PPO policy, no training")
    parser.add_argument("--raw-actions", action="store_true",
                        help="policy outputs go to the drone unscaled (how the lab's checkpoint was trained)")
    args = parser.parse_args()

    if args.load:
        env = FixedHoverEnv(rescale_actions=not args.raw_actions)
        make_policy, params = load_ppo_policy(str(args.load), env)
        final_test(make_policy, params, args.seed, args.out, rescale_actions=not args.raw_actions)
    else:
        train(args)


if __name__ == "__main__":
    main()
