"""Where does one TD-MPC2 training step spend its time? Nothing is saved but numbers.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/profile_training.py config_hover_standard --steps 1000

Mirrors tdmpc2_jax/train.py after the random-action phase, with one simulated
drone: plan an action, step the drone, store the transition and sample a
batch, one learning update. Two passes:
  attribute   waits for the GPU after every part, so each part's time is exact
              (the sum is a little slower than real training)
  realistic   runs like the trainer (the GPU works while Python continues), for
              the true steps per second
Weights are random: the time per step does not depend on what was learned.
Appends one row per pass to --csv. Sample GPU use from outside during the
realistic pass, e.g. nvidia-smi --query-gpu=utilization.gpu --format=csv -lms 250.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import gymnasium as gym
import jax
import numpy as np
from hydra import compose, initialize_config_dir

from crazyflie_hover import CrazyflieHoverEnv
from tdmpc2_jax.build import build_agent
from tdmpc2_jax.data import SequentialReplayBuffer


def make_env():
    env = CrazyflieHoverEnv()
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return gym.wrappers.Autoreset(env)


def run(config_name, steps, mode, warmup=30):
    with initialize_config_dir(config_dir=str(HERE / "tdmpc2_jax"), version_base=None):
        cfg = compose(config_name=config_name)
    env = gym.vector.SyncVectorEnv([make_env])
    obs, _ = env.reset(seed=0)
    a = env.action_space.sample()
    nobs, r, te, tr, _ = env.step(a)
    buffer = SequentialReplayBuffer(
        capacity=20000, vectorized=True, num_envs=1, seed=0,
        dummy_input=dict(observation=obs, action=a, reward=r, next_observation=nobs,
                         terminated=te, truncated=tr))
    model_key, encoder_key = jax.random.split(jax.random.PRNGKey(0))
    agent = build_agent(cfg, observation=obs, action_dim=4, model_key=model_key, encoder_key=encoder_key)
    for _ in range(600):  # enough data to sample batches from
        a = env.action_space.sample()
        nobs, r, te, tr, _ = env.step(a)
        buffer.insert(dict(observation=obs, action=a, reward=r, next_observation=nobs,
                           terminated=te, truncated=tr))
        obs = nobs

    parts = {"plan_ms": 0.0, "simulate_ms": 0.0, "replay_ms": 0.0, "update_ms": 0.0}
    rng = jax.random.PRNGKey(1)
    plan = None
    for i in range(steps + warmup):
        if i == warmup:  # compilation is over
            parts = {k: 0.0 for k in parts}
            loop_start = time.perf_counter()
        rng, act_key, update_key = jax.random.split(rng, 3)
        t0 = time.perf_counter()
        action, plan = agent.act(obs, prev_plan=plan, deterministic=False, train=True, key=act_key)
        action = np.array(action)
        t1 = time.perf_counter()
        nobs, r, te, tr, _ = env.step(action)
        t2 = time.perf_counter()
        buffer.insert(dict(observation=obs, action=action, reward=r, next_observation=nobs,
                           terminated=te, truncated=tr))
        batch = buffer.sample(agent.batch_size, agent.horizon)
        t3 = time.perf_counter()
        agent, _ = agent.update(
            observations=batch["observation"], actions=batch["action"], rewards=batch["reward"],
            next_observations=batch["next_observation"], terminated=batch["terminated"],
            truncated=batch["truncated"], key=update_key)
        if mode == "attribute":
            jax.block_until_ready(agent)
        t4 = time.perf_counter()
        obs = nobs
        if te[0] or tr[0]:
            plan = None
        parts["plan_ms"] += t1 - t0
        parts["simulate_ms"] += t2 - t1
        parts["replay_ms"] += t3 - t2
        parts["update_ms"] += t4 - t3
    jax.block_until_ready(agent)
    total = time.perf_counter() - loop_start
    row = {"config": config_name, "mode": mode, "device": jax.devices()[0].device_kind,
           "steps": steps, "steps_per_s": round(steps / total, 2),
           **{k: round(1000 * v / steps, 3) for k, v in parts.items()},
           "wall_ms_per_step": round(1000 * total / steps, 3)}
    print(", ".join(f"{k} {v}" for k, v in row.items()), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="e.g. config_hover (small) or config_hover_standard")
    parser.add_argument("--steps", type=int, default=1000, help="timed training steps per pass")
    parser.add_argument("--modes", default="attribute,realistic")
    parser.add_argument("--csv", type=Path, default=Path("outputs/profiling/training_step.csv"))
    args = parser.parse_args()

    rows = [run(args.config, args.steps, mode) for mode in args.modes.split(",")]
    os.makedirs(args.csv.parent, exist_ok=True)
    new_file = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {args.csv}")


if __name__ == "__main__":
    main()
