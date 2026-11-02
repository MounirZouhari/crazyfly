"""Could TD-MPC2 train faster with many simulated drones at once? Measured costs, projected speed.

TD-MPC2 trains with one simulated drone; PPO uses 2,048. With N drones in one
batch, a training step has three parts:
  simulate   step N drones together on the GPU and copy the results back
  plan       choose N actions together (MPPI over N drones in one call)
  update     learning updates; TD-MPC2 does one per simulated step by default
             (updates-to-data ratio, UTD = 1), so N per batched step
This times each part for several N (random weights: timing does not depend on
what was learned) and projects environment steps per second for several UTD
values. UTD < 1 is a different algorithm (fewer updates per sample) and is
labelled as such. Nothing is trained.

    .venv/bin/python code/src/tdmpc2_training/bench_parallel.py --out outputs/batch2/parallel_projection

Writes timings.csv, projection.csv and projection.png to --out.
"""
import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir

from constants import BASE_HOVER_THRUST, TARGET_POS
from crazyflie_hover import HoverDrone, place_drone, sample_training_start
from tdmpc2_jax.build import build_agent

DRONE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]
UTD_RATIOS = [1.0, 0.5, 0.25, 0.125]


def median_ms(fn, calls, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config_hover_standard")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/parallel_projection"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with initialize_config_dir(config_dir=str(HERE / "tdmpc2_jax"), version_base=None):
        cfg = compose(config_name=args.config)
    model_key, encoder_key = jax.random.split(jax.random.PRNGKey(0))
    agent = build_agent(cfg, observation=np.zeros((1, 31), np.float32), action_dim=4,
                        model_key=model_key, encoder_key=encoder_key)
    drone = HoverDrone(metric_target=True)
    start_all = jax.jit(jax.vmap(lambda key, start: place_drone(
        drone, drone.reset(key, random_reset=False, target=TARGET_POS), start, TARGET_POS)))
    step_all = jax.jit(jax.vmap(drone.step))
    rng = np.random.default_rng(0)
    device = jax.devices()[0].device_kind

    # One learning update (batch from the config), independent of the number of drones.
    h, b = agent.horizon, agent.batch_size
    batch = dict(observations=jnp.asarray(rng.normal(size=(h, b, 31)), jnp.float32),
                 actions=jnp.asarray(rng.uniform(-1, 1, size=(h, b, 4)), jnp.float32),
                 rewards=jnp.asarray(rng.normal(size=(h, b)) * 0.01, jnp.float32),
                 next_observations=jnp.asarray(rng.normal(size=(h, b, 31)), jnp.float32),
                 terminated=jnp.zeros((h, b), bool), truncated=jnp.zeros((h, b), bool))
    state = {"agent": agent}

    def one_update():
        state["agent"], _ = state["agent"].update(**batch, key=jax.random.PRNGKey(1))
        jax.block_until_ready(state["agent"])
    update_ms = median_ms(one_update, calls=30)
    print(f"{device}, {args.config}: one learning update {update_ms:.2f} ms", flush=True)

    rows = []
    hover = np.tile(np.array([BASE_HOVER_THRUST, 0, 0, 0], np.float32), (1, 1))
    for n in DRONE_COUNTS:
        starts = np.array([sample_training_start(rng) for _ in range(n)], np.float32)
        states = {"s": start_all(jax.random.split(jax.random.PRNGKey(n), n), jnp.asarray(starts))}
        command = jnp.asarray(np.repeat(hover, n, axis=0))

        def simulate():
            states["s"] = step_all(states["s"], command)
            jax.device_get((states["s"].obs, states["s"].reward, states["s"].done))
        simulate_ms = median_ms(simulate, calls=50)

        obs = np.asarray(states["s"].obs)
        plan = {"p": (jnp.zeros((n, agent.horizon, 4)), jnp.full((n, agent.horizon, 4), agent.max_plan_std))}
        keys = iter(np.asarray(jax.random.split(jax.random.PRNGKey(2), 200)))

        def plan_actions():
            action, plan["p"] = agent.act(obs, prev_plan=plan["p"], deterministic=False, train=True, key=next(keys))
            np.asarray(action)
        plan_ms = median_ms(plan_actions, calls=30)

        row = {"device": device, "config": args.config, "drones": n, "simulate_ms": round(simulate_ms, 3),
               "plan_ms": round(plan_ms, 3), "update_ms": round(update_ms, 3)}
        for utd in UTD_RATIOS:
            per_batch_ms = simulate_ms + plan_ms + n * utd * update_ms
            row[f"steps_per_s_utd{utd:g}"] = round(1000 * n / per_batch_ms, 1)
        rows.append(row)
        print(", ".join(f"{k} {v}" for k, v in row.items()), flush=True)

    with open(args.out / "projection.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ns = [r["drones"] for r in rows]
    for key, label in [("simulate_ms", "simulate N drones"), ("plan_ms", "plan for N drones")]:
        axes[0].plot(ns, [r[key] for r in rows], marker="o", label=label)
    axes[0].axhline(update_ms, color="grey", ls="--", label="one learning update")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("simulated drones per batch (N)")
    axes[0].set_ylabel("time per batched call (ms)")
    axes[0].legend(frameon=False)
    for utd in UTD_RATIOS:
        label = f"UTD {utd:g}" + (" (TD-MPC2 default)" if utd == 1 else " (fewer updates: different algorithm)")
        axes[1].plot(ns, [r[f"steps_per_s_utd{utd:g}"] for r in rows], marker="o", label=label)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("simulated drones per batch (N)")
    axes[1].set_ylabel("projected environment steps per second")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"TD-MPC2 ({args.config}) on {device}: measured parts, projected training speed "
                 "(no overlap between parts assumed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "projection.png", dpi=150)
    print(f"Saved to {args.out} ({platform.node()})")


if __name__ == "__main__":
    main()
