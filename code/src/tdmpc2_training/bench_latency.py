"""Time one TD-MPC2 control decision (planner latency) for one drone.

Nothing is trained and no checkpoint is needed: the time per decision depends
on the network sizes and the planning budget, not on the learned weights.
Each setting is compiled, warmed up, then called one decision at a time the
way a control loop would: observation in from the CPU, action back on the CPU,
the previous plan kept on the device between calls.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/bench_latency.py
    JAX_PLATFORMS=cpu .venv/bin/python code/src/tdmpc2_training/bench_latency.py

Writes outputs/latency_benchmark/<device>.csv. The drone is controlled at
50 Hz (crazyflie_env.py ctrl_dt = 0.02), so every decision has 20 ms at most,
before radio and state-estimation delays are counted.
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
import numpy as np
from hydra import compose, initialize_config_dir

from tdmpc2_jax.build import build_agent
from tdmpc2_jax.tdmpc2 import TDMPC2

OBSERVATION_SIZE = 31
ACTION_SIZE = 4
CONTROL_PERIOD_MS = 20.0

# Network sizes: the two configs used so far plus one in between.
NETWORKS = {
    "small_128": ("config_hover", []),
    "medium_256": ("config_hover_standard", ["world_model.latent_dim=256"]),
    "standard_512": ("config_hover_standard", []),
}
# Planning budgets: (plans per round, rounds, elites, policy-prior plans).
# 128 x 3 is the small config's budget, 512 x 6 the published one.
BUDGETS = {
    "policy_only": None,
    "64x2": (64, 2, 8, 8),
    "128x3": (128, 3, 16, 16),
    "256x4": (256, 4, 32, 24),
    "512x6": (512, 6, 64, 24),
}


def make_agent(config_name, overrides):
    with initialize_config_dir(config_dir=str(HERE / "tdmpc2_jax"), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    model_key, encoder_key = jax.random.split(jax.random.PRNGKey(0))
    return build_agent(
        cfg,
        observation=np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
        action_dim=ACTION_SIZE,
        model_key=model_key,
        encoder_key=encoder_key,
    )


def parameter_counts(agent):
    def count(state):
        return int(sum(x.size for x in jax.tree_util.tree_leaves(state.params)))
    m = agent.model
    parts = {name: count(getattr(m, name)) for name in
             ("encoder", "dynamics_model", "reward_model", "policy_model", "value_model")}
    return {
        "params_learned": sum(parts.values()),
        "params_policy_only": parts["encoder"] + parts["policy_model"],
    }


def estimated_flops(agent, obs, plan, mpc, key):
    """XLA's own operation count for one compiled decision (None if unavailable)."""
    try:
        compiled = TDMPC2.act.lower(agent, obs, prev_plan=plan, mpc=mpc,
                                    deterministic=True, train=False, key=key).compile()
        cost = compiled.cost_analysis()
        if isinstance(cost, (list, tuple)):
            cost = cost[0]
        return float(cost.get("flops", float("nan")))
    except Exception as error:  # cost analysis is best effort
        print(f"  (no FLOP count: {type(error).__name__})")
        return None


def time_decisions(agent, mpc, max_calls, max_seconds, warmup=20):
    rng = np.random.default_rng(0)
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(1), max_calls + warmup))
    plan = None
    if mpc:
        plan = (jnp.zeros((1, agent.horizon, ACTION_SIZE)),
                jnp.full((1, agent.horizon, ACTION_SIZE), agent.max_plan_std))

    compile_start = time.perf_counter()
    latencies = []
    for i in range(max_calls + warmup):
        obs = rng.normal(size=(1, OBSERVATION_SIZE)).astype(np.float32)
        key = keys[i]
        start = time.perf_counter()
        action, new_plan = agent.act(obs, prev_plan=plan, mpc=mpc,
                                     deterministic=True, train=False, key=key)
        action = np.asarray(action)  # waits for the device, like a real loop
        elapsed = time.perf_counter() - start
        if mpc:
            plan = new_plan
        if i == 0:
            compile_seconds = time.perf_counter() - compile_start
        if i >= warmup:
            latencies.append(elapsed * 1000.0)
            if sum(latencies) / 1000.0 > max_seconds and len(latencies) >= 50:
                break
    return np.array(latencies), compile_seconds


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=1000, help="timed decisions per setting")
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="stop timing a setting after this much measured time")
    parser.add_argument("--networks", default="all", help=f"comma list from {list(NETWORKS)}")
    parser.add_argument("--budgets", default="all", help=f"comma list from {list(BUDGETS)}")
    parser.add_argument("--out", type=Path, default=Path("outputs/latency_benchmark"))
    args = parser.parse_args()

    device = jax.devices()[0]
    device_label = device.platform
    hardware = device.device_kind if device.platform == "gpu" else cpu_name()
    print(f"Device: {device.platform} ({hardware}), {os.cpu_count()} CPU threads, jax {jax.__version__}")

    networks = list(NETWORKS) if args.networks == "all" else args.networks.split(",")
    budgets = list(BUDGETS) if args.budgets == "all" else args.budgets.split(",")

    rows = []
    for net in networks:
        config_name, overrides = NETWORKS[net]
        base = make_agent(config_name, overrides)
        counts = parameter_counts(base)
        for budget in budgets:
            setting = BUDGETS[budget]
            mpc = setting is not None
            agent = base
            if mpc:
                population, iterations, elites, prior = setting
                agent = base.replace(population_size=population, mppi_iterations=iterations,
                                     num_elites=elites, policy_prior_samples=prior)
            obs = np.zeros((1, OBSERVATION_SIZE), dtype=np.float32)
            plan = ((jnp.zeros((1, agent.horizon, ACTION_SIZE)),
                     jnp.full((1, agent.horizon, ACTION_SIZE), agent.max_plan_std)) if mpc else None)
            flops = estimated_flops(agent, obs, plan, mpc, jax.random.PRNGKey(0))
            lat, compile_s = time_decisions(agent, mpc, args.calls, args.max_seconds)
            row = {
                "device": device_label, "hardware": hardware, "network": net, "budget": budget,
                **counts,
                "gflop_per_decision": None if flops is None else round(flops / 1e9, 4),
                "calls": len(lat),
                "median_ms": np.median(lat), "p95_ms": np.percentile(lat, 95),
                "p99_ms": np.percentile(lat, 99), "max_ms": lat.max(), "mean_ms": lat.mean(),
                "over_20ms_fraction": float(np.mean(lat > CONTROL_PERIOD_MS)),
                "first_call_s": compile_s,
            }
            rows.append(row)
            print(f"{net:<14} {budget:<12} median {row['median_ms']:8.2f} ms  "
                  f"p99 {row['p99_ms']:8.2f} ms  max {row['max_ms']:8.2f} ms  "
                  f"over 20 ms {row['over_20ms_fraction']:6.1%}  "
                  f"GFLOP {row['gflop_per_decision']}  ({len(lat)} calls)")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{device_label}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
