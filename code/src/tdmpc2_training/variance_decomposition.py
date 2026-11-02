"""Why did short TD-MPC2 runs differ so much? Evaluation noise vs real seed differences.

Re-flies every seed's final agent from the Batch 1 training start with several
sensor-noise seeds, and splits the spread of the late error into:
  between seeds   the trained agents really differ (randomness of training)
  within a seed   flights of the same agent differ (sensor noise and planner
                  randomness): the evaluation noise
Nothing is trained. The noise seeds are the same for every agent (paired).

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/variance_decomposition.py \
        outputs/batch1_small_narval outputs/batch1_fixed_start outputs/batch1_standard \
        --out outputs/batch2/variance_decomposition

Writes flights.csv (one row per flight), summary.md and variance.png to --out.
Statistics: one-way random-effects ANOVA per batch; the intraclass correlation
(ICC) is the share of the flight-to-flight variance that is due to the agent.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_hover import load_agent
from hover_eval import TOLERANCE_M, FlightTester

NOISE_SEEDS = [5000, 5001, 5002, 5003, 5004]   # the same for every agent
FLIGHTS_PER_NOISE_SEED = 4


def anova(groups):
    """One-way random-effects ANOVA for equal group sizes -> (between var, within var, ICC)."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    k, n = len(groups), len(groups[0])
    grand = np.mean(np.concatenate(groups))
    ms_between = n * sum((g.mean() - grand) ** 2 for g in groups) / (k - 1)
    ms_within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (k * (n - 1))
    var_between = max(0.0, (ms_between - ms_within) / n)
    icc = var_between / (var_between + ms_within) if var_between + ms_within > 0 else float("nan")
    return var_between, ms_within, icc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+", help="Batch 1 folders with seed_XX runs")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/variance_decomposition"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    tester = FlightTester(flight_steps=500)
    for batch in args.batches:
        for run_dir in sorted(batch.glob("seed_*")):
            agent, step, cfg = load_agent(run_dir)
            for noise_seed in NOISE_SEEDS:
                flights = tester.fly(agent, ["home"] * FLIGHTS_PER_NOISE_SEED, seed=noise_seed)
                for i, flight in enumerate(flights):
                    s = flight.summary()
                    rows.append({"batch": batch.name, "seed": cfg.seed, "checkpoint_step": step,
                                 "noise_seed": noise_seed, "flight": i, "crashed": s["crashed"],
                                 "late_error_m": s["late_error_m"],
                                 "success": (not s["crashed"]) and s["late_error_m"] < TOLERANCE_M})
            print(f"{batch.name}/{run_dir.name}: done", flush=True)

    with open(args.out / "flights.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Evaluation noise vs real differences between seeds (Batch 1 agents)", "",
             f"Every final agent flew {len(NOISE_SEEDS) * FLIGHTS_PER_NOISE_SEED} flights from the Batch 1 "
             f"training start: {len(NOISE_SEEDS)} sensor-noise seeds x {FLIGHTS_PER_NOISE_SEED} flights, "
             "the same noise seeds for every agent.", "",
             "| batch | seeds | late error: between-seed SD (m) | within-seed SD (m) | ICC | "
             "SE of one seed's mean with Batch 1's 3 flights (m) | crash rate: range over seeds |",
             "|---|---:|---:|---:|---:|---:|---|"]
    batch_names = list(dict.fromkeys(r["batch"] for r in rows))
    fig, axes = plt.subplots(1, len(batch_names), figsize=(5.5 * len(batch_names), 4.5), squeeze=False)
    for ax, name in zip(axes[0], batch_names):
        mine = [r for r in rows if r["batch"] == name]
        seeds = sorted(set(r["seed"] for r in mine))
        late = {s: [r["late_error_m"] for r in mine if r["seed"] == s] for s in seeds}
        crash = {s: np.mean([r["crashed"] for r in mine if r["seed"] == s]) for s in seeds}
        var_b, var_w, icc = anova([late[s] for s in seeds])
        lines.append(f"| {name} | {len(seeds)} | {np.sqrt(var_b):.3f} | {np.sqrt(var_w):.3f} | {icc:.2f} | "
                     f"{np.sqrt(var_w / 3):.3f} | {min(crash.values()):.0%} to {max(crash.values()):.0%} |")
        order = sorted(seeds, key=lambda s: np.mean(late[s]))
        for x, s in enumerate(order):
            ax.scatter(np.full(len(late[s]), x) + np.random.default_rng(s).uniform(-0.15, 0.15, len(late[s])),
                       late[s], s=10, color="#a7b1bc", zorder=2)
            ax.scatter(x, np.mean(late[s]), s=45, color="#1f6f8b", zorder=3)
        ax.axhline(TOLERANCE_M, color="black", ls=":", lw=1)
        ax.set_xticks(range(len(order)), [str(s) for s in order])
        ax.set_xlabel("seed (sorted by mean)")
        ax.set_ylabel("late-flight distance to target (m)")
        ax.set_title(f"{name}: ICC {icc:.2f}")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Each grey dot is one flight, each blue dot one agent's mean "
                 f"({len(NOISE_SEEDS) * FLIGHTS_PER_NOISE_SEED} flights, training start)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "variance.png", dpi=150)
    lines += ["", "ICC near 1: the spread comes from the agents themselves (training), not from "
              "evaluation noise. ICC near 0: re-testing the same agent varies as much as changing seed."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
