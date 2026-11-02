"""Summarise the test-time planning horizon test (eval_horizon.py) across the seeds of one or more batches.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/analyze_horizon.py \
        outputs/batch2/random_start_1M outputs/batch2/random_start_small_1M --out outputs/batch2/horizon_summary

For each batch, horizon, condition and start group: success and crash
shares, averaged per seed first, mean +/- standard error across seeds, and
the paired change relative to the trained horizon of 3 (same agent, starts
and sensor noise). Writes summary.md and horizon.png.
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ["train_area", "outside"]
HORIZONS = [1, 3, 5, 10]
CONDITIONS = ["none", "wind 3 m/s", "mass x1.15"]
TRAINED_HORIZON = 3
TOLERANCE_M = 0.10


def mean_se(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), (float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)


def shares(rows, horizon, condition, group):
    mine = [r for r in rows if int(r["horizon"]) == horizon and r["condition"] == condition and r["test_set"] == group]
    if not mine:
        return float("nan"), float("nan")
    success = np.mean([r["crashed"] == "False" and float(r["late_error_m"]) < TOLERANCE_M for r in mine])
    crash = np.mean([r["crashed"] == "True" for r in mine])
    return float(success), float(crash)


def load(batch):
    seeds = {}
    for run in sorted(batch.glob("seed_*")):
        path = run / "horizon_test" / "summary.csv"
        if path.exists():
            seeds[run.name] = list(csv.DictReader(open(path)))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/horizon_summary"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = ["# Planning horizon changed at test time (same agents, starts and sensor noise)", "",
             f"Trained horizon: {TRAINED_HORIZON} steps (0.06 s). Success = full 10 s flight with late error below "
             "10 cm. Cells: success / crash share, mean ± SE across seeds; in brackets the paired change in success "
             f"vs horizon {TRAINED_HORIZON}.", "",
             "| batch | seeds | condition | horizon | " + " | ".join(GROUPS) + " |", "|---|---:|---|---:|" + "---|" * len(GROUPS)]
    loaded = [(batch, load(batch)) for batch in args.batches]
    loaded = [(batch, seeds) for batch, seeds in loaded if seeds]
    if not loaded:
        raise SystemExit("no horizon_test/summary.csv in any seed of the given batches")
    fig, axes = plt.subplots(len(loaded), len(GROUPS), figsize=(5.2 * len(GROUPS), 3.6 * len(loaded)),
                             squeeze=False, sharey=True)
    for b, (batch, seeds) in enumerate(loaded):
        for condition in CONDITIONS:
            for horizon in HORIZONS:
                cells = []
                for group in GROUPS:
                    per_seed = [shares(rows, horizon, condition, group) for rows in seeds.values()]
                    base = [shares(rows, TRAINED_HORIZON, condition, group) for rows in seeds.values()]
                    s, s_se = mean_se([p[0] for p in per_seed])
                    c, c_se = mean_se([p[1] for p in per_seed])
                    cell = f"{s:.0%} ± {s_se:.0%} / {c:.0%} ± {c_se:.0%}"
                    if horizon != TRAINED_HORIZON:
                        ds, ds_se = mean_se([p[0] - q[0] for p, q in zip(per_seed, base)])
                        cell += f" ({ds:+.0%} ± {ds_se:.0%})"
                    cells.append(cell)
                lines.append(f"| {batch.name} | {len(seeds)} | {condition} | {horizon} | " + " | ".join(cells) + " |")
        for g, group in enumerate(GROUPS):
            ax = axes[b][g]
            for condition, color in zip(CONDITIONS, ["#1f6f8b", "#d1495b", "#e09f3e"]):
                stats = [mean_se([shares(rows, h, condition, group)[0] for rows in seeds.values()]) for h in HORIZONS]
                mean = np.array([m for m, _ in stats])
                se = np.array([e for _, e in stats])
                ax.errorbar(HORIZONS, 100 * mean, yerr=100 * se, marker="o", capsize=3, color=color, label=condition)
            ax.axvline(TRAINED_HORIZON, color="grey", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xticks(HORIZONS)
            ax.set_xticklabels([str(h) for h in HORIZONS])
            ax.set_ylim(-3, 103)
            ax.set_title(f"{batch.name} ({len(seeds)} seeds): {group}", fontsize=9)
            ax.set_xlabel("planning horizon (steps of 0.02 s)")
            if g == 0:
                ax.set_ylabel("success (%)")
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Success against the planning horizon used at test time (trained horizon: dotted line)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "horizon.png", dpi=150)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {args.out / 'summary.md'} and {args.out / 'horizon.png'}")


if __name__ == "__main__":
    main()
