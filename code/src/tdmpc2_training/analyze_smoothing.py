"""Summarise the smoothing test (eval_smoothing.py) across the seeds of one or more batches.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/analyze_smoothing.py outputs/batch2/random_start_1M \
        outputs/batch2/random_start_small_1M --out outputs/batch2/smoothing_summary

For each batch, motor lag and variant (training-area and held-out starts
together): success and crash shares, the thrust jitter of the full flights
and the share of steps with a command at its limit, averaged per seed first,
then mean +/- standard error across seeds; and the paired change in success
relative to the planner as trained at the same motor lag. Writes summary.md
and smoothing.png.
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["planner", "planner_mean", "planner_filtered", "policy_only"]
LABELS = {"planner": "planner as trained", "planner_mean": "averaged plan",
          "planner_filtered": "planner, 40 ms filter", "policy_only": "policy network only"}
TOLERANCE_M = 0.10


def mean_se(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), (float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)


def seed_stats(rows, variant, lag):
    mine = [r for r in rows if r["variant"] == variant and np.isclose(float(r["motor_lag_s"]), lag)]
    if not mine:
        return [float("nan")] * 4
    full = [r for r in mine if r["crashed"] == "False"]
    return [float(np.mean([r["crashed"] == "False" and float(r["late_error_m"]) < TOLERANCE_M for r in mine])),
            float(np.mean([r["crashed"] == "True" for r in mine])),
            float(np.nanmean([float(r["thrust_jitter_pct"]) for r in full])) if full else float("nan"),
            float(np.mean([float(r["saturated_pct"]) for r in mine]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/smoothing_summary"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    loaded = []
    for batch in args.batches:
        seeds = [list(csv.DictReader(open(run / "smoothing_test" / "summary.csv")))
                 for run in sorted(batch.glob("seed_*")) if (run / "smoothing_test" / "summary.csv").exists()]
        if seeds:
            loaded.append((batch, seeds))
    if not loaded:
        raise SystemExit("no smoothing_test/summary.csv in any seed of the given batches")

    lines = ["# Smoother ways of acting, and slow motors (training-area and held-out starts)", "",
             "Cells: success / crash share, mean ± SE across seeds; in brackets the paired change in success vs the "
             "planner as trained at the same motor lag. Jitter: mean step-to-step change of the thrust command in "
             "the last 5 s of full flights (% of its range). Saturated: % of steps with a command at its limit.", "",
             "| batch | seeds | motor lag | variant | success / crash | thrust jitter (%) | saturated (%) |",
             "|---|---:|---:|---|---|---:|---:|"]
    fig, axes = plt.subplots(1, len(loaded), figsize=(5.2 * len(loaded), 3.8), squeeze=False, sharey=True)
    for ax, (batch, seeds) in zip(axes[0], loaded):
        lags = sorted({float(r["motor_lag_s"]) for rows in seeds for r in rows})
        for variant in VARIANTS:
            means, errors = [], []
            for lag in lags:
                stats = [seed_stats(rows, variant, lag) for rows in seeds]
                base = [seed_stats(rows, "planner", lag) for rows in seeds]
                s, s_se = mean_se([x[0] for x in stats])
                c, c_se = mean_se([x[1] for x in stats])
                jitter, jitter_se = mean_se([x[2] for x in stats])
                sat, sat_se = mean_se([x[3] for x in stats])
                cell = f"{s:.0%} ± {s_se:.0%} / {c:.0%} ± {c_se:.0%}"
                if variant != "planner":
                    d, d_se = mean_se([x[0] - b[0] for x, b in zip(stats, base)])
                    cell += f" ({d:+.0%} ± {d_se:.0%})"
                lines.append(f"| {batch.name} | {len(seeds)} | {1000 * lag:.0f} ms | {LABELS[variant]} | {cell} | "
                             f"{jitter:.1f} ± {jitter_se:.1f} | {sat:.0f} ± {sat_se:.0f} |")
                means.append(100 * s)
                errors.append(100 * s_se)
            ax.errorbar([1000 * lag for lag in lags], means, yerr=errors, marker="o", capsize=3, label=LABELS[variant])
        ax.set_title(f"{batch.name} ({len(seeds)} seeds)", fontsize=9)
        ax.set_xlabel("motor lag (ms, first-order time constant)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("success (%), training area + held-out")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Smoother commands and slower motors (same agents, starts and sensor noise)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "smoothing.png", dpi=150)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {args.out / 'summary.md'} and {args.out / 'smoothing.png'}")


if __name__ == "__main__":
    main()
