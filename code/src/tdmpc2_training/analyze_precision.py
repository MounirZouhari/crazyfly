"""Where the precision comes from: horizontal and vertical late error, and the height offset, per decision rule.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/analyze_precision.py outputs/batch2/random_start_1M \
        outputs/batch2/random_start_small_1M --out outputs/batch2/precision_summary

For each batch and decision rule (with planning, policy network only, 64x2
planner), over the full flights (no crash) from the training-area and
held-out starts: the median late error, its horizontal and vertical parts,
and the median height offset (late height minus the target height of 1 m;
negative means the drone hovers too low). Each seed's median first, then
mean +/- standard error across seeds. Writes summary.md and precision.png.
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ("train_area", "held_out")
TARGET_HEIGHT_M = 1.0
RULES = {"planner": "with planning", "policy_only": "policy network only", "plan64x2": "64x2 planner"}


def mean_se(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), (float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)


def folders(batch):
    prefix = "evaluation_batch2_tests" if batch.name.startswith("batch1") else "evaluation"
    return {"planner": prefix, "policy_only": f"{prefix}_policy_only", "plan64x2": f"{prefix}_plan64x2"}


def seed_medians(path):
    """Medians over one seed's full flights: late error, horizontal, vertical, height offset; and the flight count."""
    rows = [r for r in csv.DictReader(open(path)) if r["crashed"] == "False" and r["test_set"] in GROUPS]
    if not rows:
        return [float("nan")] * 4 + [0]
    column = lambda name: np.array([float(r[name]) for r in rows])
    return [float(np.median(column("late_error_m"))), float(np.median(column("late_error_xy_m"))),
            float(np.median(column("late_error_z_m"))),
            float(np.median(column("late_height_m") - TARGET_HEIGHT_M)), len(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/precision_summary"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = ["# Where the precision comes from (full flights from the training-area and held-out starts)", "",
             "Medians per seed, then mean ± SE across seeds, in cm. Height offset = late height minus 1 m "
             "(negative: too low).", "",
             "| batch | decision rule | seeds | full flights | late error | horizontal | vertical | height offset |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    results = []
    for batch in args.batches:
        for rule, folder in folders(batch).items():
            per_seed = [seed_medians(run / folder / "summary.csv") for run in sorted(batch.glob("seed_*"))
                        if (run / folder / "summary.csv").exists()]
            if not per_seed:
                continue
            stats = [mean_se([100 * s[i] for s in per_seed]) for i in range(4)]
            flights = sum(s[4] for s in per_seed)
            lines.append(f"| {batch.name} | {RULES[rule]} | {len(per_seed)} | {flights} | "
                         + " | ".join(f"{m:.1f} ± {se:.1f}" for m, se in stats) + " |")
            results.append((batch.name, rule, stats))
    if not results:
        raise SystemExit("no final evaluations found in the given batches")

    names = list(dict.fromkeys(name for name, _, _ in results))
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.2), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], names):
        mine = [(rule, stats) for n, rule, stats in results if n == name]
        x = np.arange(len(mine))
        for offset, index, label, color in [(-0.2, 1, "horizontal", "#1f6f8b"), (0.2, 2, "vertical", "#e09f3e")]:
            ax.bar(x + offset, [s[index][0] for _, s in mine], width=0.4, yerr=[s[index][1] for _, s in mine],
                   capsize=3, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([RULES[rule] for rule, _ in mine], fontsize=8)
        ax.axhline(10, color="grey", ls=":", lw=1)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("median late error of full flights (cm)\nmean ± SE across seeds")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Horizontal and vertical parts of the late error (dotted: the 10 cm success threshold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "precision.png", dpi=150)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {args.out / 'summary.md'} and {args.out / 'precision.png'}")


if __name__ == "__main__":
    main()
