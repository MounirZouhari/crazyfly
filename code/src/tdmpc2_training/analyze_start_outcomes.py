"""Where each controller succeeds: the test starts seen from above, coloured by the share of seeds that succeeded.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/analyze_start_outcomes.py outputs/batch2/random_start_1M \
        outputs/batch2/fixed_start_1M --out outputs/batch2/start_outcomes.png

One row per batch, one column per decision rule (with planning, policy
network only). Each dot is one fixed test start of the training-area,
held-out and outside groups; its colour is the share of the batch's seeds
whose flight from that start succeeded (full 10 s, late error below 10 cm),
and a black ring marks starts from which most seeds crashed. The training
area and the held-out block are drawn as in reports/figures/start_map.png.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from crazyflie_hover import HELD_OUT_XY_HIGH, HELD_OUT_XY_LOW, START_AREA_HIGH, START_AREA_LOW

GROUPS = ("train_area", "held_out", "outside")
TOLERANCE_M = 0.10
RULES = [("planner", "with planning"), ("policy_only", "policy network only")]


def outcomes(batch, rule):
    """{(x, y, z): [successes], [crashes]} over the batch's seeds, for one decision rule."""
    prefix = "evaluation_batch2_tests" if batch.name.startswith("batch1") else "evaluation"
    folder = prefix if rule == "planner" else f"{prefix}_{rule}"
    per_start = defaultdict(lambda: ([], []))
    seeds = 0
    for run in sorted(batch.glob("seed_*")):
        path = run / folder / "summary.csv"
        if not path.exists():
            continue
        seeds += 1
        for row in csv.DictReader(open(path)):
            if row["test_set"] not in GROUPS:
                continue
            key = (float(row["start_x"]), float(row["start_y"]), float(row["start_z"]))
            crashed = row["crashed"] == "True"
            per_start[key][0].append((not crashed) and float(row["late_error_m"]) < TOLERANCE_M)
            per_start[key][1].append(crashed)
    return per_start, seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/start_outcomes.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(len(args.batches), len(RULES), figsize=(4.4 * len(RULES) + 1.2, 4.2 * len(args.batches)),
                             squeeze=False, constrained_layout=True)
    dots = None
    for b, batch in enumerate(args.batches):
        for r, (rule, title) in enumerate(RULES):
            ax = axes[b][r]
            ax.add_patch(Rectangle(START_AREA_LOW[:2], *(START_AREA_HIGH[:2] - START_AREA_LOW[:2]), fill=False,
                                   edgecolor="#1f6f8b", lw=1.2))
            ax.add_patch(Rectangle(HELD_OUT_XY_LOW, *(HELD_OUT_XY_HIGH - HELD_OUT_XY_LOW), facecolor="#e09f3e",
                                   alpha=0.15, edgecolor="#e09f3e", lw=1.2))
            per_start, seeds = outcomes(batch, rule)
            if per_start:
                points = np.array(list(per_start))
                success = np.array([np.mean(s) for s, _ in per_start.values()])
                mostly_crashed = np.array([np.mean(c) > 0.5 for _, c in per_start.values()])
                dots = ax.scatter(points[:, 0], points[:, 1], c=success, cmap="RdYlGn", vmin=0, vmax=1, s=38,
                                  edgecolors=np.where(mostly_crashed, "black", "none"), linewidths=1.2, zorder=3)
            ax.scatter([0], [0], marker="*", s=200, color="black", zorder=4)
            ax.set_title(f"{batch.name} ({seeds} seeds)\n{title}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_xlabel("x (m)", fontsize=8)
            ax.set_ylabel("y (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
    if dots is not None:
        fig.colorbar(dots, ax=axes, shrink=0.5, label="share of seeds that succeeded from this start")
    fig.suptitle("Test starts seen from above (black ring: most seeds crashed; star: target)", fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
