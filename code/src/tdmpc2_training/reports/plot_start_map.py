"""Draw the Batch 2 start area and every fixed test and validation start, seen from above and from the side.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/reports/plot_start_map.py

Writes code/src/tdmpc2_training/reports/figures/start_map.png.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "gpu_training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from crazyflie_hover import HELD_OUT_XY_HIGH, HELD_OUT_XY_LOW, START_AREA_HIGH, START_AREA_LOW
from eval_snapshots import validation_starts
from hover_eval import TEST_SETS

GROUPS = [("train_area", "#1f6f8b", "training area (50)"), ("held_out", "#e09f3e", "held-out block (30)"),
          ("outside", "#d1495b", "outside (30)"), ("batch1", "#333333", "Batch 1 starts (6)")]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    top, side = axes
    top.add_patch(Rectangle(START_AREA_LOW[:2], *(START_AREA_HIGH[:2] - START_AREA_LOW[:2]),
                            facecolor="#1f6f8b", alpha=0.08, edgecolor="#1f6f8b", lw=1.2, label="training area"))
    top.add_patch(Rectangle(HELD_OUT_XY_LOW, *(HELD_OUT_XY_HIGH - HELD_OUT_XY_LOW),
                            facecolor="#e09f3e", alpha=0.18, edgecolor="#e09f3e", lw=1.2, label="held-out block"))
    val = np.array([p for _, p in validation_starts()])
    top.scatter(val[:, 0], val[:, 1], s=14, marker="x", color="#7b8794", label="validation (40)")
    side.scatter(val[:, 0], val[:, 2], s=14, marker="x", color="#7b8794")
    for name, color, label in GROUPS:
        pts = np.array([p for _, p in TEST_SETS[name]])
        top.scatter(pts[:, 0], pts[:, 1], s=22, color=color, label=label, zorder=3)
        side.scatter(pts[:, 0], pts[:, 2], s=22, color=color, zorder=3)
    for ax in axes:
        ax.scatter([0], [0 if ax is top else 1], marker="*", s=260, color="black", zorder=4)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    top.set_title("seen from above (target: star at x = y = 0)")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_aspect("equal")
    top.set_xlim(-1.6, 1.6)
    top.set_ylim(-1.6, 1.6)
    side.add_patch(Rectangle((START_AREA_LOW[0], START_AREA_LOW[2]), START_AREA_HIGH[0] - START_AREA_LOW[0],
                             START_AREA_HIGH[2] - START_AREA_LOW[2], facecolor="#1f6f8b", alpha=0.08,
                             edgecolor="#1f6f8b", lw=1.2))
    side.set_title("seen from the side (target: star at height 1 m)")
    side.set_xlabel("x (m)")
    side.set_ylabel("height (m)")
    side.set_xlim(-1.6, 1.6)
    side.set_ylim(0, 1.7)
    top.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.suptitle("Where Batch 2 trains and tests: every fixed test start and validation start", fontsize=12)
    fig.tight_layout()
    out = HERE / "figures" / "start_map.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
