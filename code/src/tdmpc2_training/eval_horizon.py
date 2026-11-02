"""Does looking further ahead help? The planning horizon changed at test time. Nothing is trained.

TD-MPC2 plans 3 steps (0.06 s) ahead and relies on its value function for
everything after that. This flies the same trained agent with planning
horizons of 1, 3 (as trained), 5 and 10 steps, from the training-area and
outside starts, with nominal physics, in a 3 m/s steady wind, and with 15%
extra mass, all with the same sensor noise. Longer horizons use the world
model further than it was trained to predict (3 steps), so its errors may
compound.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_horizon.py <run_dir>

Writes <run_dir>/horizon_test/summary.csv (one row per flight with horizon
and condition) and prints success and crash shares.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import numpy as np

from eval_disturbances import DisturbedFlightTester, end_cause, wind_profile
from eval_hover import load_agent
from hover_eval import TEST_SETS, TOLERANCE_M, _write_rows

HORIZONS = [1, 3, 5, 10]
CONDITIONS = [("none", {}), ("wind 3 m/s", {"wind": 3.0}), ("mass x1.15", {"mass": 1.15})]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.run_dir / "horizon_test"
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")
    agent, step, cfg = load_agent(args.run_dir)
    metric = bool(cfg.env.get("metric_target", False))
    starts = TEST_SETS["train_area"] + TEST_SETS["outside"]
    tester = DisturbedFlightTester(500, mpc=True, metric_target=metric)

    rows = []
    for horizon in HORIZONS:
        variant = agent.replace(horizon=horizon)
        for name, change in CONDITIONS:
            flights = tester.fly(variant, starts, seed=cfg.seed + 7000,
                                 wind=wind_profile("wind", change.get("wind", 0.0), 500),
                                 mass_factor=change.get("mass", 1.0))
            mine = [{"seed": cfg.seed, "checkpoint_step": step, "horizon": horizon, "condition": name,
                     "flight": i, **f.summary(), "end_cause": end_cause(f)} for i, f in enumerate(flights)]
            rows += mine
            cells = []
            for group in ["train_area", "outside"]:
                g = [r for r in mine if r["test_set"] == group]
                success = np.mean([(not r["crashed"]) and r["late_error_m"] < TOLERANCE_M for r in g])
                cells.append(f"{group} success {success:.0%} crash {np.mean([r['crashed'] for r in g]):.0%}")
            print(f"horizon {horizon:>2}, {name:<11}: " + ", ".join(cells), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _write_rows(out_dir / "summary.csv", rows, append=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
