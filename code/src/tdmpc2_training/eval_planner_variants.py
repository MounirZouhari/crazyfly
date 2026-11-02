"""Does more conservative planning remove the extra crashes? A test of model exploitation. Nothing is trained.

Single-start agents crash much more with their planner than with their policy
network alone, and the planner's flights visit states where the world model
predicts badly (model_error.py). If the planner is exploiting model errors,
keeping its candidate actions closer to what the policy network proposes
should remove those crashes. This flies the same agent, from the same starts
with the same sensor noise, with:
  policy_only    the policy network alone
  planner        the planner as trained (512 plans x 6 rounds, sampling
                 spread max_plan_std = 2, 24 plans from the policy network)
  narrow_std     the same planner with max_plan_std = 0.5 (candidates stay
                 closer to the previous plan)
  more_prior     the same planner with half of its 512 candidates proposed
                 by the policy network (policy_prior_samples = 256)
The planner settings are changed only at test time.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_planner_variants.py <run_dir>

Writes <run_dir>/planner_variants/summary.csv (one row per flight, with the
variant) and prints success and crash shares per start group.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import numpy as np

from eval_hover import load_agent
from hover_eval import TEST_SETS, TOLERANCE_M, FlightTester, _write_rows


def variants(agent):
    prior = min(256, agent.population_size // 2)
    return [
        ("policy_only", agent, False),
        ("planner", agent, True),
        ("narrow_std", agent.replace(max_plan_std=0.5), True),
        ("more_prior", agent.replace(policy_prior_samples=prior), True),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.run_dir / "planner_variants"
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")
    agent, step, cfg = load_agent(args.run_dir)
    metric = bool(cfg.env.get("metric_target", False))
    starts = TEST_SETS["train_area"] + TEST_SETS["held_out"] + TEST_SETS["outside"]
    testers = {True: FlightTester(500, mpc=True, metric_target=metric),
               False: FlightTester(500, mpc=False, metric_target=metric)}

    rows = []
    for name, variant, mpc in variants(agent):
        flights = testers[mpc].fly(variant, starts, seed=cfg.seed + 6000)   # same noise for every variant
        mine = [{"seed": cfg.seed, "checkpoint_step": step, "variant": name, "flight": i, **f.summary()}
                for i, f in enumerate(flights)]
        rows += mine
        cells = []
        for group in ["train_area", "held_out", "outside"]:
            g = [r for r in mine if r["test_set"] == group]
            success = np.mean([(not r["crashed"]) and r["late_error_m"] < TOLERANCE_M for r in g])
            cells.append(f"{group} success {success:.0%} crash {np.mean([r['crashed'] for r in g]):.0%}")
        print(f"{name:<12} " + ", ".join(cells), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _write_rows(out_dir / "summary.csv", rows, append=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
