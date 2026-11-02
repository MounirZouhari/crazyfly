"""How the drone flies, not only where it ends: time to stabilise, smoothness of the commands, saturation and the worst flights.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/analyze_flight_quality.py outputs/batch2/random_start_1M \
        outputs/batch2/random_start_small_1M --out outputs/batch2/flight_quality

Reads the step-by-step trajectories.csv of the final tests (with planning,
policy network only, 64x2 planner) of the training-area and held-out starts.
For every seed (then mean +/- standard error across seeds):
  reach        median time to first come within 20 cm of the target, over
               the full flights (s)
  settle       median time after which the drone stays within 10 cm of the
               target until the end, over the flights that get there (s)
  settled      share of flights that settle at all
  jitter       mean absolute change of a command from one step to the next
               during the last 5 s of full flights, in % of its range
               (thrust: 0-0.35 N; body moments: -1 to 1)
  saturated    share of steps with the thrust at 0 or 0.35 N, or a body
               moment at -1 or 1 (within 1% of the range)
  p90 / worst  90th percentile and largest late error of the full flights (cm)
Writes summary.md.
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = ("train_area", "held_out")
TOLERANCE_M = 0.10
CONTROL_DT_S = 0.02
THRUST_MAX_N = 0.35
STEADY_FROM_STEP = 250            # the last 5 s of a 10 s flight
RULES = {"planner": "with planning", "policy_only": "policy network only", "plan64x2": "64x2 planner"}


def mean_se(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), (float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)


def group_of(start):
    return start.split("/")[0] if "/" in start else "batch1"


def seed_quality(folder):
    """The statistics of one seed's test folder (trajectories.csv and summary.csv)."""
    late = {}
    for row in csv.DictReader(open(folder / "summary.csv")):
        if group_of(row["start"]) in GROUPS and row["crashed"] == "False":
            late[int(row["flight"])] = float(row["late_error_m"])
    steps = defaultdict(list)
    for row in csv.DictReader(open(folder / "trajectories.csv")):
        if group_of(row["start"]) in GROUPS:
            steps[int(row["flight"])].append((float(row["error_m"]), float(row["thrust"]), float(row["roll"]),
                                              float(row["pitch"]), float(row["yaw"])))
    settle, settled, reach, thrust_jitter, moment_jitter, saturated = [], [], [], [], [], []
    for flight, rows in steps.items():
        data = np.array(rows)
        error, thrust, moments = data[:, 0], data[:, 1] / THRUST_MAX_N, data[:, 2:5]
        outside = np.flatnonzero(error >= TOLERANCE_M)
        if flight in late:                       # full flight
            near = np.flatnonzero(error < 2 * TOLERANCE_M)
            if len(near):
                reach.append((near[0] + 1) * CONTROL_DT_S)
            if len(outside) == 0:
                settle.append(0.0)
            elif outside[-1] < len(error) - 1:
                settle.append((outside[-1] + 1) * CONTROL_DT_S)
            settled.append(len(outside) == 0 or outside[-1] < len(error) - 1)
            steady = slice(STEADY_FROM_STEP, None)
            thrust_jitter.append(100 * np.mean(np.abs(np.diff(thrust[steady]))))
            moment_jitter.append(100 * np.mean(np.abs(np.diff(moments[steady], axis=0))) / 2)
        saturated.append(np.mean((thrust <= 0.01) | (thrust >= 0.99) | np.any(np.abs(moments) >= 0.98, axis=1)))
    errors = np.array(list(late.values())) * 100
    return {"reach": float(np.median(reach)) if reach else float("nan"),
            "settle": float(np.median(settle)) if settle else float("nan"),
            "settled": float(np.mean(settled)) if settled else float("nan"),
            "thrust_jitter": float(np.mean(thrust_jitter)) if thrust_jitter else float("nan"),
            "moment_jitter": float(np.mean(moment_jitter)) if moment_jitter else float("nan"),
            "saturated": 100 * float(np.mean(saturated)),
            "p90": float(np.percentile(errors, 90)) if len(errors) else float("nan"),
            "worst": float(errors.max()) if len(errors) else float("nan")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/flight_quality"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = ["# Flight quality (training-area and held-out starts, final agents)", "",
             "Mean ± SE across seeds of each seed's value. reach: median time to first come within 20 cm (s); "
             "settle: median time to stay within 10 cm until the end (s); settled: share of full flights that do; "
             "jitter: mean step-to-step change of the command in the last 5 s, % of its range; saturated: % of "
             "steps with a command at its limit; p90 and worst: late error of full flights (cm).", "",
             "| batch | decision rule | seeds | reach 20 cm (s) | settle (s) | settled | thrust jitter (%) | "
             "moment jitter (%) | saturated (%) | p90 late error (cm) | worst late error (cm) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for batch in args.batches:
        prefix = "evaluation_batch2_tests" if batch.name.startswith("batch1") else "evaluation"
        for rule in RULES:
            folder = prefix if rule == "planner" else f"{prefix}_{rule}"
            per_seed = [seed_quality(run / folder) for run in sorted(batch.glob("seed_*"))
                        if (run / folder / "trajectories.csv").exists()]
            if not per_seed:
                continue
            cell = lambda key, fmt: "{} ± {}".format(*(format(v, fmt) for v in mean_se([s[key] for s in per_seed])))
            lines.append(f"| {batch.name} | {RULES[rule]} | {len(per_seed)} | {cell('reach', '.2f')} | {cell('settle', '.2f')} | "
                         f"{cell('settled', '.0%')} | {cell('thrust_jitter', '.2f')} | {cell('moment_jitter', '.2f')} | "
                         f"{cell('saturated', '.1f')} | {cell('p90', '.1f')} | {cell('worst', '.1f')} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
