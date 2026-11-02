"""Summarise test-time planner variants (eval_planner_variants.py) across the seeds of one or more batches.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/analyze_planner_variants.py \
        outputs/batch1_standard_1M outputs/batch1_small_narval --out outputs/batch2/planner_variants_summary

For each batch, variant and start group: success and crash shares, averaged
per seed first, mean +/- standard error across seeds, and the paired change
relative to the planner as trained (same agent, starts and sensor noise).
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

GROUPS = ["train_area", "held_out", "outside"]
VARIANTS = ["policy_only", "planner", "narrow_std", "more_prior"]
TOLERANCE_M = 0.10


def mean_se(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), (float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)


def shares(rows, variant, group):
    mine = [r for r in rows if r["variant"] == variant and r["test_set"] == group]
    if not mine:
        return float("nan"), float("nan")
    success = np.mean([r["crashed"] == "False" and float(r["late_error_m"]) < TOLERANCE_M for r in mine])
    crash = np.mean([r["crashed"] == "True" for r in mine])
    return float(success), float(crash)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/planner_variants_summary"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = ["# Test-time planner variants (same agents, starts and sensor noise)", "",
             "Success = full 10 s flight with late error below 10 cm. Cells: success / crash share, mean ± SE "
             "across seeds; in brackets the paired change vs the planner as trained.", "",
             "| batch | seeds | variant | " + " | ".join(GROUPS) + " |", "|---|---:|---|" + "---|" * len(GROUPS)]
    for batch in args.batches:
        seeds = {}
        for run in sorted(batch.glob("seed_*")):
            path = run / "planner_variants" / "summary.csv"
            if path.exists():
                seeds[run.name] = list(csv.DictReader(open(path)))
        if not seeds:
            lines.append(f"| {batch.name} | 0 | (no results) |" + " |" * len(GROUPS))
            continue
        for variant in VARIANTS:
            cells = []
            for group in GROUPS:
                per_seed = [shares(rows, variant, group) for rows in seeds.values()]
                base = [shares(rows, "planner", group) for rows in seeds.values()]
                s, s_se = mean_se([p[0] for p in per_seed])
                c, c_se = mean_se([p[1] for p in per_seed])
                cell = f"{s:.0%} ± {s_se:.0%} / {c:.0%} ± {c_se:.0%}"
                if variant != "planner":
                    ds, _ = mean_se([p[0] - b[0] for p, b in zip(per_seed, base)])
                    dc, _ = mean_se([p[1] - b[1] for p, b in zip(per_seed, base)])
                    cell += f" ({ds:+.0%} / {dc:+.0%})"
                cells.append(cell)
            lines.append(f"| {batch.name} | {len(seeds)} | {variant} | " + " | ".join(cells) + " |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
