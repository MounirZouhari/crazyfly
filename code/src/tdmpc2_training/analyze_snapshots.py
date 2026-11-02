"""Best vs final checkpoint, and how much a run really changes between snapshots (eval_snapshots.py output).

From the repository root, after copying the snapshot_evaluation/ folders home:

    .venv/bin/python code/src/tdmpc2_training/analyze_snapshots.py \
        outputs/batch2/random_start_1M outputs/batch2/random_start_small_1M --out outputs/batch2/snapshot_summary

For every seed: success on the 40 validation starts and on each test group at
every snapshot (100k, 200k, ... 1M). The best snapshot is chosen on
validation success only (ties go to the later snapshot); its test success is
compared with the final snapshot's. Within-run change: the standard deviation
of a seed's train_area success across snapshots from 500k on, compared with
the binomial noise expected from 50 flights at that success rate.
Writes summary.md and snapshots.png.
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ["validation", "train_area", "held_out", "outside"]
TOLERANCE_M = 0.10


def success_table(path):
    """{step: {group: success share}} from a snapshot_evaluation summary.csv."""
    hits = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        ok = r["crashed"] == "False" and float(r["late_error_m"]) < TOLERANCE_M
        hits[int(r["checkpoint_step"])][r["test_set"]].append(ok)
    return {step: {g: float(np.mean(v)) for g, v in groups.items()} for step, groups in hits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--folder", default="snapshot_evaluation", help="or snapshot_evaluation_policy_only")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/snapshot_summary"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = [f"# Snapshots: best (chosen on validation) vs final ({args.folder})", "",
             "Success = full 10 s flight with late error below 10 cm.", ""]
    fig, axes = plt.subplots(len(args.batches), 4, figsize=(16, 3.3 * len(args.batches)), squeeze=False, sharey=True)
    for row_axes, batch in zip(axes, args.batches):
        seeds = {}
        for run in sorted(batch.glob("seed_*")):
            path = run / args.folder / "summary.csv"
            if path.exists():
                seeds[run.name] = success_table(path)
        if not seeds:
            lines.append(f"{batch.name}: no snapshot results yet")
            continue
        lines += [f"## {batch.name} ({len(seeds)} seeds)", "",
                  "| seed | best step (validation) | test success at best: train_area / held_out / outside | "
                  "at final | within-run SD of train_area success (>= 500k) | binomial SD expected |",
                  "|---|---:|---|---|---:|---:|"]
        gains = defaultdict(list)
        for name, table in seeds.items():
            steps = sorted(table)
            best = max(steps, key=lambda s: (table[s].get("validation", 0.0), s))
            final = steps[-1]
            late = [table[s]["train_area"] for s in steps if s >= 500_000]
            p = float(np.mean(late)) if late else float("nan")
            fmt = lambda s: " / ".join(f"{table[s].get(g, float('nan')):.0%}" for g in GROUPS[1:])
            lines.append(f"| {name} | {best // 1000}k | {fmt(best)} | {fmt(final)} | "
                         f"{np.std(late, ddof=1) if len(late) > 1 else float('nan'):.3f} | "
                         f"{np.sqrt(p * (1 - p) / 50) if np.isfinite(p) else float('nan'):.3f} |")
            for g in GROUPS[1:]:
                gains[g].append(table[best].get(g, np.nan) - table[final].get(g, np.nan))
        lines.append("")
        lines.append("Best minus final test success, mean ± SE across seeds: " + ", ".join(
            f"{g} {np.nanmean(v):+.0%} ± {np.nanstd(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.0%}"
            for g, v in gains.items()))
        lines.append("")
        for ax, group in zip(row_axes, GROUPS):
            for name, table in seeds.items():
                steps = sorted(table)
                ax.plot(np.array(steps) / 1000, [table[s].get(group, np.nan) for s in steps], lw=1, alpha=0.7)
            all_steps = sorted(set().union(*(t.keys() for t in seeds.values())))
            mean = [np.nanmean([t.get(s, {}).get(group, np.nan) for t in seeds.values()]) for s in all_steps]
            ax.plot(np.array(all_steps) / 1000, mean, color="black", lw=2.5, label="mean of seeds")
            ax.set_title(f"{batch.name}: {group}", fontsize=9)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        row_axes[0].set_ylabel("success share")
    for ax in axes[-1]:
        ax.set_xlabel("training steps (thousands)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Every kept snapshot flown from the same starts with the same sensor noise "
                 "(thin lines: one seed each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / f"{args.folder}.png", dpi=150)
    (args.out / f"{args.folder}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
