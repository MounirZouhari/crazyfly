"""Compare several seed batches (e.g. TD-MPC2 small, TD-MPC2 standard, PPO).

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/compare_methods.py \
        "TD-MPC2 small=outputs/batch1_fixed_start" \
        "TD-MPC2 standard=outputs/batch1_standard" \
        "PPO=outputs/ppo_fixed_start" --out outputs/comparison

Each argument is "label=batch folder" (folders made by run_seeds.py,
cluster/seed_job.sh or cluster/ppo_job.sh). Writes to --out:

    by_env_steps.png    test-flight error and return against environment steps
                        (log scale): how much flying experience each method needs
    by_train_time.png   the same against training wall-clock time
    test_starts.png     final agents from every test start
    comparison.md       one table with the final numbers

Values are mean +/- standard error across seeds, each seed first averaged over
its own flights, as in analyze_seeds.py. Wall-clock times are only comparable
between batches that ran on the same hardware (see each batch_info.json).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_seeds import TOLERANCE_M, load_batch, mean_and_se

COLORS = ["#1f6f8b", "#d1495b", "#edae49", "#66a182", "#7d5ba6"]


def curve_matrix(seeds, attr):
    """Rows = seeds, columns = the learning-curve points all seeds share."""
    shared = sorted(set.intersection(*(set(s.curve_steps) for s in seeds)))
    rows = []
    for s in seeds:
        lookup = dict(zip(s.curve_steps, getattr(s, attr)))
        rows.append([lookup[step] for step in shared])
    return np.array(shared), np.array(rows)


def curve_times(seeds, batch_dir):
    """Mean training minutes at each shared learning-curve point."""
    import csv
    shared = sorted(set.intersection(*(set(s.curve_steps) for s in seeds)))
    per_seed = []
    for s in seeds:
        times = {}
        with open(batch_dir / s.name / "learning_curve.csv", newline="") as f:
            for row in csv.DictReader(f):
                times[int(row["env_steps"])] = float(row["train_seconds"]) / 60
        per_seed.append([times[step] for step in shared])
    return np.mean(per_seed, axis=0)


def plot_curves(batches, x_kind, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for color, (label, batch_dir, seeds) in zip(COLORS, batches):
        steps, error = curve_matrix(seeds, "curve_error")
        _, ret = curve_matrix(seeds, "curve_return")
        x = np.maximum(steps, 1) if x_kind == "steps" else curve_times(seeds, batch_dir)
        for ax, values in zip(axes, (error, ret)):
            mean, se = mean_and_se(values)
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, lw=0)
            ax.plot(x, mean, color=color, lw=2, marker="o", ms=3, label=f"{label} ({len(seeds)} seeds)")
    axes[0].axhline(TOLERANCE_M, color="black", ls=":", lw=1)
    axes[0].set_ylabel("late-flight distance to target (m)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("return per test flight")
    for ax in axes:
        if x_kind == "steps":
            ax.set_xscale("log")
            ax.set_xlabel("environment steps (log scale)")
        else:
            ax.set_xlabel("training wall-clock time (min)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    title = "against environment steps" if x_kind == "steps" else "against training time"
    fig.suptitle(f"Test flights during training, {title} (mean ± standard error)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_starts(batches, starts, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(batches)
    for b, (color, (label, _, seeds)) in enumerate(zip(COLORS, batches)):
        for i, start in enumerate(starts):
            errors = np.array([s.mean_of(start, "late_error_m") for s in seeds])
            mean, se = mean_and_se(errors[:, None])
            x = i - 0.4 + width * (b + 0.5)
            ax.bar(x, mean[0], width * 0.9, yerr=se[0], color=color, capsize=3,
                   label=label if i == 0 else None)
    ax.axhline(TOLERANCE_M, color="black", ls=":", lw=1)
    ax.set_xticks(range(len(starts)), starts)
    ax.set_ylabel("late-flight distance to target (m)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Final agents from each test start (mean ± standard error across seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_table(batches, starts, out_path):
    def fmt(values, digits=3):
        mean, se = mean_and_se(np.asarray(values, dtype=float)[:, None])
        return f"{mean[0]:.{digits}f} ± {se[0]:.{digits}f}"

    lines = [
        "# Method comparison: one-drone hover, start (0, 0, 0.2), target (0, 0, 1)",
        "",
        "Mean ± standard error across seeds. Late error = mean distance to target over the last 2 s flown.",
        "",
        "| method | seeds | training steps | training time (min) | late error, training start (m) "
        "| late error, all starts (m) | crashes, all starts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, batch_dir, seeds in batches:
        home = [s.mean_of("home", "late_error_m") for s in seeds]
        every = [np.mean([s.mean_of(start, "late_error_m") for start in starts]) for s in seeds]
        crashed = sum(s.crashes()[0] for s in seeds)
        flights = sum(s.crashes()[1] for s in seeds)
        steps = max(int(s.curve_steps[-1]) for s in seeds)
        lines.append(f"| {label} | {len(seeds)} | {steps:,} | {fmt([s.train_minutes for s in seeds], 1)} "
                     f"| {fmt(home)} | {fmt(every)} | {crashed}/{flights} |")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", nargs="+", help='"label=batch folder"')
    parser.add_argument("--out", type=Path, default=Path("outputs/comparison"))
    args = parser.parse_args()

    batches = []
    for item in args.batches:
        label, folder = item.split("=", 1)
        batches.append((label, Path(folder), load_batch(Path(folder))))
    starts = list(dict.fromkeys(r["start"] for r in batches[0][2][0].final_flights))
    args.out.mkdir(parents=True, exist_ok=True)

    plot_curves(batches, "steps", args.out / "by_env_steps.png")
    plot_curves(batches, "time", args.out / "by_train_time.png")
    plot_starts(batches, starts, args.out / "test_starts.png")
    write_table(batches, starts, args.out / "comparison.md")
    print((args.out / "comparison.md").read_text())
    print(f"Plots saved to {args.out}")


if __name__ == "__main__":
    main()
