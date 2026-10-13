"""Plots and a written summary for a batch of seeds made by run_seeds.py.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/analyze_seeds.py outputs/batch1_fixed_start

Writes to <batch>/analysis/:

    learning_curves.png   test-flight error and return during training
    home_flights.png      the final agents flying from the training start
    test_starts.png       final agents from every test start
    summary.md            per-seed table, mean +/- standard error, findings

Every plot comes in two versions side by side: the mean across seeds with a
standard-error band, and every seed as a thin line with the median seed thick.
The median seed is the one with the median late-flight error from the
training start in the final test.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEAN_COLOR = "#1f6f8b"
SEED_COLOR = "#a7b1bc"
MEDIAN_COLOR = "#d1495b"
TOLERANCE_M = 0.10
TARGET_HEIGHT_M = 1.0


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_and_se(curves):
    """Mean and standard error across seeds (rows).

    A point where any seed is missing is left out rather than averaged over
    fewer seeds.
    """
    curves = np.asarray(curves, dtype=float)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return mean, std / np.sqrt(len(curves))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Seed:
    """Everything one seed produced."""

    def __init__(self, run_dir):
        self.name = run_dir.name
        self.number = int(run_dir.name.split("_")[-1])
        self.final_flights = read_csv(run_dir / "evaluation" / "summary.csv")
        self.trajectories = read_csv(run_dir / "evaluation" / "trajectories.csv")

        curve_rows = read_csv(run_dir / "learning_curve.csv")
        by_step = defaultdict(list)
        for row in curve_rows:
            by_step[int(row["env_steps"])].append(row)
        self.curve_steps = np.array(sorted(by_step))
        self.curve_error = np.array([np.mean([float(r["late_error_m"]) for r in by_step[s]])
                                     for s in self.curve_steps])
        self.curve_return = np.array([np.mean([float(r["return"]) for r in by_step[s]])
                                      for s in self.curve_steps])
        self.train_minutes = float(curve_rows[-1]["train_seconds"]) / 60

    def flights_from(self, start):
        return [r for r in self.final_flights if r["start"] == start]

    def mean_of(self, start, column):
        return float(np.mean([float(r[column]) for r in self.flights_from(start)]))

    def crashes(self, start=None):
        rows = self.final_flights if start is None else self.flights_from(start)
        return sum(r["crashed"] == "True" for r in rows), len(rows)

    def home_series(self, column, steps):
        """Per-step value from the training start, averaged over this seed's flights.

        The line stops when the first of those flights ends (a crash), so the
        average never silently switches to the flights that survived.
        """
        per_flight = defaultdict(lambda: np.full(steps, np.nan))
        for row in self.trajectories:
            if row["start"] == "home":
                per_flight[row["flight"]][int(row["step"])] = float(row[column])
        return np.mean(np.array(list(per_flight.values())), axis=0)


def load_batch(batch_dir):
    seeds, missing = [], []
    for run_dir in sorted(batch_dir.glob("seed_*")):
        if (run_dir / "evaluation" / "summary.csv").exists():
            seeds.append(Seed(run_dir))
        else:
            missing.append(run_dir.name)
    if missing:
        print(f"Not finished, left out: {', '.join(missing)}")
    if not seeds:
        raise SystemExit(f"No finished seeds in {batch_dir}")
    return seeds


def median_seed(seeds):
    errors = np.array([s.mean_of("home", "late_error_m") for s in seeds])
    order = np.argsort(errors)
    return seeds[order[(len(seeds) - 1) // 2]]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def draw_pair(ax_mean, ax_seeds, x, curves, median_index, ylabel):
    """Left: mean +/- standard error. Right: every seed thin, median seed thick."""
    mean, se = mean_and_se(curves)
    ax_mean.fill_between(x, mean - se, mean + se, color=MEAN_COLOR, alpha=0.25, lw=0,
                         label="± standard error")
    ax_mean.plot(x, mean, color=MEAN_COLOR, lw=2, label=f"mean of {len(curves)} seeds")

    for i, curve in enumerate(curves):
        if i != median_index:
            ax_seeds.plot(x, curve, color=SEED_COLOR, lw=0.9)
    ax_seeds.plot([], [], color=SEED_COLOR, lw=0.9, label="one seed")
    ax_seeds.plot(x, curves[median_index], color=MEDIAN_COLOR, lw=2.5, label="median seed")

    for ax in (ax_mean, ax_seeds):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)


def plot_learning_curves(seeds, median, out_path):
    steps = np.array(sorted(set().union(*(s.curve_steps for s in seeds))))

    def aligned(attr):
        rows = []
        for s in seeds:
            lookup = dict(zip(s.curve_steps, getattr(s, attr)))
            rows.append([lookup.get(step, np.nan) for step in steps])
        return np.array(rows)

    x = steps / 1000
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    m = seeds.index(median)
    draw_pair(axes[0, 0], axes[0, 1], x, aligned("curve_error"), m, "late-flight distance to target (m)")
    draw_pair(axes[1, 0], axes[1, 1], x, aligned("curve_return"), m, "return per test flight")
    for ax in axes[0]:
        ax.axhline(TOLERANCE_M, color="black", ls=":", lw=1)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("training steps (thousands)")
    axes[0, 0].set_title("Mean across seeds")
    axes[0, 1].set_title(f"Every seed (median: seed {median.number})")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Test flights during training (start (0, 0, 0.2), target (0, 0, 1))", fontsize=13)
    fig.text(0.5, 0.005, "Each point averages that seed's test flights. Late-flight distance = mean over "
             "the last 2 s of the flight. Dotted line: 10 cm.", ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_home_flights(seeds, median, out_path):
    # Longest flight in the batch (a seed whose flights all crashed has shorter ones).
    steps = max(int(r["step"]) for s in seeds for r in s.trajectories) + 1
    time_s = np.arange(1, steps + 1) * 0.02
    m = seeds.index(median)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    draw_pair(axes[0, 0], axes[0, 1], time_s, [s.home_series("error_m", steps) for s in seeds], m,
              "distance to target (m)")
    draw_pair(axes[1, 0], axes[1, 1], time_s, [s.home_series("z_m", steps) for s in seeds], m,
              "height (m)")
    for ax in axes[0]:
        ax.axhline(TOLERANCE_M, color="black", ls=":", lw=1)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.axhline(TARGET_HEIGHT_M, color="black", ls="--", lw=1)
        ax.set_xlabel("simulated flight time (s)")
    axes[0, 0].set_title("Mean across seeds")
    axes[0, 1].set_title(f"Every seed (median: seed {median.number})")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Final agents flying from the training start", fontsize=13)
    fig.text(0.5, 0.005, "Each seed line averages that seed's flights and stops when the first of them ends; "
             "the mean stops when any seed line stops. Dotted: 10 cm. Dashed: target height.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_test_starts(seeds, median, starts, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(0)
    starts_labels = []
    for i, start in enumerate(starts):
        errors = np.array([s.mean_of(start, "late_error_m") for s in seeds])
        mean, se = mean_and_se(errors[:, None])
        first = i == 0  # label each kind of mark once for the legend
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(errors)), errors, s=18, color=SEED_COLOR,
                   zorder=2, label="one seed" if first else None)
        ax.scatter(i, median.mean_of(start, "late_error_m"), s=60, color=MEDIAN_COLOR, zorder=3,
                   label=f"median seed ({median.number})" if first else None)
        ax.errorbar(i + 0.3, mean[0], yerr=se[0], fmt="o", color=MEAN_COLOR, capsize=4, zorder=4,
                    label="mean ± standard error" if first else None)
        crashed = sum(s.crashes(start)[0] for s in seeds)
        total = sum(s.crashes(start)[1] for s in seeds)
        starts_labels.append(f"{start}\ncrashes {crashed}/{total}")
    ax.axhline(TOLERANCE_M, color="black", ls=":", lw=1)
    ax.set_xticks(range(len(starts)), starts_labels)
    ax.set_ylabel("late-flight distance to target (m)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Final agents from each test start (target (0, 0, 1))")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Written summary
# ---------------------------------------------------------------------------

def fmt(values, digits=3):
    mean, se = mean_and_se(np.asarray(values, dtype=float)[:, None])
    return f"{mean[0]:.{digits}f} ± {se[0]:.{digits}f}"


def write_summary(seeds, median, starts, batch_dir, out_path):
    info_path = batch_dir / "batch_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    home_error = [s.mean_of("home", "late_error_m") for s in seeds]
    home_xy = [s.mean_of("home", "late_error_xy_m") for s in seeds]
    home_z = [s.mean_of("home", "late_error_z_m") for s in seeds]
    home_inside = [s.mean_of("home", "time_in_10cm") for s in seeds]
    minutes = [s.train_minutes for s in seeds]

    lines = [
        f"# {batch_dir.name}: TD-MPC2 hover, {len(seeds)} seeds",
        "",
        f"Seeds: {', '.join(str(s.number) for s in seeds)}. "
        f"Training steps per seed: {info.get('steps', 'unknown')}. "
        f"Extra settings: {', '.join(info.get('overrides', [])) or 'none (config_hover.yaml)'}.",
        f"Code version: {info.get('launches', [{}])[-1].get('commit', 'unknown')}"
        + (" (with uncommitted changes)" if info.get('launches', [{}])[-1].get('uncommitted_changes') else ""),
        "",
        "Values are mean ± standard error across seeds; each seed is first averaged over its own flights.",
        "Late error = mean distance to target over the last 2 s flown. "
        "Time in 10 cm = share of the flight after the first 2 s within 0.10 m of the target.",
        "",
        "## Each seed, from the training start",
        "",
        "| seed | late error (m) | horizontal (m) | vertical (m) | time in 10 cm | crashes (all starts) | training (min) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in seeds:
        crashed, total = s.crashes()
        mark = " (median)" if s is median else ""
        lines.append(
            f"| {s.number}{mark} | {s.mean_of('home', 'late_error_m'):.3f} | "
            f"{s.mean_of('home', 'late_error_xy_m'):.3f} | {s.mean_of('home', 'late_error_z_m'):.3f} | "
            f"{s.mean_of('home', 'time_in_10cm'):.0%} | {crashed}/{total} | {s.train_minutes:.1f} |"
        )
    lines += [
        f"| **all** | **{fmt(home_error)}** | {fmt(home_xy)} | {fmt(home_z)} | "
        f"{fmt(np.array(home_inside) * 100, 0)} % | {sum(s.crashes()[0] for s in seeds)}/"
        f"{sum(s.crashes()[1] for s in seeds)} | {fmt(minutes, 1)} |",
        "",
        "## Each test start",
        "",
        "| start | late error (m) | best seed (m) | worst seed (m) | crashes |",
        "|---|---:|---:|---:|---:|",
    ]
    for start in starts:
        errors = [s.mean_of(start, "late_error_m") for s in seeds]
        crashed = sum(s.crashes(start)[0] for s in seeds)
        total = sum(s.crashes(start)[1] for s in seeds)
        lines.append(f"| {start} | {fmt(errors)} | {min(errors):.3f} | {max(errors):.3f} | {crashed}/{total} |")

    within = sum(e < TOLERANCE_M for e in home_error)
    lines += [
        "",
        "## Findings (computed, not interpreted)",
        "",
        f"- From the training start, the late-flight error was {fmt(home_error)} m "
        f"(median seed {median.number}: {median.mean_of('home', 'late_error_m'):.3f} m; "
        f"range {min(home_error):.3f} to {max(home_error):.3f} m).",
        f"- {within} of {len(seeds)} seeds ended their home flights within {TOLERANCE_M:.2f} m on average.",
        f"- Of that error, the horizontal part was {fmt(home_xy)} m and the vertical part {fmt(home_z)} m.",
        f"- Crashes in the final test: {sum(s.crashes()[0] for s in seeds)} of "
        f"{sum(s.crashes()[1] for s in seeds)} flights across all starts.",
        f"- Training time per seed (test flights excluded): {fmt(minutes, 1)} min.",
        "",
        "Plots: learning_curves.png, home_flights.png, test_starts.png",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()

    seeds = load_batch(args.batch_dir)
    median = median_seed(seeds)
    starts = list(dict.fromkeys(r["start"] for r in seeds[0].final_flights))
    out_dir = args.batch_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    plot_learning_curves(seeds, median, out_dir / "learning_curves.png")
    plot_home_flights(seeds, median, out_dir / "home_flights.png")
    plot_test_starts(seeds, median, starts, out_dir / "test_starts.png")
    write_summary(seeds, median, starts, args.batch_dir, out_dir / "summary.md")
    print((out_dir / "summary.md").read_text())
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
