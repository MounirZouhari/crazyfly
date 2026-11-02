"""Plots and a written summary for Batch 2 (random starts), in the style of analyze_seeds.py.

From the repository root, after copying the batches home from Narval:

    .venv/bin/python code/src/tdmpc2_training/analyze_batch2.py outputs/batch2/random_start_1M \
        --compare outputs/batch2/fixed_start_1M outputs/batch1_standard_1M

Writes to <first batch>/analysis/:

    learning_curves_by_group.png   success and crashes during training, per test group
    final_by_group.png             final agents per test group, with and without planning
    comparison.png                 every batch side by side (with --compare)
    disturbances.png               success under each disturbance and strength
                                   (runs tested with eval_disturbances.py)
    summary.md                     tables, mean +/- standard error across seeds

Definitions, fixed before any Batch 2 result was seen:
  success   the flight lasts the full 10 s and its late error (mean distance to
            the target over the last 2 s) is below 10 cm
  crash     the flight ended early (hit the ground or flew out of bounds)
Each seed is first averaged over its own flights; mean +/- standard error is
then taken across seeds. The median seed is the one with the median success
on the train_area group. Batch 1 runs are read from their
evaluation_batch2_tests* folders (the same test starts as Batch 2).
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
GROUPS = ["train_area", "held_out", "outside", "batch1"]
GROUP_TITLES = {
    "train_area": "training area (seen kind of start)",
    "held_out": "held-out block (never trained on)",
    "outside": "outside the training area",
    "batch1": "the six Batch 1 starts",
    "home": "training start of Batch 1 (0, 0, 0.2)",
}
RULES = {"planner": "with planning (as trained)", "policy_only": "policy network only",
         "plan64x2": "small planner 64x2"}
# Disturbance tests: baseline value on the x axis, strengths, axis label.
DISTURBANCE_AXES = {
    "wind": (0.0, [1.0, 2.0, 3.0, 4.0], "steady wind (m/s)"),
    "gust": (0.0, [2.0, 4.0, 6.0], "1 s gust from 3 s (m/s)"),
    "mass": (1.0, [1.05, 1.10, 1.15], "mass and inertia (x nominal)"),
    "motor": (1.0, [0.95, 0.90], "motor strength (x nominal)"),
    "delay": (0.0, [1.0, 2.0], "command delay (control steps of 20 ms)"),
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_and_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def fmt(values, digits=2, percent=False):
    mean, se = mean_and_se(values)
    scale = 100 if percent else 1
    unit = "%" if percent else ""
    return f"{mean * scale:.{digits}f} ± {se * scale:.{digits}f}{unit}"


def success(row):
    return row["crashed"] == "False" and float(row["late_error_m"]) < TOLERANCE_M


def group_of(row):
    if row.get("test_set"):
        return row["test_set"]
    return row["start"].split("/")[0] if "/" in row["start"] else "batch1"


class Seed:
    """One training run: its learning curve and its final tests."""

    def __init__(self, run_dir, batch1=False):
        self.dir = run_dir
        self.number = int(run_dir.name.split("_")[-1])
        prefix = "evaluation_batch2_tests" if batch1 else "evaluation"
        folders = {"planner": prefix, "policy_only": f"{prefix}_policy_only", "plan64x2": f"{prefix}_plan64x2"}
        self.final = {rule: read_csv(run_dir / folder / "summary.csv")
                      for rule, folder in folders.items() if (run_dir / folder / "summary.csv").exists()}
        # Disturbance tests (eval_disturbances.py), with planning and policy only.
        self.disturb = {rule: read_csv(run_dir / folder / "summary.csv")
                        for rule, folder in [("planner", "disturbances"), ("policy_only", "disturbances_policy_only")]
                        if (run_dir / folder / "summary.csv").exists()}
        self.curve = defaultdict(lambda: defaultdict(list))   # step -> group -> rows
        curve_path = run_dir / "learning_curve.csv"
        if curve_path.exists() and not batch1:
            for row in read_csv(curve_path):
                group = "home" if row["start"] == "home" else group_of(row)
                self.curve[int(row["env_steps"])][group].append(row)

    def final_stat(self, rule, group, stat):
        rows = [r for r in self.final.get(rule, []) if group_of(r) == group]
        if not rows:
            return np.nan
        if stat == "success":
            return float(np.mean([success(r) for r in rows]))
        if stat == "crash":
            return float(np.mean([r["crashed"] == "True" for r in rows]))
        if stat == "late_error_survived":   # median over flights that lasted the full 10 s
            late = [float(r["late_error_m"]) for r in rows if r["crashed"] == "False"]
            return float(np.median(late)) if late else np.nan
        raise ValueError(stat)

    def disturb_stat(self, rule, kind, strength, stat):
        """Success or crash share of one disturbance condition ('none' is the undisturbed baseline)."""
        rows = [r for r in self.disturb.get(rule, [])
                if r["disturbance"] == kind and np.isclose(float(r["strength"]), strength)]
        if not rows:
            return np.nan
        if stat == "success":
            return float(np.mean([success(r) for r in rows]))
        return float(np.mean([r["crashed"] == "True" for r in rows]))

    def curve_stat(self, step, group, stat):
        rows = self.curve.get(step, {}).get(group, [])
        if not rows:
            return np.nan
        if stat == "success":
            return float(np.mean([success(r) for r in rows]))
        if stat == "crash":
            return float(np.mean([r["crashed"] == "True" for r in rows]))
        raise ValueError(stat)


def load_batch(batch_dir, require_final=True):
    batch1 = batch_dir.name.startswith("batch1")   # Batch 1 folders keep their original names
    seeds, missing = [], []
    for run_dir in sorted(batch_dir.glob("seed_*")):
        seed = Seed(run_dir, batch1=batch1)
        if "planner" in seed.final or (not require_final and seed.curve):
            seeds.append(seed)
        else:
            missing.append(run_dir.name)
    if missing:
        print(f"{batch_dir.name}: not finished, left out: {', '.join(missing)}")
    if not seeds:
        raise SystemExit(f"No finished seeds in {batch_dir}")
    return seeds, batch1


def median_seed(seeds):
    values = np.array([s.final_stat("planner", "train_area", "success") for s in seeds])
    order = np.argsort(values, kind="stable")
    return seeds[order[(len(seeds) - 1) // 2]]


def draw_pair(ax_mean, ax_seeds, x, curves, median_index, ylabel):
    """Left: mean +/- standard error. Right: every seed thin, median seed thick."""
    curves = np.asarray(curves, dtype=float)
    mean = np.nanmean(curves, axis=0)
    se = (np.nanstd(curves, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(curves), axis=0))
          if len(curves) > 1 else np.zeros_like(mean))
    ax_mean.fill_between(x, mean - se, mean + se, color=MEAN_COLOR, alpha=0.25, lw=0, label="± standard error")
    ax_mean.plot(x, mean, color=MEAN_COLOR, lw=2, label=f"mean of {len(curves)} seeds")
    for i, curve in enumerate(curves):
        if i != median_index:
            ax_seeds.plot(x, curve, color=SEED_COLOR, lw=0.9)
    ax_seeds.plot([], [], color=SEED_COLOR, lw=0.9, label="one seed")
    ax_seeds.plot(x, curves[median_index], color=MEDIAN_COLOR, lw=2.5, label="median seed")
    for ax in (ax_mean, ax_seeds):
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)


def plot_learning_curves(seeds, median, title, out_path):
    steps = np.array(sorted(set().union(*(s.curve.keys() for s in seeds))))
    groups = ["home", "train_area", "held_out", "outside"]
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 3.2 * len(groups)), sharex=True)
    m = seeds.index(median)
    for row, group in enumerate(groups):
        curves = [[s.curve_stat(step, group, "success") for step in steps] for s in seeds]
        draw_pair(axes[row, 0], axes[row, 1], steps / 1000, curves, m, "success share")
        axes[row, 0].set_title(f"{GROUP_TITLES[group]}: mean across seeds", fontsize=10)
        axes[row, 1].set_title(f"{GROUP_TITLES[group]}: every seed (median: seed {median.number})", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("training steps (thousands)")
    fig.suptitle(f"{title}: test flights during training", fontsize=13)
    fig.text(0.5, 0.003, "Success = full 10 s flight and late error below 10 cm. Per seed and test point: "
             "1 flight from the training start, 8 / 6 / 6 from the other groups.", ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final(seeds, median, title, out_path):
    rules = [r for r in RULES if any(r in s.final for s in seeds)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    rng = np.random.default_rng(0)
    width = 0.8 / len(rules)
    colors = [MEAN_COLOR, "#e09f3e", "#6a994e"]
    for ax, stat, label in [(axes[0], "success", "success share"), (axes[1], "crash", "crash share")]:
        for j, rule in enumerate(rules):
            for i, group in enumerate(GROUPS):
                values = np.array([s.final_stat(rule, group, stat) for s in seeds])
                x = i + (j - (len(rules) - 1) / 2) * width
                ax.scatter(x + rng.uniform(-width / 4, width / 4, len(values)), values, s=12, color=SEED_COLOR, zorder=2)
                mean, se = mean_and_se(values[~np.isnan(values)])
                ax.errorbar(x, mean, yerr=se, fmt="o", color=colors[j], capsize=4, zorder=3,
                            label=RULES[rule] if i == 0 else None)
        ax.set_xticks(range(len(GROUPS)), [GROUP_TITLES[g].replace(" (", "\n(") for g in GROUPS], fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title}: final agents, {len(seeds)} seeds (grey dots: one seed each)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curve_comparison(batches, out_path):
    """Learning curves of several Batch 2 batches side by side: crash and success share per group."""
    batches = [(name, seeds) for name, seeds in batches if any(s.curve for s in seeds)]
    groups = ["train_area", "held_out", "outside"]
    fig, axes = plt.subplots(2, len(groups), figsize=(5 * len(groups), 7), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.arange(len(batches)))
    for j, (name, seeds) in enumerate(batches):
        steps = np.array(sorted(set().union(*(s.curve.keys() for s in seeds))))
        for i, group in enumerate(groups):
            for row, stat in enumerate(["crash", "success"]):
                curves = np.array([[s.curve_stat(step, group, stat) for step in steps] for s in seeds])
                mean = np.nanmean(curves, axis=0)
                se = (np.nanstd(curves, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(curves), axis=0))
                      if len(curves) > 1 else np.zeros_like(mean))
                axes[row, i].fill_between(steps / 1000, mean - se, mean + se, color=colors[j], alpha=0.2, lw=0)
                axes[row, i].plot(steps / 1000, mean, color=colors[j], lw=2,
                                  label=f"{name} ({len(seeds)} seeds)" if (row, i) == (0, 0) else None)
    for i, group in enumerate(groups):
        axes[0, i].set_title(GROUP_TITLES[group], fontsize=10)
        axes[1, i].set_xlabel("training steps (thousands)")
    axes[0, 0].set_ylabel("crash share")
    axes[1, 0].set_ylabel("success share")
    for ax in axes.flat:
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Test flights during training (mean ± SE across seeds; 8 / 6 / 6 flights per seed and point)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(batches, out_path):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    width = 0.8 / len(batches)
    colors = plt.cm.tab10(np.arange(len(batches)))
    for j, (name, seeds) in enumerate(batches):
        for i, group in enumerate(GROUPS):
            values = np.array([s.final_stat("planner", group, "success") for s in seeds])
            values = values[~np.isnan(values)]
            mean, se = mean_and_se(values)
            ax.bar(i + (j - (len(batches) - 1) / 2) * width, mean, width * 0.9, yerr=se, capsize=3,
                   color=colors[j], label=f"{name} ({len(seeds)} seeds)" if i == 0 else None)
    ax.set_xticks(range(len(GROUPS)), [GROUP_TITLES[g].replace(" (", "\n(") for g in GROUPS], fontsize=8)
    ax.set_ylabel("success share (mean ± SE across seeds)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Final agents with planning, same test starts for every batch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def disturbance_curve(seeds, rule, kind):
    """x values and per-seed success, with the undisturbed baseline first."""
    base_x, strengths, _ = DISTURBANCE_AXES[kind]
    xs = [base_x] + strengths
    per_seed = np.array([[s.disturb_stat(rule, "none", 0.0, "success")]
                         + [s.disturb_stat(rule, kind, x, "success") for x in strengths] for s in seeds])
    return np.array(xs), per_seed


def plot_disturbances(batches, out_path):
    batches = [(name, seeds) for name, seeds in batches if all(s.disturb for s in seeds)]
    fig, axes = plt.subplots(1, len(DISTURBANCE_AXES), figsize=(3.6 * len(DISTURBANCE_AXES), 4.4), sharey=True)
    colors = plt.cm.tab10(np.arange(len(batches)))
    for ax, kind in zip(axes, DISTURBANCE_AXES):
        for j, (name, seeds) in enumerate(batches):
            for rule, style in [("planner", "-"), ("policy_only", "--")]:
                if not all(rule in s.disturb for s in seeds):
                    continue
                xs, per_seed = disturbance_curve(seeds, rule, kind)
                mean = np.nanmean(per_seed, axis=0)
                se = (np.nanstd(per_seed, axis=0, ddof=1) / np.sqrt(len(per_seed))
                      if len(per_seed) > 1 else np.zeros_like(mean))
                ax.errorbar(xs, mean, yerr=se, color=colors[j], ls=style, marker="o", ms=4, capsize=3,
                            label=f"{name}: {RULES[rule]}")
        ax.set_xlabel(DISTURBANCE_AXES[kind][2], fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success share, 50 training-area starts\n(mean ± SE across seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=8)
    fig.suptitle("Disturbances never seen in training (solid: with planning; dashed: policy network only)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def disturbance_lines(batches):
    batches = [(name, seeds) for name, seeds in batches if all(s.disturb for s in seeds)]
    if not batches:
        return []
    header = ["none"] + [f"{kind} {x:g}" for kind, (_, strengths, _) in DISTURBANCE_AXES.items() for x in strengths]
    lines = ["", "## Disturbances (success share on the 50 training-area starts, mean across seeds)", "",
             "| batch | decision rule | " + " | ".join(header) + " |",
             "|---|---|" + "---:|" * len(header)]
    for name, seeds in batches:
        for rule in ["planner", "policy_only"]:
            if not all(rule in s.disturb for s in seeds):
                continue
            cells = [np.nanmean([s.disturb_stat(rule, "none", 0.0, "success") for s in seeds])]
            for kind, (_, strengths, _) in DISTURBANCE_AXES.items():
                cells += [np.nanmean([s.disturb_stat(rule, kind, x, "success") for s in seeds]) for x in strengths]
            lines.append(f"| {name} | {RULES[rule]} | " + " | ".join(f"{c:.0%}" for c in cells) + " |")
    # How crashed flights ended (column added on Nov 2; older results lack it).
    cause_rows = []
    for name, seeds in batches:
        for rule in ["planner", "policy_only"]:
            rows = [r for s in seeds for r in s.disturb.get(rule, []) if r.get("end_cause")]
            for label, kinds in [("no disturbance", {"none"}), ("wind and gusts", {"wind", "gust"}),
                                 ("mass, motors, delay", {"mass", "motor", "delay"})]:
                crashed = [r for r in rows if r["disturbance"] in kinds and r["crashed"] == "True"]
                if crashed:
                    out = np.mean([r["end_cause"] == "out_of_bounds" for r in crashed])
                    cause_rows.append(f"| {name} | {RULES[rule]} | {label} | {len(crashed)} | {out:.0%} | {1 - out:.0%} |")
    if cause_rows:
        lines += ["", "How the crashed flights ended:", "",
                  "| batch | decision rule | conditions | crashed flights | out of bounds | hit the ground |",
                  "|---|---|---|---:|---:|---:|"] + cause_rows
    return lines


def bootstrap_ci(a, b=None, paired=False, n=10000, seed=0):
    """Mean difference a - b (or mean of a) with a 95% percentile bootstrap interval over seeds.

    paired=True: a and b come from the same seeds (resample seeds together).
    Otherwise seeds are resampled within each batch separately.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    if b is None:
        a = a[np.isfinite(a)]
        boots = [rng.choice(a, len(a)).mean() for _ in range(n)]
        return a.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    b = np.asarray(b, dtype=float)
    if paired:
        d = a - b
        d = d[np.isfinite(d)]
        boots = [rng.choice(d, len(d)).mean() for _ in range(n)]
        return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def comparison_lines(batches):
    """95% bootstrap intervals: first batch vs each other batch, and planner vs other rules within each batch."""
    lines = ["", "## Statistical comparisons (95% bootstrap intervals over seeds)", "",
             "Differences in success share (percentage points). Intervals from fewer than about 5 seeds are "
             "unreliable.", "",
             "| comparison | " + " | ".join(GROUPS) + " |", "|---|" + "---:|" * len(GROUPS)]
    fmt = lambda m, lo, hi: f"{m * 100:+.0f} [{lo * 100:+.0f}, {hi * 100:+.0f}]"
    main_name, main_seeds = batches[0]
    for name, seeds in batches[1:]:
        cells = [fmt(*bootstrap_ci([s.final_stat("planner", g, "success") for s in main_seeds],
                                   [s.final_stat("planner", g, "success") for s in seeds])) for g in GROUPS]
        lines.append(f"| {main_name} - {name} (with planning) | " + " | ".join(cells) + " |")
    for name, seeds in batches:
        for other in ["policy_only", "plan64x2"]:
            if not all(other in s.final for s in seeds):
                continue
            cells = [fmt(*bootstrap_ci([s.final_stat("planner", g, "success") for s in seeds],
                                       [s.final_stat(other, g, "success") for s in seeds], paired=True))
                     for g in GROUPS]
            lines.append(f"| {name}: planner - {RULES[other]} (paired) | " + " | ".join(cells) + " |")
    return lines


def batch_table(name, seeds, rule):
    lines = [f"| {name} | {RULES[rule]} | " + " | ".join(
        f"{fmt([s.final_stat(rule, g, 'success') for s in seeds], 0, True)} / "
        f"{fmt([s.final_stat(rule, g, 'crash') for s in seeds], 0, True)}" for g in GROUPS) + " |"]
    return lines


def write_summary(main_name, main_dir, seeds, median, batches, out_path):
    info_path = main_dir / "batch_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    launch = info.get("launches", [{}])[-1]
    lines = [
        f"# {main_name}: TD-MPC2 hover, Batch 2, {len(seeds)} seeds",
        "",
        f"Seeds: {', '.join(str(s.number) for s in seeds)}. Steps: {info.get('steps', 'unknown')}. "
        f"Settings: {', '.join(info.get('overrides', [])) or 'unknown'}. "
        f"Code: {launch.get('commit', 'unknown')}"
        + (" (with uncommitted changes)" if launch.get("uncommitted_changes") else "") + ".",
        "",
        "Success = full 10 s flight with late error below 10 cm. Values: mean ± standard error across seeds "
        "(each seed first averaged over its flights). Cells: success / crashes.",
        "",
        "## Final agents",
        "",
        "| batch | decision rule | " + " | ".join(GROUP_TITLES[g] for g in GROUPS) + " |",
        "|---|---|" + "---:|" * len(GROUPS),
    ]
    for name, batch_seeds in batches:
        for rule in RULES:
            if all(rule in s.final for s in batch_seeds):
                lines += batch_table(name, batch_seeds, rule)
    lines += [
        "",
        f"## {main_name}, each seed (with planning)",
        "",
        "| seed | " + " | ".join(f"{g} success" for g in GROUPS) + " | train_area median late error, full flights (m) |",
        "|---:|" + "---:|" * (len(GROUPS) + 1),
    ]
    for s in seeds:
        mark = " (median)" if s is median else ""
        lines.append(f"| {s.number}{mark} | " + " | ".join(
            f"{s.final_stat('planner', g, 'success'):.0%}" for g in GROUPS)
            + f" | {s.final_stat('planner', 'train_area', 'late_error_survived'):.3f} |")
    lines += comparison_lines(batches)
    lines += disturbance_lines(batches)
    lines += ["", "Plots: learning_curves_by_group.png, final_by_group.png"
              + (", comparison.png" if len(batches) > 1 else "")
              + (", disturbances.png" if disturbance_lines(batches) else "")]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch_dir", type=Path, help="Batch 2 folder with seed_XX runs")
    parser.add_argument("--compare", type=Path, nargs="*", default=[], help="other batches to compare with")
    parser.add_argument("--curves-only", action="store_true",
                        help="only compare learning curves (works while runs are still training)")
    args = parser.parse_args()

    if args.curves_only:
        batches = [(d.name, load_batch(d, require_final=False)[0]) for d in [args.batch_dir] + args.compare]
        out_dir = args.batch_dir / "analysis"
        out_dir.mkdir(exist_ok=True)
        plot_curve_comparison(batches, out_dir / "learning_curves_comparison.png")
        print(f"Saved {out_dir / 'learning_curves_comparison.png'}")
        return

    seeds, _ = load_batch(args.batch_dir)
    median = median_seed(seeds)
    out_dir = args.batch_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    name = args.batch_dir.name
    batches = [(name, seeds)] + [(d.name, load_batch(d)[0]) for d in args.compare]

    if any(s.curve for s in seeds):
        plot_learning_curves(seeds, median, name, out_dir / "learning_curves_by_group.png")
    plot_final(seeds, median, name, out_dir / "final_by_group.png")
    if len(batches) > 1:
        plot_comparison(batches, out_dir / "comparison.png")
        if sum(any(s.curve for s in batch_seeds) for _, batch_seeds in batches) > 1:
            plot_curve_comparison(batches, out_dir / "learning_curves_comparison.png")
    if any(all(s.disturb for s in batch_seeds) for _, batch_seeds in batches):
        plot_disturbances(batches, out_dir / "disturbances.png")
    write_summary(name, args.batch_dir, seeds, median, batches, out_dir / "summary.md")
    print((out_dir / "summary.md").read_text())
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
