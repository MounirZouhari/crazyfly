"""Summarise world-model prediction errors (model_error.py) across the seeds of one or more batches.

From the repository root, after copying the model_error/ folders home:

    .venv/bin/python code/src/tdmpc2_training/analyze_model_error.py \
        outputs/batch2/random_start_1M outputs/batch1_standard_1M --out outputs/batch2/model_error_summary

For each batch, start group and decision rule: the 3-step latent prediction
error and the reward prediction error, averaged per seed first (mean +/-
standard error across seeds), crash counts, and the error in the last 0.5 s
before crashes relative to the same seed's ordinary steps. Also the paired
ratio planner / policy-only per seed (same starts, same seed).
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ["train_area", "held_out", "outside"]
RULES = ["planner", "policy_only"]


def mean_se(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def load(batch_dir):
    seeds = {}
    for run in sorted(batch_dir.glob("seed_*")):
        path = run / "model_error" / "summary.csv"
        if path.exists():
            seeds[run.name] = list(csv.DictReader(open(path)))
    return seeds


def per_seed(rows, rule, group, column):
    values = [float(r[column]) for r in rows if r["decision"] == rule and r["test_set"] == group]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/model_error_summary"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = ["# World-model prediction error along test flights", "",
             "Latent error: mean squared error of the 3-step latent prediction from the actions actually taken "
             "(TD-MPC2's consistency loss), averaged over each flight's steps, then per seed, then mean ± SE "
             "across seeds. Crashes: total over all seeds.", "",
             "| batch | seeds | decision | starts | crashes | latent error k=3 | reward error | "
             "planner / policy (paired, per seed) |", "|---|---:|---|---|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(1, len(args.batches), figsize=(4.6 * len(args.batches), 4), squeeze=False, sharey=True)
    for ax, batch in zip(axes[0], args.batches):
        seeds = load(batch)
        if not seeds:
            lines.append(f"| {batch.name} | 0 | (no model_error results) | | | | | |")
            continue
        for rule in RULES:
            for group in GROUPS:
                latent = [per_seed(rows, rule, group, "latent_error_k3") for rows in seeds.values()]
                reward = [per_seed(rows, rule, group, "reward_error") for rows in seeds.values()]
                crashes = sum(sum(r["crashed"] == "True" for r in rows if r["decision"] == rule and r["test_set"] == group)
                              for rows in seeds.values())
                flights = sum(sum(1 for r in rows if r["decision"] == rule and r["test_set"] == group)
                              for rows in seeds.values())
                ratio = ""
                if rule == "planner":
                    pol = [per_seed(rows, "policy_only", group, "latent_error_k3") for rows in seeds.values()]
                    ratios = [a / b for a, b in zip(latent, pol) if np.isfinite(a) and np.isfinite(b) and b > 0]
                    m, s = mean_se(ratios)
                    ratio = f"{m:.2f} ± {s:.2f}"
                m_l, s_l = mean_se(latent)
                m_r, s_r = mean_se(reward)
                lines.append(f"| {batch.name} | {len(seeds)} | {rule} | {group} | {crashes}/{flights} | "
                             f"{m_l:.5f} ± {s_l:.5f} | {m_r:.4f} ± {s_r:.4f} | {ratio} |")
            means = [mean_se([per_seed(rows, rule, g, "latent_error_k3") for rows in seeds.values()]) for g in GROUPS]
            ax.errorbar(range(len(GROUPS)), [m for m, _ in means], yerr=[s for _, s in means], marker="o",
                        capsize=3, ls="-" if rule == "planner" else "--", label=rule.replace("_", " "))
        # Error just before crashes vs the same seed's ordinary flights (planner).
        before, ordinary = [], []
        for rows in seeds.values():
            b = [float(r["latent_error_k3_before_crash"]) for r in rows
                 if r["decision"] == "planner" and r["crashed"] == "True"]
            o = [float(r["latent_error_k3"]) for r in rows if r["decision"] == "planner" and r["crashed"] == "False"]
            b = [v for v in b if np.isfinite(v)]
            if b and o:
                before.append(np.mean(b))
                ordinary.append(np.mean(o))
        if before:
            lines.append(f"| {batch.name} | {len(before)} | planner | last 0.5 s before crashes vs completed flights "
                         f"| | {np.mean(before):.5f} vs {np.mean(ordinary):.5f} | | "
                         f"{np.mean(np.array(before) / np.array(ordinary)):.2f}x |")
        ax.set_xticks(range(len(GROUPS)), ["training area", "held-out block", "outside"], fontsize=8)
        ax.set_title(f"{batch.name} ({len(seeds)} seeds)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("3-step latent prediction error\n(mean ± SE across seeds)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("How wrong the agent's own world model is along its test flights", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "model_error.png", dpi=150)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
