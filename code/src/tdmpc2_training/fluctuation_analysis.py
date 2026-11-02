"""Do the ups and downs of a long TD-MPC2 run line up with what training logs?

After a run first hovers well, its test flights keep getting better and worse
(Batch 1 1M runs: 13-39% of later test points worse than 10 cm). This pairs
every test point (learning_curve.csv, every 5k steps) with the training
statistics logged over the 5k steps before it (TensorBoard: world-model,
reward, value and policy losses, policy spread, value scale) and reports
Spearman correlations:
  level    test late error vs the statistic, after the run first hovered
  change   change in late error vs change in the statistic between test
           points (removes slow trends, which otherwise make everything
           correlate)
Nothing is trained. From the repository root:

    .venv/bin/python code/src/tdmpc2_training/fluctuation_analysis.py outputs/batch1_standard_1M \
        --out outputs/batch2/fluctuation_analysis
"""
import argparse
import csv
import os
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX.proto import event_pb2

STATS = ["consistency_loss", "reward_loss", "value_loss", "policy_loss", "policy_log_std", "value_scale"]


def read_scalars(path):
    """All scalar summaries in a TensorBoard event file -> {tag: [(step, value)]}."""
    out = defaultdict(list)
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            (length,) = struct.unpack("<Q", header)
            f.read(4)                      # CRC of the length
            data = f.read(length)
            f.read(4)                      # CRC of the data
            event = event_pb2.Event()
            event.ParseFromString(data)
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    out[value.tag].append((event.step, value.simple_value))
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 5:
        return float("nan"), len(x)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def permutation_p(x, y, rho, n=2000, seed=0):
    """Two-sided p-value of a Spearman correlation by shuffling (ignores autocorrelation)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    null = [spearman(x, rng.permutation(y))[0] for _ in range(n)]
    return float(np.mean(np.abs(null) >= abs(rho)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/batch2/fluctuation_analysis"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = ["# Within-run fluctuations vs training statistics", "",
             f"Runs: {args.batch_dir}. Test point = mean late error of that point's test flights. "
             "Statistic = mean of the training log over the 5k steps before the test point. "
             "Only points after the run first averaged under 10 cm are used.", "",
             "| seed | statistic | level: rho (n) | change: rho (n) | change: permutation p |",
             "|---:|---|---:|---:|---:|"]
    fig, axes = plt.subplots(len(STATS) + 1, 1, figsize=(11, 2.1 * (len(STATS) + 1)), sharex=True)
    for run_dir in sorted(args.batch_dir.glob("seed_*")):
        curve = defaultdict(list)
        for r in csv.DictReader(open(run_dir / "learning_curve.csv")):
            curve[int(r["env_steps"])].append(float(r["late_error_m"]))
        steps = np.array(sorted(curve))
        late = np.array([np.mean(curve[s]) for s in steps])
        first_good = next((i for i, v in enumerate(late) if v < 0.10), None)
        if first_good is None:
            continue
        scalars = {}
        for event_file in (run_dir / "tensorboard").glob("events.out.tfevents.*"):
            for tag, values in read_scalars(event_file).items():
                scalars.setdefault(tag, []).extend(values)
        window = steps[1] - steps[0]
        aligned = {}
        for stat in STATS:
            values = sorted(scalars.get(f"train/{stat}_mean", []))
            if not values:
                continue
            at, val = np.array([v[0] for v in values]), np.array([v[1] for v in values])
            aligned[stat] = np.array([val[(at > s - window) & (at <= s)].mean() if np.any((at > s - window) & (at <= s))
                                      else np.nan for s in steps])
        sel = slice(first_good, None)
        for stat, series in aligned.items():
            rho_level, n_level = spearman(late[sel], series[sel])
            rho_change, n_change = spearman(np.diff(late[sel]), np.diff(series[sel]))
            p = permutation_p(np.diff(late[sel]), np.diff(series[sel]), rho_change)
            lines.append(f"| {run_dir.name} | {stat} | {rho_level:+.2f} ({n_level}) | "
                         f"{rho_change:+.2f} ({n_change}) | {p:.3f} |")
        axes[0].plot(steps / 1000, late, lw=1, label=run_dir.name)
        for ax, stat in zip(axes[1:], STATS):
            if stat in aligned:
                ax.plot(steps / 1000, aligned[stat], lw=1)
                ax.set_ylabel(stat.replace("_", " "), fontsize=8)
    axes[0].set_ylabel("test late error (m)", fontsize=8)
    axes[0].axhline(0.10, color="black", ls=":", lw=1)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[-1].set_xlabel("training steps (thousands)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Test error during training and the training statistics logged at the same time", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "fluctuations.png", dpi=150)
    lines += ["", "Permutation p-values ignore autocorrelation in the series and are not corrected for "
              "testing several statistics; treat them as screening only."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
