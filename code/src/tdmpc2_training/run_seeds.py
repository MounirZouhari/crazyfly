"""Train and test the TD-MPC2 hover agent once per seed, then analyze the batch.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/run_seeds.py
    .venv/bin/python code/src/tdmpc2_training/run_seeds.py --seeds 0-10 --parallel 2

Every seed uses the same settings (tdmpc2_jax/config_hover.yaml plus any
--override); only the seed changes. Each seed gets its own folder:

    outputs/<batch>/seed_00/   checkpoint, learning_curve.csv, evaluation/
    outputs/<batch>/logs/      terminal output of every run

Seeds that already finished are skipped, so after an interruption just run the
same command again.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PYTHON = sys.executable


def parse_seeds(text):
    """'0-10' -> [0, ..., 10];  '0,3,7' -> [0, 3, 7]."""
    seeds = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return seeds


def git_state():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return result.stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}


def run_logged(command, log_file, env):
    log_file.write(f"\n$ {' '.join(command)}\n")
    log_file.flush()
    return subprocess.run(command, cwd=REPO, stdout=log_file, stderr=subprocess.STDOUT, env=env).returncode


def run_seed(seed, args, batch_dir, env):
    run_dir = batch_dir / f"seed_{seed:02d}"
    trained = (run_dir / "checkpoint" / str(args.steps)).exists()
    tested = (run_dir / "evaluation" / "summary.csv").exists()

    if tested:
        print(f"seed {seed}: already done, skipping")
        return True
    if run_dir.exists() and not trained:
        # Restarting here would silently resume from an old checkpoint.
        print(f"seed {seed}: {run_dir} exists but training did not finish. "
              f"Rename or delete it to rerun this seed.")
        return False

    started = time.time()
    with open(batch_dir / "logs" / f"seed_{seed:02d}.txt", "a") as log:
        if not trained:
            print(f"seed {seed}: training ({args.steps} steps)")
            train = [PYTHON, str(HERE / "train_hover.py"), f"seed={seed}",
                     f"max_steps={args.steps}", f"hydra.run.dir={run_dir}", *args.override]
            if run_logged(train, log, env) != 0:
                print(f"seed {seed}: training FAILED, see logs/seed_{seed:02d}.txt")
                return False

        print(f"seed {seed}: testing")
        test = [PYTHON, str(HERE / "eval_hover.py"), str(run_dir)]
        if run_logged(test, log, env) != 0:
            print(f"seed {seed}: testing FAILED, see logs/seed_{seed:02d}.txt")
            return False

    print(f"seed {seed}: done in {(time.time() - started) / 60:.1f} min")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", default="batch1_fixed_start", help="folder name under outputs/")
    parser.add_argument("--seeds", default="0-10", help="e.g. 0-10 or 0,3,7")
    parser.add_argument("--steps", type=int, default=50_000, help="training steps per seed")
    parser.add_argument("--parallel", type=int, default=1, help="seeds trained at the same time")
    parser.add_argument("--override", nargs="*", default=[],
                        help="extra Hydra settings for every seed, e.g. tdmpc2.batch_size=256")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    batch_dir = REPO / "outputs" / args.batch
    os.makedirs(batch_dir / "logs", exist_ok=True)

    # Keep a record of exactly what this batch ran.
    info_path = batch_dir / "batch_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {"launches": []}
    info.update(seeds=seeds, steps=args.steps, overrides=args.override)
    info["launches"].append({"time": datetime.now().isoformat(timespec="seconds"),
                             "parallel": args.parallel, **git_state()})
    info_path.write_text(json.dumps(info, indent=2))

    env = dict(os.environ)
    if args.parallel > 1:
        # By default each JAX process reserves 75% of the GPU memory for itself.
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    print(f"Batch {args.batch}: seeds {seeds}, {args.steps} steps each, {args.parallel} at a time")
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(lambda s: run_seed(s, args, batch_dir, env), seeds))
    print(f"\nFinished {sum(results)}/{len(seeds)} seeds in {(time.time() - started) / 3600:.2f} h")

    if all(results):
        subprocess.run([PYTHON, str(HERE / "analyze_seeds.py"), str(batch_dir)], cwd=REPO)


if __name__ == "__main__":
    main()
