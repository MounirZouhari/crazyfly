"""How wrong is an agent's own world model where its flights fail? Nothing is trained.

Hypothesis H1 (model exploitation): where the world model has seen little
data (new starts, disturbances) its predictions are poor, and a planner that
optimises against it picks actions it wrongly rates highly. That would make
planning worse than the policy network alone off the training distribution,
which is what the Batch 1 1M agents showed.

This flies an agent, records what it observed and did at every step, then
asks its own world model to predict the next 1-3 steps from the actions
actually taken and compares with what happened:
  latent error   mean squared difference between the predicted latent state
                 and the encoding of the observation actually reached (the
                 consistency loss TD-MPC2 trains on), k = 1 and 3 steps ahead
  reward error   |predicted reward - actual reward| for the next step
Compared between start groups (training area, held-out block, outside),
decision rules (planner vs policy network), and the last 0.5 s before a
crash vs ordinary steps.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/model_error.py <run_dir> [--snapshot N]

Writes <run_dir>/model_error/summary.csv (one row per flight) and summary.md.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np

from constants import TARGET_POS
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, place_drone
from eval_hover import load_agent
from hover_eval import TEST_SETS, FlightTester, resolve_starts

HORIZON = 3            # TD-MPC2's planning and training horizon
CRASH_WINDOW = 25      # steps (0.5 s) before a crash


class RecordingFlightTester(FlightTester):
    """FlightTester that also records each observation and the agent's raw action."""

    def _fly_batch(self, agent, reset_keys, starts, targets, plan_key):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            return place_drone(self.drone, state, start, target)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)
        n, horizon, action_dim = len(starts), agent.horizon, agent.model.action_dim
        plan = (jnp.zeros((n, horizon, action_dim)), jnp.full((n, horizon, action_dim), agent.max_plan_std))

        def control_step(carry, key):
            states, plan, alive = carry
            obs = states.obs
            if self.mpc:
                action, plan = agent.act(obs, prev_plan=plan, deterministic=True, train=False, key=key)
            else:
                action, _ = agent.act(obs, mpc=False, deterministic=True, train=False, key=key)
            command = jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)
            states = step_all(states, command)
            done = states.done > 0
            record = {"obs": obs, "action": action, "reward": states.reward, "done": done, "alive": alive,
                      "position": states.data.qpos[:, :3], "command": command}
            return (states, plan, alive & ~done), record

        keys = jax.random.split(plan_key, self.flight_steps)
        _, trajectory = jax.lax.scan(control_step, (states, plan, jnp.ones(n, dtype=bool)), keys)
        return trajectory


def prediction_errors(agent, obs, actions, rewards, key):
    """obs (T, n, 31), actions (T, n, 4), rewards (T, n) -> latent error (T-k, n) for k=1..H, reward error (T, n)."""
    model = agent.model
    z_all = model.encode(obs, model.encoder.params, key)                 # (T, n, latent)
    reward_pred, _ = model.reward(z_all, actions, model.reward_model.params)
    reward_pred = reward_pred.reshape(rewards.shape) if reward_pred.size == rewards.size else reward_pred[..., 0]
    reward_err = jnp.abs(reward_pred - rewards)
    latent_err = []
    z = z_all
    for k in range(1, HORIZON + 1):
        z = model.next(z[:-1], actions[k - 1:k - 1 + z.shape[0] - 1], model.dynamics_model.params)
        target = z_all[k:]
        latent_err.append(jnp.mean((z - target) ** 2, axis=-1))           # (T-k, n)
    return latent_err, reward_err


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--snapshot", type=int, default=None)
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    agent, step, cfg = load_agent(args.run_dir, snapshot=args.snapshot)
    metric = bool(cfg.env.get("metric_target", False))
    out_dir = args.run_dir / (args.out_name or ("model_error" if args.snapshot is None
                                                else f"model_error_snapshot{args.snapshot}"))
    os.makedirs(out_dir, exist_ok=True)
    starts = TEST_SETS["train_area"] + TEST_SETS["held_out"] + TEST_SETS["outside"]
    labels, coords = resolve_starts(starts)
    errors_fn = jax.jit(prediction_errors)

    rows = []
    for rule, mpc in [("planner", True), ("policy_only", False)]:
        tester = RecordingFlightTester(flight_steps=500, mpc=mpc, metric_target=metric)
        reset_key, plan_key = jax.random.split(jax.random.PRNGKey(cfg.seed + 5000))
        traj = tester._fly(agent, jax.random.split(reset_key, len(coords)), coords,
                           np.tile(np.asarray(TARGET_POS, np.float32), (len(coords), 1)), plan_key)
        latent_err, reward_err = errors_fn(agent, traj["obs"], traj["action"], traj["reward"], jax.random.PRNGKey(0))
        latent_err = [np.asarray(e) for e in latent_err]
        reward_err = np.asarray(reward_err)
        alive = np.asarray(traj["alive"])
        done = np.asarray(traj["done"])
        for i, label in enumerate(labels):
            flown = int(alive[:, i].sum())
            crashed = bool(done[:flown, i].any())
            usable = max(0, flown - HORIZON)       # steps whose k-step prediction stays inside the flight
            row = {"seed": cfg.seed, "checkpoint_step": step, "decision": rule, "start": label,
                   "test_set": label.split("/")[0], "steps_flown": flown, "crashed": crashed}
            for k in (1, HORIZON):
                e = latent_err[k - 1][:usable, i]
                row[f"latent_error_k{k}"] = float(e.mean()) if usable else float("nan")
                if crashed:
                    w = e[max(0, usable - CRASH_WINDOW):usable]
                    row[f"latent_error_k{k}_before_crash"] = float(w.mean()) if len(w) else float("nan")
                else:
                    row[f"latent_error_k{k}_before_crash"] = float("nan")
            row["reward_error"] = float(reward_err[:flown, i].mean())
            rows.append(row)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# World-model prediction error, {args.run_dir} at {step} steps", "",
             f"Latent error k: mean squared error of the {HORIZON}-step latent prediction from the actions "
             "actually taken (TD-MPC2's consistency loss). Before crash: last 0.5 s before a crash.", "",
             "| decision | starts | flights | crashes | latent error k=1 | latent error k=3 | "
             "k=3 before crashes | reward error |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for rule in ["planner", "policy_only"]:
        for group in ["train_area", "held_out", "outside"]:
            mine = [r for r in rows if r["decision"] == rule and r["test_set"] == group]
            before = [r["latent_error_k3_before_crash"] for r in mine if r["crashed"]]
            lines.append(f"| {rule} | {group} | {len(mine)} | {sum(r['crashed'] for r in mine)} | "
                         f"{np.nanmean([r['latent_error_k1'] for r in mine]):.4f} | "
                         f"{np.nanmean([r['latent_error_k3'] for r in mine]):.4f} | "
                         f"{(np.nanmean(before) if before else float('nan')):.4f} | "
                         f"{np.nanmean([r['reward_error'] for r in mine]):.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
