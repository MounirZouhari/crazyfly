"""Does the planner need its jumpy commands? Smoother ways of acting, and slow motors. Nothing is trained.

At every step, TD-MPC2's planner executes the first action of the single
best candidate plan it sampled. That action jumps from step to step (in
steady hover the thrust changes by about 30% of its range per step, and
about half of the steps have a command at its limit), which a real drone's
motors cannot follow. This flies the same trained agent, from the same
starts with the same sensor noise, with:
  planner          the planner as trained (best candidate's first action)
  planner_mean     the first action of the planner's averaged plan (the
                   weighted mean of its best candidates, which it computes
                   anyway to start the next step)
  planner_filtered the planner's commands through a first-order low-pass
                   filter with a 40 ms time constant
  policy_only      the policy network alone
each with instant motors (as in training) and with motors that follow the
command with a first-order lag of 30 or 60 ms. These lags are assumptions
chosen to probe sensitivity; the real Crazyflie's motor response must be
measured before any value is trusted.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_smoothing.py <run_dir>

Writes <run_dir>/smoothing_test/summary.csv (one row per flight, with the
variant, the motor lag and the smoothness of the commands sent) and prints
success, crash and thrust jitter per variant.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu_training"))

import jax
import jax.numpy as jnp
import numpy as np

from constants import BASE_HOVER_THRUST, TARGET_POS
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, place_drone
from eval_disturbances import DisturbedFlightTester, end_cause
from eval_hover import load_agent
from hover_eval import CONTROL_DT, TEST_SETS, TOLERANCE_M, Flight, _write_rows, resolve_starts

FILTER_TIME_S = 0.04
MOTOR_LAGS_S = [0.0, 0.03, 0.06]
# variant: (mpc, weight of the averaged plan's action, command filter time constant)
VARIANTS = {"planner": (True, 0.0, 0.0), "planner_mean": (True, 1.0, 0.0),
            "planner_filtered": (True, 0.0, FILTER_TIME_S), "policy_only": (False, 0.0, 0.0)}
STEADY_FROM_STEP = 250            # the last 5 s of a 10 s flight


def smoothing(t_s):
    """Weight of the previous value in a first-order filter with time constant t_s at the control rate."""
    return float(np.exp(-CONTROL_DT / t_s)) if t_s > 0 else 0.0


def command_statistics(flight):
    """Jitter (mean step-to-step change in the last 5 s, % of range) and saturation (% of steps at a limit)."""
    low, high = np.asarray(ACTION_LOW), np.asarray(ACTION_HIGH)
    scaled = (flight.command[:flight.steps_flown] - low) / (high - low)          # each command in [0, 1]
    steady = scaled[STEADY_FROM_STEP:]
    jitter = 100 * np.abs(np.diff(steady, axis=0)).mean(axis=0) if len(steady) > 1 else np.full(4, np.nan)
    saturated = 100 * np.mean(np.any((scaled <= 0.01) | (scaled >= 0.99), axis=1))
    return {"thrust_jitter_pct": float(jitter[0]), "moment_jitter_pct": float(np.mean(jitter[1:])),
            "saturated_pct": float(saturated)}


class SmoothedFlightTester(DisturbedFlightTester):
    """DisturbedFlightTester with a choice of planner action, a command filter and a motor lag.

    The three are inputs of the compiled flight, so one compilation serves
    every combination.
    """

    def __init__(self, flight_steps=500, mpc=True, metric_target=False):
        super().__init__(flight_steps=flight_steps, mpc=mpc, metric_target=metric_target)
        self._fly = jax.jit(self._fly_smoothed)

    def fly(self, agent, starts, seed, mean_weight=0.0, filter_time_s=0.0, motor_lag_s=0.0):
        start_names, starts = resolve_starts(starts)
        targets = np.tile(np.asarray(TARGET_POS, dtype=np.float32), (len(starts), 1))
        reset_key, plan_key = jax.random.split(jax.random.PRNGKey(seed))
        reset_keys = jax.random.split(reset_key, len(starts))
        trajectory = jax.device_get(self._fly(
            agent, reset_keys, starts, targets, plan_key, jnp.float32(mean_weight),
            jnp.float32(smoothing(filter_time_s)), jnp.float32(smoothing(motor_lag_s))))
        flights = []
        for i, name in enumerate(start_names):
            steps_flown = int(trajectory["alive"][:, i].sum())
            flights.append(Flight(
                start_name=name, start=starts[i], target=np.asarray(targets[i]),
                position=trajectory["position"][:, i], command=trajectory["command"][:, i],
                reward=trajectory["reward"][:, i], steps_flown=steps_flown,
                crashed=bool(trajectory["done"][:steps_flown, i].any()),
            ))
        return flights

    def _fly_smoothed(self, agent, reset_keys, starts, targets, plan_key, mean_weight, filter_a, lag_b):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            state = place_drone(self.drone, state, start, target)
            info = dict(state.info)
            info["wind"] = jnp.zeros(3)                                   # no wind
            return state.replace(info=info)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)
        n, horizon, action_dim = len(starts), agent.horizon, agent.model.action_dim
        plan = (jnp.zeros((n, horizon, action_dim)),
                jnp.full((n, horizon, action_dim), agent.max_plan_std))
        hover = jnp.tile(jnp.array([BASE_HOVER_THRUST, 0.0, 0.0, 0.0]), (n, 1))

        def control_step(carry, key):
            states, plan, alive, sent, applied = carry
            if self.mpc:
                best, plan = agent.act(states.obs, prev_plan=plan, deterministic=True, train=False, key=key)
                action = (1.0 - mean_weight) * best + mean_weight * plan[0][:, 0, :]
            else:
                action, _ = agent.act(states.obs, mpc=False, deterministic=True, train=False, key=key)
            command = jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)
            sent = filter_a * sent + (1.0 - filter_a) * command          # what the controller sends
            applied = lag_b * applied + (1.0 - lag_b) * sent              # what the motors produce
            info = dict(states.info)
            info["wind"] = jnp.zeros((n, 3))
            states = step_all(states.replace(info=info), applied)
            done = states.done > 0
            record = {"position": states.data.qpos[:, :3], "command": sent,
                      "reward": states.reward, "done": done, "alive": alive}
            return (states, plan, alive & ~done, sent, applied), record

        keys = jax.random.split(plan_key, self.flight_steps)
        carry = (states, plan, jnp.ones(n, dtype=bool), hover, hover)
        _, trajectory = jax.lax.scan(control_step, carry, keys)
        return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.run_dir / "smoothing_test"
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")
    agent, step, cfg = load_agent(args.run_dir)
    metric = bool(cfg.env.get("metric_target", False))
    starts = TEST_SETS["train_area"] + TEST_SETS["held_out"]
    testers = {mpc: SmoothedFlightTester(500, mpc=mpc, metric_target=metric) for mpc in (True, False)}

    rows = []
    for lag in MOTOR_LAGS_S:
        for name, (mpc, mean_weight, filter_time) in VARIANTS.items():
            flights = testers[mpc].fly(agent, starts, seed=cfg.seed + 8000, mean_weight=mean_weight,
                                       filter_time_s=filter_time, motor_lag_s=lag)
            mine = [{"seed": cfg.seed, "checkpoint_step": step, "variant": name, "motor_lag_s": lag,
                     "flight": i, **f.summary(), "end_cause": end_cause(f), **command_statistics(f)}
                    for i, f in enumerate(flights)]
            rows += mine
            success = np.mean([(not r["crashed"]) and r["late_error_m"] < TOLERANCE_M for r in mine])
            crash = np.mean([r["crashed"] for r in mine])
            jitter = np.nanmean([r["thrust_jitter_pct"] for r in mine if not r["crashed"]] or [np.nan])
            saturated = np.mean([r["saturated_pct"] for r in mine])
            print(f"motor lag {1000 * lag:>3.0f} ms, {name:<16}: success {success:.0%} crash {crash:.0%}, "
                  f"thrust jitter {jitter:.1f}%, saturated steps {saturated:.0f}%", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _write_rows(out_dir / "summary.csv", rows, append=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
