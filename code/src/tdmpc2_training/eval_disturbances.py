"""Test a trained TD-MPC2 hover agent under disturbances it never trained with.

From the repository root:

    .venv/bin/python code/src/tdmpc2_training/eval_disturbances.py <run_dir>
    .venv/bin/python code/src/tdmpc2_training/eval_disturbances.py <run_dir> --no-mpc

Every disturbance is flown at several strengths from the same test starts
(hover_eval.TEST_SETS, default the 50 train_area starts) with the same sensor
noise, so the only thing that changes between conditions is the disturbance.
The undisturbed flights are flown too, as the paired baseline. Nothing is
trained. Writes <run_dir>/disturbances[_policy_only]/summary.csv, one row per
flight, with the disturbance and its strength as the first columns.

Disturbances (none of them happens in training):
  wind    steady wind along +x, in m/s, acting through the simulator's air
          model (air density 1.225 kg/m3 in cf2.xml). Indoors: a fan or an
          air-conditioning draft is about 1-3 m/s.
  gust    the same wind, but only for 1 s, starting 3 s into the flight
  mass    extra weight (an expansion deck, a marker, a small camera): body
          mass and inertia times the strength. The real drone weighs 27 g.
  motor   weaker motors (battery voltage dropping, worn propellers): thrust
          and moments times the strength
  delay   every command reaches the drone 1 or 2 control steps late (20 or
          40 ms), like radio lag on a real drone; the first commands are a
          plain hover
The simulated drone needs 76% of its maximum thrust to hover, so extra mass
and weaker motors are kept small.
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
from crazyflie_hover import ACTION_HIGH, ACTION_LOW, HoverDrone, place_drone
from eval_hover import load_agent
from hover_eval import CONTROL_DT, TEST_SETS, Flight, FlightTester, _write_rows, resolve_starts

# Strengths per disturbance. The undisturbed baseline is flown separately.
DISTURBANCES = {
    "wind": [1.0, 2.0, 3.0, 4.0],   # m/s
    "gust": [2.0, 4.0, 6.0],        # m/s for 1 s
    "mass": [1.05, 1.10, 1.15],     # x mass and inertia
    "motor": [0.95, 0.90],          # x thrust and moments
    "delay": [1, 2],                # control steps late
}
GUST_START_S, GUST_LENGTH_S = 3.0, 1.0


class WindyDrone(HoverDrone):
    """HoverDrone flying in the wind given by state.info['wind'] (m/s, world axes)."""

    def step(self, state, action):
        # The lab's step builds its model from self._mjx_model and keeps its
        # wind, so swap in this flight's wind while the step is traced.
        base = self._mjx_model
        self._mjx_model = base.replace(opt=base.opt.replace(wind=state.info["wind"]))
        try:
            return super().step(state, action)
        finally:
            self._mjx_model = base


class DisturbedFlightTester(FlightTester):
    """FlightTester whose flights can have wind, extra mass, weaker motors and a command delay.

    Wind, mass and motor strength are inputs of the compiled flight, so one
    compilation serves all their strengths; each delay length compiles once.
    """

    def __init__(self, flight_steps=500, mpc=True, metric_target=False, delay_steps=0):
        super().__init__(flight_steps=flight_steps, mpc=mpc, metric_target=metric_target)
        self.drone = WindyDrone(metric_target=metric_target)
        self.delay_steps = int(delay_steps)
        self._fly = jax.jit(self._fly_disturbed)

    def fly(self, agent, starts, seed, wind=None, mass_factor=1.0, motor_factor=1.0):
        """Like FlightTester.fly; wind is (flight_steps, 3) m/s, one row per control step."""
        start_names, starts = resolve_starts(starts)
        targets = np.tile(np.asarray(TARGET_POS, dtype=np.float32), (len(starts), 1))
        if wind is None:
            wind = np.zeros((self.flight_steps, 3), dtype=np.float32)
        reset_key, plan_key = jax.random.split(jax.random.PRNGKey(seed))
        reset_keys = jax.random.split(reset_key, len(starts))
        trajectory = jax.device_get(self._fly(
            agent, reset_keys, starts, targets, plan_key,
            jnp.asarray(wind, dtype=jnp.float32), jnp.float32(mass_factor), jnp.float32(motor_factor)))

        flights = []
        for i, name in enumerate(start_names):
            alive = trajectory["alive"][:, i]
            steps_flown = int(alive.sum())
            flights.append(Flight(
                start_name=name, start=starts[i], target=np.asarray(targets[i]),
                position=trajectory["position"][:, i], command=trajectory["command"][:, i],
                reward=trajectory["reward"][:, i], steps_flown=steps_flown,
                crashed=bool(trajectory["done"][:steps_flown, i].any()),
            ))
        return flights

    def _fly_disturbed(self, agent, reset_keys, starts, targets, plan_key, wind, mass_factor, motor_factor):
        def start_flight(key, start, target):
            state = self.drone.reset(key, random_reset=False, target=target)
            state = place_drone(self.drone, state, start, target)
            info = dict(state.info)
            info["body_mass"] = info["body_mass"] * mass_factor
            info["body_inertia"] = info["body_inertia"] * mass_factor
            info["actuator_gainprm"] = info["actuator_gainprm"] * motor_factor
            info["wind"] = wind[0]
            return state.replace(info=info)

        states = jax.vmap(start_flight)(reset_keys, starts, targets)
        step_all = jax.vmap(self.drone.step)
        low, high = jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH)
        n, horizon, action_dim = len(starts), agent.horizon, agent.model.action_dim
        plan = (jnp.zeros((n, horizon, action_dim)),
                jnp.full((n, horizon, action_dim), agent.max_plan_std))
        # Commands sent but not yet received (delay); a plain hover to begin with.
        hover = jnp.array([BASE_HOVER_THRUST, 0.0, 0.0, 0.0])
        queue = jnp.tile(hover, (n, max(self.delay_steps, 1), 1))

        def control_step(carry, inputs):
            states, plan, alive, queue = carry
            key, wind_now = inputs
            if self.mpc:
                action, plan = agent.act(states.obs, prev_plan=plan,
                                         deterministic=True, train=False, key=key)
            else:
                action, _ = agent.act(states.obs, mpc=False,
                                      deterministic=True, train=False, key=key)
            command = jnp.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)
            if self.delay_steps:
                received = queue[:, 0]
                queue = jnp.concatenate([queue[:, 1:], command[:, None]], axis=1)
            else:
                received = command
            info = dict(states.info)
            info["wind"] = jnp.broadcast_to(wind_now, (n, 3))
            states = step_all(states.replace(info=info), received)
            done = states.done > 0
            record = {"position": states.data.qpos[:, :3], "command": received,
                      "reward": states.reward, "done": done, "alive": alive}
            return (states, plan, alive & ~done, queue), record

        keys = jax.random.split(plan_key, self.flight_steps)
        carry = (states, plan, jnp.ones(n, dtype=bool), queue)
        _, trajectory = jax.lax.scan(control_step, carry, (keys, wind))
        return trajectory


def wind_profile(kind, strength, steps):
    wind = np.zeros((steps, 3), dtype=np.float32)
    if kind == "wind":
        wind[:, 0] = strength
    elif kind == "gust":
        t = np.arange(steps) * CONTROL_DT
        wind[(t >= GUST_START_S) & (t < GUST_START_S + GUST_LENGTH_S), 0] = strength
    return wind


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="training output folder")
    parser.add_argument("--test-sets", default="train_area",
                        help=f"comma-separated from: {', '.join(TEST_SETS)}")
    parser.add_argument("--only", default="all", help=f"'all' or comma-separated from: {', '.join(DISTURBANCES)}")
    parser.add_argument("--no-mpc", action="store_true", help="fly the policy network alone, without planning")
    parser.add_argument("--snapshot", type=int, default=None, help="test the agent kept at this many steps")
    parser.add_argument("--test-seed", type=int, default=None,
                        help="noise seed (default: training seed + 3000, the same for every condition)")
    parser.add_argument("--flight-steps", type=int, default=500)
    parser.add_argument("--out-name", default=None, help="output folder (default: disturbances[_policy_only])")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    kinds = list(DISTURBANCES) if args.only == "all" else args.only.split(",")
    unknown = [k for k in kinds if k not in DISTURBANCES]
    if unknown:
        parser.error(f"unknown disturbance(s): {unknown}")
    variant = ("" if args.snapshot is None else f"_snapshot{args.snapshot}") + ("_policy_only" if args.no_mpc else "")
    out_dir = args.run_dir / (args.out_name or f"disturbances{variant}")
    if (out_dir / "summary.csv").exists() and not args.overwrite:
        parser.error(f"{out_dir} already has results; pass --overwrite to replace them")

    agent, step, cfg = load_agent(args.run_dir, snapshot=args.snapshot)
    metric_target = bool(cfg.env.get("metric_target", False))
    seed = cfg.seed + 3000 if args.test_seed is None else args.test_seed
    starts = [entry for name in args.test_sets.split(",") for entry in TEST_SETS[name]]
    decision = "policy_only" if args.no_mpc else f"planner_{agent.population_size}x{agent.mppi_iterations}"
    print(f"Loaded {args.run_dir} at {step} steps; {len(starts)} starts per condition; {decision}.")

    testers = {0: DisturbedFlightTester(args.flight_steps, not args.no_mpc, metric_target)}
    conditions = [("none", 0.0)] + [(kind, s) for kind in kinds for s in DISTURBANCES[kind]]
    rows = []
    for kind, strength in conditions:
        delay = int(strength) if kind == "delay" else 0
        if delay not in testers:
            testers[delay] = DisturbedFlightTester(args.flight_steps, not args.no_mpc, metric_target, delay)
        flights = testers[delay].fly(
            agent, starts, seed,
            wind=wind_profile(kind, strength, args.flight_steps),
            mass_factor=strength if kind == "mass" else 1.0,
            motor_factor=strength if kind == "motor" else 1.0)
        mine = [{"disturbance": kind, "strength": strength, **r} for r in
                [{"seed": cfg.seed, "checkpoint_step": step, "decision": decision, "flight": i, **f.summary()}
                 for i, f in enumerate(flights)]]
        rows += mine
        late = np.array([r["late_error_m"] for r in mine])
        print(f"{kind:<6} {strength:>5}: crashes {sum(r['crashed'] for r in mine):>3}/{len(mine)}, "
              f"late error mean {late.mean():.3f} m, median {np.median(late):.3f} m", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _write_rows(out_dir / "summary.csv", rows, append=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
