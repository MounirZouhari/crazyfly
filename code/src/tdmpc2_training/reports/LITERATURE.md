# Literature used in this project

Each entry was checked on Nov 1, 2026 against the arXiv or publisher listing.
"Establishes" is what the source itself reports. "For us" is a hypothesis to
test, not a fact about the Crazyflie task.

## World models and planning

**Hansen, N., Su, H., Wang, X. (2024). TD-MPC2: Scalable, Robust World
Models for Continuous Control. ICLR 2024.** arXiv:2310.16828,
https://arxiv.org/abs/2310.16828
- **Establishes:** decoder-free latent world model with local MPPI planning;
  one fixed set of hyperparameters across 104 continuous-control tasks in four
  domains; agents up to 317M parameters.
- **For us:** we use its published hyperparameters unchanged, so a failure
  here is not a tuning failure.
- **Does not transfer automatically:** those benchmarks are not
  quadrotor-like. Our planning horizon is 3 steps at 50 Hz, i.e. 60 ms of
  look-ahead, short for a drone. Our implementation is the JAX port by
  S. Flandermeyer, with documented differences (CONTEXT.md section 9).

**Hansen, N., Wang, X., Su, H. (2022). Temporal Difference Learning for
Model Predictive Control. ICML 2022, PMLR 162:8387-8406.** arXiv:2203.04955,
https://arxiv.org/abs/2203.04955
- **Establishes:** TD-MPC combines a short-horizon latent model for
  trajectory optimisation with a learned terminal value function, both
  trained by TD learning.
- **For us:** with a 3-step horizon, the value function, not the model
  rollout, carries most of the long-term judgement. So planning-vs-policy
  differences may mostly reflect how MPPI uses the value function off the
  states seen in training.

**Janner, M., Fu, J., Zhang, M., Levine, S. (2019). When to Trust Your
Model: Model-Based Policy Optimization. NeurIPS 2019.** arXiv:1906.08253,
https://arxiv.org/abs/1906.08253
- **Establishes:** the benefit of model-generated data must be weighed
  against its bias. Short model rollouts branched from real data work best.
- **For us, hypothesis H1 (model exploitation):** where the world model has
  little data (unseen starts, disturbances), its errors are largest. A
  planner that optimises against the model then picks actions the model
  wrongly rates highly. The Batch 1 agents crashing more with planning than
  without (R4) fit this.
- **Test:** R1 vs C1 (more diverse data should shrink the gap); measure model
  prediction error along failing flights (next step).

**Lambert, N., Amos, B., Yadan, O., Calandra, R. (2020). Objective Mismatch
in Model-based Reinforcement Learning. L4DC 2020, PMLR 120:761-770.**
arXiv:2002.04523, https://arxiv.org/abs/2002.04523
- **Establishes:** one-step prediction accuracy of a dynamics model does not
  necessarily correlate with control performance.
- **For us:** low TD-MPC2 training losses need not mean good flights. Check
  whether learning-curve dips (within-run fluctuation) line up with the
  logged losses at all.

## Evaluating RL reliably

**Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.
(2018). Deep Reinforcement Learning That Matters. AAAI 2018.**
doi:10.1609/aaai.v32i1.11694, arXiv:1709.06560
- **Establishes:** reported deep-RL results vary strongly with random seeds
  and implementation details; it gives reporting guidelines.
- **For us:** the 6.5 cm to 1.05 m spread of the Batch 1 small runs is
  expected behaviour for short runs. A1 measures how much of it is
  evaluation noise.

**Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., Bellemare, M. G.
(2021). Deep Reinforcement Learning at the Edge of the Statistical
Precipice. NeurIPS 2021.** arXiv:2108.13264, https://arxiv.org/abs/2108.13264
- **Establishes:** with few runs, point estimates mislead. It recommends
  interval estimates (bootstrap), performance profiles and the interquartile
  mean (rliable library).
- **For us:** report mean ± SE across seeds, every seed, and paired
  differences; avoid claims from 3 seeds.

## Sim-to-real for quadrotors

**Molchanov, A., et al. (incl. Hönig, W., Preiss, J. A., Ayanian, N.)
(2019). Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies
to Multiple Quadrotors. IROS 2019.** arXiv:1903.04628,
https://arxiv.org/abs/1903.04628
- **Establishes:** low-level policies (state to motor outputs) trained only
  in simulation transfer to several real quadrotors without a pre-tuned
  inner controller. They withstand disturbances and harsh (thrown) starts.
  The paper studies which randomised parameters matter.
- **For us:** robustness comes from training on varied starts and
  randomised dynamics. That supports Batch 2's random starts and suggests
  testing dynamics randomisation next.
- **Does not transfer automatically:** their action space (motor outputs)
  differs from ours (thrust + body moments); see DEPLOYMENT_READINESS.md.

**Eschmann, J., Albani, D., Loianno, G. (2024). Learning to Fly in Seconds.
IEEE RA-L.** arXiv:2311.13081, https://arxiv.org/abs/2311.13081
- **Establishes:** sim-to-real transfer for direct RPM control after 18 s of
  training on a laptop, with the policy running on microcontrollers under
  real-time guarantees.
- **For us:** a small policy network on board is feasible. This is relevant
  because TD-MPC2's policy network alone decides in 0.2-0.4 ms (P), whereas
  its planner needs a GPU.

**Peng, X. B., Andrychowicz, M., Zaremba, W., Abbeel, P. (2018).
Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA
2018.** doi:10.1109/ICRA.2018.8460528, arXiv:1710.06537
- **Establishes:** randomising simulator dynamics in training gives policies
  that work on the real robot without real-world training, robust to
  calibration error (robot-arm pushing task).
- **For us:** the lab environment already supports mass, inertia and motor
  randomisation (crazyflie_env.reset with random_reset). Batch 2 keeps it
  off so that start generalisation and disturbance robustness stay separate.
  Turning it on is a candidate follow-up.
