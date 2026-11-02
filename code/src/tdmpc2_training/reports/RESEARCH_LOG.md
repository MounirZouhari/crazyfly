# Research log: TD-MPC2 Crazyflie hover

Chronological record of hypotheses, experiments, results, decisions and
failures. Times are EST. The experiment registry is `experiment_registry.json`;
the full experiment descriptions are in `../EXPERIMENTS.md`.

## 2026-11-01 (Sunday)

**Afternoon: audit after Prof. Bellinger's Oct 12 feedback.**
- Verified in code:
  - The PPO learning-rate schedule never anneals: it is counted in environment steps but stepped per optimizer update, so PPO ran at a constant 3e-3.
  - The PPO auto-reset carries the previous position and previous action into the next episode.
  - The observation divides the target offset by the start distance.
  - Only the final TD-MPC2 checkpoint is kept.
  - The real-drone code skips PPO observation normalisation and caps body rates at 1 deg/s.
- Decision: TD-MPC2 is the focus. PPO stays a historical baseline, its bugs are documented, and it is not rerun.

**Afternoon: measurements that need no training.**
- Decision latency (home, RTX 3060 under WSL2): published planner 10.7 ms median, policy network only under 1 ms; on the CPU 589 ms and 0.26 ms.
- Planner vs policy only on the 11 small 50k agents: planning gave 0.32 m late error vs 0.48 m (better in 7 of 11 seeds, not significant), but 11 vs 3 crashes over 198 flights. A bigger 512x6 planner added nothing (0.31 m).
- Training-step profile (home): one simulated drone about 10 ms per step, GPU busy 25-28% of the time.
- Learning curves of the 1M runs: after first reaching 10 cm, 13-39% of later test points were worse again. Typical late error 6-8 cm; best single points 1.2-1.8 cm.

**Evening: Batch 2 built and validated (commits 8edab5b, 8612881).**
- New code:
  - random starts in an area with a held-out block, and the target offset in metres (both off by default);
  - fixed test groups;
  - snapshots every 100k steps;
  - tests with and without planning;
  - cluster jobs.
- Validation:
  - 23 automatic checks pass.
  - Testing old agents with the old and new code gives identical results, step by step.
  - Training on the CPU gives identical results with old and new code. The GPU is not bit-repeatable: two runs of the old code also differ once learning starts.
  - End-to-end runs at home and on Narval pass.
- Backup of Narval scratch to /project verified.

**20:41-21:44: launch.**
- Test job 7659264 passed (15 min).
- R1 job 7659438 and C1 job 7659439 were submitted with a 10 h limit. They waited in the queue about 50 min: low fairshare, and 10 h jobs are hard to backfill. Plan B was prepared and validated: 3 h pieces that resume from checkpoints, with a fix so a resumed run draws fresh starts and noise (commit 52873af). It was not needed: R1 started 21:38-21:44 and C1 at 21:44.
- R4 job 7659440 finished: Batch 1 agents on the Batch 2 test starts.
  - The 1M single-start agents crash far more often **with** planning than with their policy network only: 66% vs 15% on training-area starts, 91% vs 54% in the held-out block, 3 seeds.
  - Hypothesis H1 (model exploitation): out of the states seen in training, the world model is wrong and MPPI selects actions it wrongly predicts as good, while the policy network is smoother. R1 tests whether training on diverse starts removes this.
- P job 7659441, Narval A100 2/7 slice, native Linux:
  - Published planner 6.1 ms median, 11.7 ms worst; policy only 0.37 ms; 516 ms on 2 CPU cores.
  - One training step: simulate 6.8 ms, plan 10.2 ms, update 2.6 ms (overlapped), 49 steps/s.
  - nvidia-smi reports no utilisation on MIG slices.
- Disturbance tests: Batch 1 1M agents (job 7660609); R1 and C1 queued to start after each seed (jobs 7660720, 7660721, `aftercorr`).
  - `check_disturbances.py` validated the physics: wind drag grows about with speed squared; +15% mass sinks 0.62 m in 1 s (0.64 m predicted).

**21:50: overnight program started** (priorities in `../EXPERIMENTS.md` and the handoff).
- A1 started: re-fly 33 Batch 1 agents with 5 noise seeds x 4 flights each, to split the seed-to-seed spread into evaluation noise vs real agent differences (Q1).

**22:00-22:40.**
- **D pilot launched:** small TD-MPC2 with random starts, 3 seeds x 1M (`config_batch2_random_start_small`), in two pieces of at most 3 h (jobs 7661493, then 7661496 with `aftercorr`), disturbance tests chained (job 7661497). The pieces started immediately: jobs of 3 h or less are scheduled at once on Narval, while 10 h jobs waited about 50 min.
- **Disturbance tests of the Batch 1 1M agents done** (job 7660609, 30 min per seed). Success from 50 training-area starts, planning vs policy only:
  - none: 21% vs 46%
  - wind 1 m/s: 15% vs 15%; wind 2 m/s and above: 0% vs 0%
  - gust 2 m/s: 17% vs 41%; gust 4 m/s and above: 0% vs 0%
  - mass x1.15: 9% vs 36%
  - motors x0.9: 12% vs 40%
  - delay 40 ms: 12% vs 41%

  So for single-start agents, planning is worse under every disturbance. Caveat: these starts are already new for these agents, so this is a combined shift. "Crash" includes leaving the bounds; `eval_disturbances.py` now also records `end_cause` (ground / out_of_bounds) and final height, checked by `check_disturbances.py`.
- **R1 learning check at 75k steps:** crash share falling (home 91% to 64%, training area 99% to 83%). Training speed 46-49 steps/s, expected end about 04:20 plus final tests. C1 learns its single start faster (0% home crashes at 75k).
- **A2 (within-run fluctuations vs training logs), 3 Batch 1 1M runs:** step-to-step changes in test late error do not correlate with changes in any logged training statistic (consistency, reward, value and policy losses, policy log std, value scale). Spearman rho between -0.14 and +0.10, all permutation p > 0.07 before correcting for 18 tests.
  - Level correlations up to ±0.34 in two seeds mostly reflect shared slow trends.
  - Conclusion: the training losses do not reveal when the controller gets worse, consistent with objective mismatch (Lambert et al. 2020).
  - Open question: each learning-curve point is only 3 flights with one noise seed, so part of the fluctuation may be evaluation noise. This motivates the snapshot evaluation (many flights per snapshot).
- **Literature checked** (`LITERATURE.md`): TD-MPC2, TD-MPC, MBPO (model exploitation), objective mismatch, Henderson 2018, Agarwal 2021, Molchanov 2019, Eschmann 2024, Peng 2018.
- **Deployment readiness** written (`DEPLOYMENT_READINESS.md`). New `deploy_controller.py` builds the observation from Crazyflie-style state estimates and refuses to produce a firmware command; `check_deployment.py` tests it in closed loop against the simulator.
