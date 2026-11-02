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

**23:00-23:20.**
- **200k check of R1 (11 seeds), pilot criterion "learning occurs": passed.** Crash share at 100k vs 200k:
  - home 64% to 9%; training area 77% to 14%; held-out block 76% to 29%; outside 92% to 74%.
  - Precision not reached yet (median late error about 0.35 m, almost no successes).
- **C1 (single start) specialises.** Home error 0.12 m at 200k. Its held-out crash share went from 56% (150k) to 100% (200k), so generalisation peaks early and then declines. This is the "memorising" effect, now seen during training.
- **Small model with random starts (3 seeds) learns much faster.** About 0% crashes in all groups by 100k; outside crash share 0-6% at 200-225k (big R1 at 200k: 74%); success 17-42%.
  - Decision: scale the small model to 11 seeds (seeds 3-10, jobs 7662436 then 7662437, disturbance tests 7662438), for a credible small vs big comparison (Q9).
  - Snapshot evaluation chained to every Batch 2 seed (jobs 7662439-7662442).
- **Deployment checks pass** (`check_deployment.py`): the adapter's observation equals the simulator's over 300 closed-loop steps for a Batch 1 agent (offset divided by the start distance) and a Batch 2 agent (metres), with planning and without (largest difference 1.9e-6). Commands stay in the actuator limits. The first decision compiles for 20-44 s, so warm-up is required. The latencies in this run are inflated because the GPU was shared.
- **A1 result (Q1): the Batch 1 seed spread was real, not evaluation noise.** 33 agents x 20 flights from the training start:
  - small 50k (Narval): between-seed SD 0.32 m, within-seed SD 0.09 m, ICC 0.93;
  - small 50k (home): ICC 0.94;
  - published-size 50k: ICC 0.52 (many crashes, so flights vary a lot).
  - Batch 1's 3-flight estimate per seed had an SE of about 0.05 m, small next to the 0.32 m spread.
  - Home seed 0 is at 2.3 m on all 20 flights: its Batch 1 "collapse" is a property of the final agent.
  - With the 50k learning curves still improving and the 1M runs converging, this supports "short runs had not finished training".

**23:20-23:45: E, parallel drones (Q5), home RTX 3060, published settings (`bench_parallel.py`).**
- Simulating N drones in one batch costs 8-10 ms for any N from 1 to 128, so the physics parallelises perfectly.
- Planning grows with N: 10.6 ms for 1 drone, 57 ms for 8, 195 ms for 32, 735 ms for 128.
- One learning update (batch 256) takes 12.4 ms.
- Projected training speed with TD-MPC2's default of one update per simulated step: 32 steps/s with 1 drone, 49 with 8, 55 with 128, so at most about 1.7x faster.
  - With 1 update per 8 steps (a different algorithm): up to 136 steps/s.
  - The 1-drone projection (32) matches the measured home training speed (31 steps/s), which validates the cost model.
- Conclusion: TD-MPC2's per-sample costs (planning for every drone, one update per step) scale with the number of drones, unlike PPO, whose updates are shared over large batches. Larger speedups need fewer updates per sample or cheaper planning during data collection. Both change the algorithm and need their own sample-efficiency check.
- A Narval run of the same benchmark is next.

**23:45-00:05.**
- **Model error, validation run:** `model_error.py` on one Batch 1 small agent (50k, seed 4), 3-step latent error:
  - planner flights: training area 0.0003, held-out block 0.0005, outside 0.0009;
  - policy-only flights from the same starts: 0.0003, 0.0003, 0.0006;
  - reward error outside: 0.0055 planner vs 0.0022 policy;
  - crashes outside: 21/30 planner vs 2/30 policy.
  - This fits H1: the model is less accurate on new starts and the planner steers into its weak regions. One agent only.
- **Submitted:** model error for all 25 Batch 1 runs (job 7663280); chained after R1, C1 and small (7663281-7663284); parallel benchmark on Narval (7663279). All 11 small seeds are in piece 1.
- **Mid-run learning curves** (`outputs/batch2/midrun/random_start_1M/analysis/learning_curves_comparison.png`), R1 at about 325k, C1 300k, small 150k-450k:
  - Random-start training generalises to the held-out block: R1 held-out crash share about 10% at 300k, close to its training-area level. The single-start control C1 crashes 40-60% from training-area starts and 70-100% from the held-out block.
  - The small model leads at equal steps: about 0% crashes in the training area and held-out block from 50k, 0-20% outside. R1 is at about 50% outside at 325k.
  - Small-model success is rising, but the points after 175k come from only 1-3 seeds, so they are noisy.
  - To be confirmed by the final 128-flight evaluations.

**00:00-00:15: model error of all 25 Batch 1 runs (job 7663280; `outputs/batch2/model_error_summary_batch1/`).**
- 1M single-start agents (3 seeds):
  - The planner's flights have 2.8 ± 0.7x (training area) and 2.4 ± 0.2x (held-out block) the 3-step latent prediction error of the policy-only flights from the same starts, paired per seed. Outside, 1.1x: both fail there.
  - Crashes, planner vs policy: training area 99/150 vs 22/150; held-out 83/90 vs 48/90.
  - In the last 0.5 s before a crash, the model error is 4.0x that of completed flights.
- Small 50k agents (11 seeds):
  - Planner / policy error ratio 1.4, 1.9 and 1.3; error 7.0x higher before crashes.
  - Error on outside starts is 2.5x that on training-area starts.
- Published-size 50k agents: almost every flight crashes, so this is uninformative.
- **H1 (model exploitation) supported, correlationally:** the planner drives the drone into states where its world model is wrong, and crashes follow rising model error.
  - Alternative explanation: unusual states near a crash are hard to predict whatever caused them.
  - Causal test started: `eval_planner_variants.py` keeps the trained weights but plans more conservatively (sampling spread 0.5 instead of 2, or half the candidates from the policy network). If exploitation is the cause, these should crash less than the default planner.

**00:05: planner variants, validation run.** Small 50k agent, seed 4, same starts and noise for every variant. Success / crash share in training area, held-out block, outside:
- policy only: 0/0%, 0/0%, 0/7%
- planner as trained: 70/6%, 50/23%, 10/67%
- narrower sampling (max_plan_std 0.5): 64/8%, 63/20%, 10/67%, no change
- half the candidates from the policy network: 76/0%, 90/0%, 13/43%

So anchoring the planner to the policy network's proposals removes most of its extra crashes and keeps (even raises) its precision; only narrowing the search does not. One agent only. Submitted for all 25 Batch 1 runs (job 7663755) and chained after R1, C1 and small (7663756-7663773).
- If it holds across seeds, it is a test-time, no-retraining improvement, and causal evidence for exploitation: the crashes come from candidates far from the policy prior.

**00:35: planner variants on all 25 Batch 1 runs** (job 7663755, `outputs/batch2/planner_variants_summary_batch1/`). Paired change vs the planner as trained, mean ± SE across seeds:
- **1M agents (3 seeds), training-area starts:**
  - half the candidates from the policy network: success +15 ± 2 points, crashes -19 ± 1;
  - narrower sampling: +9 ± 2 / -13 ± 1;
  - policy network only: +21 ± 8 / -49 ± 12.
  - In the held-out block and outside, all changes have the same direction but are smaller and noisier.
- **Small 50k agents (11 seeds):** half-policy candidates reduce crashes by 3-7 points (SE 2-4); success changes are within noise.
- **Published-size 50k agents:** crash 95-100% whatever the planner setting.
- **Revised conclusion:** more conservative planning reliably removes part of the planner's extra crashes, consistent in all 3 1M seeds. So exploitation of model errors is one mechanism. It is not the whole story: for single-start 1M agents the policy network alone remains far more robust (63% vs 15% crashes).
  - The seed-4 result (23% to 0% crashes) was an extreme case.
  - Remaining candidates: the value function is wrong off the training data (MPPI ranks plans by model reward plus value); 256 free candidates still allow exploitation.

**00:35: parallel projection on Narval** (job 7663279, 36 min, `outputs/batch2/parallel_projection_narval/`), with 1 update per simulated step:
- **Published size:** simulate 6.0 ms for any N; plan 6.4 ms (1 drone) to 475 ms (128); update 9.7 ms. Projected 45 steps/s with 1 drone, 68 with 8, 74 with 128 (1.6x).
- **Small size:** plan 1.2 to 16.6 ms; update 3.5 ms. Projected 93, 223 and 271 steps/s (2.9x).
- The 1-drone projections match the measured training speeds (46-49 and 88 steps/s), so the cost model holds.
