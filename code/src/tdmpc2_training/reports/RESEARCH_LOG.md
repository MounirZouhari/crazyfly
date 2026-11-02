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

**00:35: health check.** 84 jobs completed, 25 running, none failed. Small pilot piece 1 took 104-112 min per 500k steps; piece 2 is running.

**00:40-00:45: H2 pilot launched.**
- **Hypothesis H2:** planning helps when the world model has trained on the relevant variation (interpolation) and hurts when it has not (H1, extrapolation).
- **Setup:** `crazyflie_hover_dr.py` (new env id `CrazyflieHoverRandomized-v0`, registered lazily so existing runs are untouched). The small model with random starts, plus in every training flight: mass and inertia x1.0-1.15, motor strength x0.9-1.0, steady wind 0-3 m/s from a random direction. The agent does not observe these values.
- **Validated:**
  - the draws stay in range;
  - the flight state matches the draws;
  - the wind in training pushes the drone exactly as in the disturbance tests (0.417 m in 1 s at 3 m/s, both);
  - a 1,500-step training run works.
- **Launched:** 3 seeds x 1M in two 3 h pieces (7664888 then 7664889), with disturbance tests (7664892) and planner variants (7664893) chained.
- **Comparison:** the same tests on the nominal-trained small agents. In-range disturbances: mass, motors, wind up to 3 m/s. Beyond the range: wind 4 m/s, gust 6 m/s, delays.

**02:07: first fully trained Batch 2 agents.** Small model, random starts, seeds 0-2, 1M steps (job 7661496); final tests of 128 flights each (`outputs/batch2/random_start_small_1M/analysis/`). Success / crash share:

| Decision rule | training area | held-out block | outside | Batch 1 starts |
| --- | --- | --- | --- | --- |
| with planning | 95 ± 4% / 0% | 97 ± 2% / 0% | 68 ± 10% / 2% | 94 ± 6% / 0% |
| policy network only | 1% / 0% | 0% / 0% | 8% / 0% | 6% / 0% |
| 64x2 planner | 94% / 0% | 88% / 0% | 69% / 2% | 94% / 0% |

- Median late error from training-area starts: 5.0-6.9 cm per seed.
- With planning vs the Batch 1 1M single-start agents: +75 to +94 points of success (95% bootstrap intervals all far above 0).
- **Planning vs policy only, paired:** +94 [+86, +100] (training area), +97 [+93, +100] (held-out), +60 [+27, +83] (outside). For the single-start 1M agents it was -26 [-46, -8].
- **Interpretation, for when and why the model helps (Q7):**
  - Trained on data covering the test situations, the policy network alone is safe but imprecise (no crashes, about 1% success), and planning with the world model adds the precision.
  - Trained on one start, planning off that data exploits model errors and crashes.
  - So the world model helps where its training data covers the situation, and hurts beyond it.
- 3 seeds; seeds 3-10 due about 03:30.
- **Where the precision comes from** (median late error of full flights from training-area and held-out starts, per seed):
  - planner: 5.1-7.4 cm (horizontal 4.1-7.4 cm, vertical 0.9-2.9 cm);
  - policy only: 15-41 cm (horizontal 13-16 cm, vertical 8-38 cm);
  - 64x2 planner: 5.1-8.7 cm.
- The policy network alone holds the wrong height. Planning corrects it almost entirely, mostly the thrust (hovering needs 76% of maximum thrust, so small biases matter).
- Hypothesis: the policy network, trained to maximise the value with an entropy bonus, has a biased mean thrust, and short-horizon planning against the model's predicted rewards removes the bias.

**02:19-02:27: horizon test (added after the main experiments, eval only).**
- **Question:** TD-MPC2 plans 3 steps (0.06 s) ahead. Would a longer horizon help, or do world-model errors compound beyond the horizon it was trained on?
- **Setup:** `eval_horizon.py` flies the same agent with horizons 1, 3, 5 and 10 (changed only at test time), from the 50 training-area and 30 outside starts, nominal, in 3 m/s steady wind and with 15% extra mass; same sensor noise for every horizon.
- **Validated:** the planner sizes its plan and warm start from the horizon setting (`tdmpc2.py` lines 196-201, `hover_eval.py` line 203), including horizon 1; the pilot ran without errors.
- **Pilot (Batch 1 small agent, seed 4, single start; 380 s at home).** Success / crash from training-area starts, nominal: horizon 1: 0% / 54%; 3: 64% / 10%; 5: 22% / 18%; 10: 2% / 42%. Same ranking with extra mass; everything fails in 3 m/s wind.
- **Reading:** the trained horizon is best for this agent; both shorter and longer horizons are worse. One agent only.
- **Launched:** `cluster/horizon_job.sh` on every Batch 2 agent (small 0-2 now; small 3-10, R1, C1 and randomized chained on their training). Commit b86b66b. `analyze_horizon.py` summarises across seeds.

**02:40-03:00: what the within-run swings consist of, and how the drone flies.** Small random-start agents (D), first finished seeds, all from existing test files.
- **Snapshots (seeds 0-2):** with planning, train-area success swings between about 0% and 100% from one 100k snapshot to the next (within-run SD 0.23-0.45 against 0.05-0.07 expected from 50 flights), but no snapshot of any seed crashed. What swings is the steady-state precision: the median late error of full flights moves between about 3 and 15 cm (once 55 cm), around the 10 cm success threshold, so a shift of a few cm flips almost every flight at once. Success is a threshold measure and amplifies small changes; the continuous late error is now reported next to it (`analyze_snapshots.py`).
- **Policy network alone across snapshots:** no crashes either, but its height offset changes sign between snapshots (-23 to +51 cm). One seed was precise with the policy alone at 300k-400k (100% success) and lost it later. So the policy's imprecision is an unstable offset, not a fixed bias.
- **Command jitter (`analyze_flight_quality.py`, new):** in steady hover (last 5 s), the planner's thrust changes by 31 ± 2% of its range per step and its moments by 34 ± 1%; 49 ± 2% of all steps have a command at its limit. Policy network alone: 7 ± 2%, 8 ± 0.3%, 2 ± 0.3%. The tests use the deterministic planner (`hover_eval.py` line 210: best elite's first action, no added noise), so the jitter is built into how the planner acts: it executes one sampled candidate, and many candidates sit at the limits after clipping. Same pattern for the Batch 1 agents (planner 24-27% thrust jitter, 47-51% saturated).
- **Why it matters:** a real Crazyflie's motors cannot follow commands that jump by a third of their range every 20 ms; the motor dynamics would filter them, which changes the closed loop. This is a deployment blocker not visible in success rates (Q10), and a cost of planning not visible in Q8.
- **Follow-up (eval only), `eval_smoothing.py`:** the same agents with (a) the planner as trained, (b) the first action of the planner's averaged plan, (c) the planner's commands through a 40 ms low-pass filter, (d) the policy alone; each with instant motors and with first-order motor lags of 30 and 60 ms (assumed values to probe sensitivity, not identified Crazyflie values).
- **Validation of `eval_smoothing.py` (Batch 1 small agent, seed 4, 50 starts):**
  - policy network alone: bit-identical to `DisturbedFlightTester`;
  - planner as trained: NOT bit-identical. The same program run twice is bit-identical (50/50 flights), but the new program first differs at steps 2-72 by 1e-8 to 2e-7 (one float32 rounding step) with identical positions before. Differently compiled GPU programs round differently, and the planner's choice between near-tied candidates amplifies it;
  - statistically equivalent over 4 noise seeds: success 66/66, 68/68, 66/68, 64/64%; crashes 10/8, 10/10, 10/10, 10/12%;
  - all variants of the smoothing test run in one compiled program (the variant, filter and lag are inputs), so their comparisons stay exactly paired;
  - the 40 ms filter reaches 63.2% of a step after 40 ms, as it should;
  - jitter on this agent: planner 30.7%, averaged plan 24.4%, filtered 8.8%.

**03:00: a hypothesis for the precision drift (from the reward, not yet tested).** The reward per step (`gpu_training/crazyflie_env.py` lines 293, 321-327) is 0.001 + 0.01 (1 - tanh d) + (previous d - d) - 1 on a crash. Hovering still at 5 cm earns 0.0105 per step, at 15 cm 0.0095: a difference of 0.001 per step. With discount 0.99 the value of hovering is about 1.05, and the difference between 5 and 15 cm about 0.1. In TD-MPC2's value encoding (101 bins over symlog -10..10, bin width 0.2) that is 0.05, a quarter of a bin. Within the 3-step planning horizon the rewards differ by only 0.003. So precision within about 15 cm is weakly anchored by the reward, which may explain why the steady-state error drifts between 3 and 15 cm from checkpoint to checkpoint while crashes stay at zero. Testable next: retrain with a sharper proximity term near the target (for example a Gaussian of width 5 cm) and compare the snapshot-to-snapshot drift.

**03:06: two follow-ups launched (commit 3f9f21b, home and Narval identical, no uncommitted changes).**
- **Smoothing test** (`eval_smoothing.py`, eval only) on the 11 small and 11 R1 agents: jobs 7668917 (small 0-2), 7668921 (small 3-10, after 7662437), 7668922 (R1, after 7659438).
- **Sharp-reward pilot** (`config_batch2_sharp_small`, `crazyflie_hover_sharp.py`): 3 seeds x 1M of the small model, identical to D except an extra reward 0.01 exp(-d^2 / (2 x 0.05^2)) near the target. Tests the 03:00 hypothesis. Jobs 7668941 -> 7668942 (two 3 h pieces), snapshot evaluation 7668947. Validation: bonus values exact; observations, termination and physics identical to the plain task over 600 steps; reward = plain + bonus at every step; a 1,500-step training run with snapshots works. Prediction if the hypothesis holds: smaller final late error and less snapshot-to-snapshot drift than D, with crashes still near zero. Expected done about 07:15 (training) and 07:45 (snapshots).

**03:13: first smoothing-test results (small model, 4 seeds, preliminary).** Training-area and held-out starts; success, then thrust jitter and steps at a limit:
- instant motors: planner as trained 72%, 29.5%, 50%; averaged plan 72%, 23.6%, 1%; planner + 40 ms filter 73% (+2 ± 1 paired), 8.7%, 2%; policy only 25%, 7.2%, 2%.
- 30 ms motor lag: no change for any variant.
- 60 ms motor lag: planner 71%, averaged plan 72%, but planner + 40 ms filter 54% (-17 ± 8 paired): the filter's and the motors' lags add up.
- **Reading so far:** the planner does not need its jumpy commands. Filtering them (or acting on the averaged plan) keeps the precision while making the commands as smooth as the policy network's, and the agents tolerate slow motors. The filter must be sized together with the real motor lag. Waiting for all 11 small seeds and the R1 agents.

**03:20: learning curves at about 850k steps (R1, C1, D, H2; `random_start_1M/analysis/learning_curves_comparison.png`).**
- R1 (published, random starts): crashes from training-area starts near 0 after about 300k, from the held-out block after about 400k; from outside starts still about 30% at 850k, falling slowly. Success (10 cm) only 20-30%.
- C1 (published, single start): 40-60% crashes from training-area starts, 80-100% from the held-out block and outside, throughout.
- D (small, random starts): crashes near 0 everywhere by 50-100k (about 5% outside); success 60-75%.
- H2 (small, randomised physics): crashes like D; success more variable.
- **Reading (mid-run):** random starts remove crashes for both sizes, but the small model learns much faster and is more precise at equal steps. The published model may need more than 1M steps, or a different learning rate, to match it. Final tests will tell.

**03:25: follow-up tests on 8 of the 11 small random-start agents (D).**
- **Model error (H1 check):** along the planner's flights the world model is as accurate or more accurate than along the policy network's (paired ratio 0.73 ± 0.10 training area, 0.70 ± 0.07 held-out, 1.4 ± 0.8 outside), and its 3-step latent error is about 1e-5, 15-30 times below the Batch 1 small agents (3e-4). No sign of exploitation where the data covers the situation.
- **Planner variants:** restraining the planner now costs success (narrower search -10 ± 8, half the candidates from the policy -7 ± 6 points from the training area) instead of preventing crashes. Consistent with the above: nothing to exploit, so the full search is best.
- **Horizon:** 5 steps beats the trained 3 (+15 ± 9 points from the training area, +9 ± 6 outside); 10 steps is no better in the area (-2 ± 15) and crashes more outside (14% vs 5%); 1 step is much worse (-44 ± 16). With an accurate model a slightly longer look-ahead helps; for the inaccurate Batch 1 pilot agent every change from 3 was worse.
- 3 m/s steady wind: no horizon gives any success (0%), and crashes stay low (0-3% in the area): the drone is pushed off target but kept in the air.

**03:24: continuing a finished run elsewhere (prepared, not launched).** Question raised by the curves: is the published model just slower to learn, so that it would catch up with the small one given more steps? `cluster/copy_run_job.sh` copies a finished run (checkpoint with replay buffer, snapshots, history; not the tests) to a new batch folder, where `seed_job.sh` resumes it. Validated at home: a 1,500-step run copied and resumed to 3,000 steps continues from 1,500, extends its learning curve and adds snapshots next to the copied ones. Plan if R1 is still improving at 1M: seeds 0-2 to 2M in three 3 h pieces (1.35M, 1.7M, 2M), about 9 h per seed; it would finish around 13:30, after the handoff, and be reported as still running.

**03:26: R1 extension launched (commit 622b694).** At 850-875k the R1 curves were still improving (success rising from about 20% at 600k to 30% at 800k; outside crashes falling from about 50% to 35%), so seeds 0-2 continue to 2M: copy 7669541 (after R1), then 7669542 -> 7669543 -> 7669544 (to 1.35M, 1.7M, 2M; final tests in the last piece) in `batch2/random_start_2M`. Cost about 3 x 9 h on GPU slices. Expected done about 13:30, after the handoff; to be cancelled before it starts if R1's final tests make it pointless.

**03:30: experiment D final (small model, random starts, 11 seeds; job 7662437 all exit 0).** Success / crashes:
- with planning: training area 78 ± 10% / 0%, held-out 83 ± 10% / 0%, outside 59 ± 9% / 3 ± 2%, Batch 1 starts 80 ± 10% / 0%;
- policy network only: 9-12% success everywhere, no crashes;
- 64x2 planner: as good as the full planner (+2 [-2, +7] points, paired).
- vs Batch 1 small single-start agents: +59 [+35, +81] (training area), +72 [+48, +91] (held-out), +55 [+38, +72] (outside).
- planner - policy (paired): +69 [+30, +96], +74 [+36, +98], +48 [+18, +72].
- Seeds: 8 of 11 at 86-100% from the training area (median late error 4-8 cm); seeds 3, 8, 10 at 54%, 20%, 8% with no crashes (9-13 cm). The first three pilot seeds (95%) were among the good ones; the full set is lower, which is why the scale-up mattered.
- Precision of full flights (training area + held-out): planner 7.4 ± 0.9 cm (horizontal 7.1, vertical 1.6), policy 20.6 ± 3.0 cm (horizontal 15.1, vertical 11.9; height offset +7.9 ± 4.2 cm, but its sign changes between snapshots), 64x2 7.8 ± 0.9 cm.

**03:35: snapshots of 10 small seeds (seed 9 pending).**
- Within-run SD of training-area success from 500k on: 0.10-0.48 per seed, against 0.04-0.07 expected from 50 flights. The swings are real changes of the agent.
- Mostly precision: the median late error of full flights moves between about 4 and 15 cm, sometimes 20-60 cm. Crashes are rare, but not absent: seed 6's 500k snapshot crashed in 44% of the training-area flights (median error 89 cm), then recovered.
- **Best snapshot chosen on the 40 validation starts vs the final one** (test starts): training area +23 ± 12 points, held-out +10 ± 11, outside +8 ± 9. Mean test success from the training area: 96% at the validation-chosen snapshots vs 74% at the final ones. It rescues exactly the weak seeds: seed 8 14% -> 100%, seed 10 2% -> 100%, seed 3 44% -> 76%.
- Policy network alone: best vs final +32 ± 14 points, but from a low level (its error averages 25 cm from 500k on).
- **Reading:** longer training does not remove the within-run drift; choosing the checkpoint on validation starts does most of the job, at the cost of testing every snapshot.

**03:40: disturbances, small random-start agents (11 seeds, 50 training-area starts).**
- Success with planning: none 77%; wind 1 / 2 / 3 / 4 m/s: 60 / 1 / 0 / 0%; gusts 2 / 4 / 6 m/s: 66 / 15 / 0%; mass x1.05 / 1.10 / 1.15: 76 / 74 / 72%; motors x0.95 / 0.90: 76 / 73%; delay 20 / 40 ms: 77 / 76%. Policy alone: 9-10% everywhere (18% at 1 m/s wind).
- Planning minus policy (paired): +61 to +68 points under mass, motor and delay changes, +42 [+14, +69] at 1 m/s wind, +57 [+16, +91] for 2 m/s gusts.
- Crashes: none for either rule up to 3 m/s wind and 2 m/s gusts (0-1%; the drone is pushed off target but stays up). Beyond: planner 41% vs policy 29% at 4 m/s wind, 67% vs 21% for 6 m/s gusts; 71% of the planner's wind crashes are out of bounds.
- **Reading:** for random-start agents, planning keeps the precision under every moderate disturbance, but under strong wind (states never seen in training) it crashes more than the policy network alone, as H1 predicts. Compare the single-start 1M agents: planner 66-100% crashes in every wind condition vs policy 14-100%.

**03:47: snapshots, all 11 small seeds.** Best snapshot chosen on validation vs final, test success: training area +21 ± 11 points (97% vs 76% on average), held-out +9 ± 10, outside +7 ± 8. Within-run SD of training-area success from 500k: 0.34 ± 0.03 (test noise alone: 0.04-0.07). Seed 9 behaves like the good seeds (final = best = 100%). Experiment D is complete except seed 9's policy-only snapshot test.

**03:55: probing the value network near the target (`probe_value.py`, new; no flights).** The drone at rest and level on a grid of points within 30 cm of the target (three planes, 2 cm steps), exact observation (deploy adapter), previous command a plain hover; value = mean of the value networks for the policy's action. 9 snapshots of small random-start agents: precise ones (seed 10 at 400k, seed 8 at 700k, seed 2 at 400k, seed 5 at 1M; flown errors 3-7 cm) and imprecise ones (seed 10 and seed 8 at 1M, seed 2 at 500k, seed 3 at 500k and 1M; 10-56 cm).
- In every snapshot the value is almost flat around the target: it changes by 0.000-0.004 within 10 cm and 0.003-0.012 within 20 cm, on a value of about 1.06-1.11. It is even slightly higher away from the target (the reward's improvement term lets a drone that is away still collect that distance), but a value that matched the rewards would rise by about 0.1 per 10 cm; the learned one rises by a few thousandths.
- Precise and imprecise snapshots look the same, so the value's shape does not explain which checkpoints drift.
- **Reading:** the value network does not resolve positions within about 20 cm of the target. The precision must come from the world model's predicted rewards summed over the planning horizon (the improvement term gives about 1 per metre over the 3 steps). This fits three results at once: the policy network alone, trained against the value, is imprecise (21 cm); a horizon of 1 step, which leans on the value, is much worse (-35 points); a horizon of 5 is slightly better (+11). It also supports the sharp-reward pilot: a reward that is steep within 5 cm gives both the predicted rewards and the value something to resolve.

**08:45: final results of R1, C1, H2 and the reward pilot (pull + `analyze_all.sh`, 242 s, no errors).** 294 Slurm tasks completed since the launch, none failed; only the R1 extension (7669543 running, 7669544 queued) is left.

**R1 final (published model, random starts, 11 seeds; 7659438 all exit 0).** Success / crashes:
- with planning: training area 46 ± 8% / 0%, held-out 39 ± 9% / 2 ± 1%, outside 20 ± 6% / 26 ± 5%, Batch 1 starts 39 ± 10% / 0%;
- policy network only: 60 ± 11% / 0%, 64 ± 13% / 0%, 40 ± 10% / 6 ± 2%, 65 ± 11% / 0%;
- 64x2 planner: 58 ± 9% / 0%, 46 ± 12% / 0%, 32 ± 7% / 9 ± 2%.
- vs C1 (with planning): +20 [+1, +39], +34 [+17, +51], +20 [+9, +32]; vs Batch 1 published 1M: +26 [+7, +45], +37 [+20, +52], +17 [+7, +30].
- vs D (small model): -32 [-57, -5], -44 [-68, -18], -39 [-59, -18].
- planner - policy (paired): -15 [-37, +10], -25 [-47, -1], -20 [-40, +0]; planner - 64x2: -12 [-24, +3], -8 [-24, +10], -12 [-18, -6].
- Precision of full flights: planner 12.7 ± 1.5 cm (horizontal 12.4, vertical 1.6), policy 9.3 ± 1.4 cm, 64x2 9.9 ± 1.1 cm. Seeds: median late error 6.6-18.7 cm; 4 seeds above 15 cm.
- The learning curves and the snapshots still rise at 1M (training-area success of the snapshots 0.26 at 600k, 0.34 at 900k, 0.42 at 1M; late error falling from about 20 to 13 cm): R1 has not finished learning. The extension to 2M (seeds 0-2) is the right test and keeps running.
- **Reading:** random starts help the published model too (no crashes inside the area, +20 to +34 points over C1), but at 1M it is much less precise than the small model, and its full planner is worse than its own policy network and than a smaller search.

**R1 follow-ups.**
- Model error, planner / policy (paired): 1.17 ± 0.07 (training area), 1.49 ± 0.08 (held-out), 2.00 ± 0.14 (outside), and 4.4 times higher in the last 0.5 s before a crash. Between the small random-start agents (0.68) and the single-start agents (2.4-2.8).
- Planner variants: narrower search +10 ± 3 / +11 ± 6 / +8 ± 4 points, half the candidates from the policy +13 ± 8 / +8 ± 9 / +12 ± 4; outside crashes fall from 26% to 15% and 10%. Restraining the search helps: some exploitation even inside the training distribution.
- Horizon: 1 step -5 ± 6, 5 steps -0 ± 6, 10 steps -13 ± 7 (training area); in 3 m/s wind crashes grow with the horizon (11, 32, 41, 49% for 1, 3, 5, 10 steps).
- Smoothing: averaged plan +2 ± 2 (keeps precision, saturation 46% -> 1%); 40 ms filter -11 ± 3 (unlike D, where it is free); motor lag 30 / 60 ms costs the planner 8 / 19 points but the policy network nothing (61, 61, 59%).
- Disturbances: the planner is below the policy network under every condition (-17 to -37 points); in 3 m/s wind the planner crashes in 32% of the flights, the policy network in 0%.
- Snapshots: best (validation) vs final, planner +7 ± 4 points (49% vs 42% from the training area). **Policy network alone at its validation-chosen snapshot: 90 ± 6% / 81 ± 11% / 59 ± 8% test success (training area / held-out / outside), 0 / 0 / 6% crashes, median late error 7.2 cm**, vs 60 / 62 / 39% at the final snapshot (+30 ± 9 points). This is as good as the small model's planner at its final snapshot, at 0.3 ms per decision instead of 11 ms on a CPU.
- Value probe (seeds 7 and 3 at 1M): flat like the small model's (0.001-0.002 within 10 cm, 0.005-0.007 within 20 cm).

**C1 final (published model, single start, metric target; 3 seeds).** Planning: 26 ± 7% / 35 ± 10% (training area), 4 ± 3% / 82 ± 11% (held-out), 0% / 92 ± 4% (outside). Policy network: 47 / 15%, 18 / 54%, 0 / 98%. C1 - Batch 1 published 1M: +6 [-9, +21], +2 [-3, +8], -2 [-3, +0]: the metric target offset changes nothing; R1's improvement comes from the random starts. Same exploitation signature as Batch 1 (planner / policy model error 1.68 ± 0.35; narrower search -10 points of crashes).

**H2 final (small model, random starts + randomised mass, motors, wind; 3 seeds).**
- Without disturbance: planning 13 ± 7% / 0% (training area), 7 ± 4% / 0% (held-out), 4 ± 4% / 0% (outside); policy network 34 ± 31% / 0%. Precision of full flights 14.3 ± 2.0 cm (planner), 12.5 ± 2.9 cm (policy), vs 7.4 cm for D.
- H2 - D (planning): -65 [-86, -42], -76 [-94, -55], -55 [-73, -35].
- In wind: crashes at 4 m/s 34% (D 41%), 6 m/s gusts 55% (D 67%); late error of full flights at 3 m/s 47 cm (D 67 cm). At 2-3 m/s no success for either.
- Planner - policy: -21 [-80, +16] (3 seeds, wide).
- **Reading:** H2 not supported in this pilot. Training on varied physics bought a little robustness in strong wind but cost most of the precision in calm air, and the planner does not beat the policy network. A likely reason: the agent sees no history, so it cannot tell which mass or wind it is flying in, and its world model predicts an average; that average is wrong in every particular case. Testing this needs a history or an estimate of the disturbance in the observation.

**Reward pilot final (sharp reward near the target, small model, 3 seeds).**
- Final snapshots: planning 37 ± 19% / 0% (training area), 66 ± 33% / 0% (held-out), 23 ± 17% / 2 ± 2% (outside); D - pilot +41 [+7, +78] from the training area. Seed 2's final snapshot is at 0%.
- Best snapshot on validation: 91 ± 5% / 78 ± 19% / 47 ± 3% (D: 97 / 91 / 70%).
- Drift from 500k: success SD 0.25 ± 0.04 (D 0.34 ± 0.03; test noise 0.06-0.07); median late error per snapshot 9.3 cm on average (D about 10 cm); late-error SD 2.6 ± 0.6 cm (D 11.0 ± 3.2 cm, driven by excursions); snapshots with a median error above 20 cm: 0 of 18 (D 9 of 66).
- Value probe (6 snapshots): the value now peaks at the target: it drops by 0.03-0.05 at 10 cm and 0.07-0.13 at 20 cm (D: about 0.00), and its maximum is 2-3 cm from the target in 3 of 6 snapshots.
- **Reading:** the extra reward is learned (the value now sees the target), and the large excursions of precision disappeared in these 3 seeds, but the typical precision and the swings of success around the 10 cm threshold are unchanged. So the flat value was not the cause of the drift. Not a cure; a pilot that narrows the search.

**09:30: documents finished.** `GUIDE.pdf` complete with every final result (66 pages; Section 5.18 waits for the 2M run), `BATCH2_EXPLAINED.pdf` (a 14-page plain-language version: the experiment names, what each one is for, the results by question, where every file is), `RESULTS_SUMMARY.md`, `SUPERVISOR_UPDATE.md`, `NEXT_STEPS.md`, `HANDOFF.md`, registry statuses. Batch 2 copied to `/project` (scratch purge). Still running: R1 to 2M (7669543, then 7669544).

**12:00: R1 continued to 2M steps, final (seeds 0-2; 7669542-7669544 all exit 0; backed up to /project, 901 files, identical).** Paired, same seeds at 1M -> 2M, success / crashes:
- planning: training area 66/0 -> 72/3%, held-out 65/2 -> 75/7%, outside 33/22 -> 24/58%; typical final error per seed 12.3, 8.8, 7.7 -> 7.2, 10.3, 3.1 cm;
- 64x2 planner: 51/0 -> 70/0%, 33/0 -> 70/0%, 39/6 -> 57/22%;
- policy network only: 36/0 -> 29/0%, 53/0 -> 30/0%, 20/3 -> 18/34%.
- **Reading:** the second million adds a few points and precision inside the area, much less than the first million, and makes the agents crash far more from outside starts. Hypothesis: the replay buffer holds 1M transitions, so from 1M on it discards the early, wide-ranging flights; the agent then trains only on competent flights inside the area and specialises. Both the planner and the policy network crash more outside, which points at the data. To check: distance distributions in the 1M and 2M buffers. At 2M the published planner is level with the small model at 1M inside the area and far less safe outside, after about 14 h of training against 4 h.
- 300 Slurm tasks completed for Batch 2 in total, none failed. Guide Section 5.18, the summaries and both PDFs updated.
