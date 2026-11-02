# Handoff: Batch 2 overnight run (Nov 1-2, 2026)

Status as of 09:30 on Nov 2. Everything is on the branch `tdmpc2-hover` of the
local repository (WSL: `~/Projects/multi-agentCrazyflieLift`); nothing was
pushed (origin returns 403) and nothing was merged into `main`.

Read first: `BATCH2_EXPLAINED.pdf` (14 pages, plain language), then
`GUIDE.pdf` (66 pages, everything) or `RESULTS_SUMMARY.md` (one answer per
question). `SUPERVISOR_UPDATE.md` is the one-page update for Prof. Bellinger.

## 1. What was completed

- Batch 2 design and code: random starts from a region with a held-out
  block, fixed test and validation starts, metric target offset (option),
  snapshots every 100k steps, resumable split jobs, chained follow-up tests.
- Main experiments, 1M steps each: R1 (published, random starts, 11 seeds),
  C1 (published, single start, 3), D (small, random starts, 11).
- Pilots: H2 (randomised mass, motors and wind, 3 seeds), sharp-reward pilot
  (3 seeds).
- Re-tests of Batch 1: variance decomposition (A1), loss-vs-performance
  fluctuations (A2), Batch 1 agents on the new test starts (R4).
- Follow-up tests on every Batch 2 agent: disturbances, snapshots (best on
  validation vs final), world-model error, planner variants, planning horizon,
  smoothing and motor lag, precision, flight quality, start-outcome maps.
- Speed: decision latency (GPU and CPU), training profile, parallel-drone
  projection.
- Extra tests after the main results: value-network probe (small, sharp and
  R1 agents).
- Documents: guide (PDF), plain-language summary (PDF), results summary,
  supervisor update, research log, experiment registry, deployment readiness,
  next steps, this handoff.

## 2. Code changed

72 files under `code/src/tdmpc2_training/` since the Batch 2 launch commit
`8edab5b` (about 7,000 lines added, documents included), all committed. Modified existing files:
`crazyflie_hover.py` (random starts and the metric option, off by default),
`hover_eval.py` and `eval_hover.py` (test starts, decision rules),
`tdmpc2_jax/train.py` (snapshots, resume with fresh randomness),
`tdmpc2_jax/config_hover.yaml`, `train_hover.py` (environment
registrations), `cluster/seed_job.sh`, `README.md`. The TD-MPC2 algorithm
(`tdmpc2.py`, `world_model.py`) was not changed. New: 5 Batch 2 configs, 2
environment variants (`crazyflie_hover_dr.py`, `crazyflie_hover_sharp.py`),
8 test scripts, 12 analysis scripts, 3 check scripts, the deployment adapter,
14 cluster scripts, `probe_value.py`. The full list is in Chapter 7 of the
guide; `git log 8edab5b^..HEAD` shows every commit.

## 3. Tests passed

- `check_batch2.py` (23 checks): starts, held-out exclusion, observation
  identical to the lab's with options off, metric offset, test starts.
- `check_disturbances.py` (10 checks); `check_deployment.py` (observation
  identical to the simulator over 300 closed-loop steps).
- Regression: old and new code give identical test flights for saved agents;
  identical CPU training curves.
- Validation runs: resume with fresh randomness; copy-and-continue of a
  finished run; horizon test reproduces the trained horizon; smoothing test's
  policy flights bit-identical to the disturbance tester's; sharp-reward
  environment equals the plain one plus the bonus at every step.
- A Narval test job before launch (7659264).

## 4. Tests failed

- The smoothing test's planner flights are not bit-identical to the
  disturbance tester's (first difference about 1e-7 after 2-72 steps). Cause:
  the two programs are compiled differently and round differently, which the
  planner's argmax amplifies. Statistically equivalent over 4 noise seeds;
  documented in the guide (Section 5.16). No other failures.

## 5. Experiments finished

All jobs in `GUIDE.md` Appendix B: 294 Slurm tasks completed, 0 failed,
timed out or cancelled. Main job numbers: R1 7659438, C1 7659439, D 7661493/
7661496 and 7662436/7662437, H2 7664888/7664889, sharp pilot 7668941/7668942.

## 6. Jobs running or queued

- R1 continued to 2M steps, seeds 0-2: 7669543 (piece 2) running, 7669544
  (piece 3, with final tests) queued after it. Expected end early afternoon
  Nov 2. Output: `$SCRATCH/tdmpc2/batch2/random_start_2M`.

## 7. Exact paths

- Results at home: `outputs/batch2/<experiment>/` (`analysis/summary.md` is
  the file to read) and `outputs/batch2/*_summary/`.
- Narval: `/scratch/mounirz/tdmpc2/batch2/<experiment>/seed_XX/`
  (checkpoints with replay buffers, snapshots, logs in `batch2/logs/`).
- Backup: `/project/def-cbelling-ab/mounirz/tdmpc2_scratch_backup_2026-11-01/`
  (`batch2/` added on Nov 2; scratch is purged after 60 days of no access).
- Documents: `code/src/tdmpc2_training/reports/`.

Experiment folders: R1 `random_start_1M`, C1 `fixed_start_1M`,
D `random_start_small_1M`, H2 `randomized_small_1M`, sharp pilot
`sharp_small_1M`, continuation `random_start_2M`.

## 8. Key findings

- Random starts work: R1 - C1 = +20/+34/+20 points (training area/held-out/
  outside), D - Batch 1 small = +59/+72/+55; no crashes inside the area.
- Best controllers (validation-chosen checkpoint): small model with planning
  97/91/70%; published policy network alone 90/81/59%, 0.3 ms per decision.
- At 1M the small planner beats the published one (78 vs 46% from the
  training area); the published model is still improving.
- Planning adds 69 points for the small agents and loses 15-25 points for R1;
  restraining R1's search helps (exploitation).
- Moderate disturbances barely affect the small planner (72-77%); wind of
  2 m/s or more defeats every agent.
- H2 and the sharp reward: negative results (see 9).
- The planner's commands jump; a 40 ms filter (small) or the averaged plan
  (both) fixes it in simulation.

## 9. Hypotheses supported or rejected

- Seed variance in Batch 1 came from short training (supported, ICC 0.93-0.94).
- Random starts give generalisation (supported, both sizes).
- H1, model exploitation explains when planning hurts (supported as one
  mechanism; restraining the search reduces but does not remove the crashes
  of single-start agents).
- H2, planning helps when the model is trained on varied physics (not
  supported in the 3-seed pilot).
- The weak reward gradient causes the precision drift (rejected in its simple
  form: the sharp reward was learned but the drift remained).
- A longer horizon helps (only for the small model, +11 ± 7 at 5 steps; never
  for R1).
- The planner needs its jumpy commands (rejected for the small model).

## 10. Professor questions answered

Q1, Q3, Q4, Q5, Q6, Q7, Q8 and Q10 (blockers and risks identified) are
answered; Q2 and Q9 are answered at 1M steps. See `RESULTS_SUMMARY.md`.

## 11. Questions still open

- Q9/Q2: does the published model catch up with more training (2M running)?
- What causes the precision drift (planner sampling or continual updates)?
- Q10: everything that needs hardware (state estimation, radio delay, motor
  response, firmware-compatible commands).

## 12. Follow-ups launched and why

- Horizon test: does looking further ahead help, or do model errors compound?
- Smoothing test: the flight-quality analysis showed jumpy planner commands.
- Sharp-reward pilot: the reward barely separates 5 cm from 15 cm.
- R1 to 2M: R1 was still improving at 850k-1M.
- Value probe (home): to explain where the precision comes from, then to check
  whether the sharp reward changed the value.

## 13. What next

`NEXT_STEPS.md` and Chapter 9 of the guide. In short: add the 2M results,
adopt validation-based checkpoint selection, test the source of the drift,
then firmware-compatible commands before any hardware step.

## 14. What needs Mounir

- Read `BATCH2_EXPLAINED.pdf`, then send or adapt `SUPERVISOR_UPDATE.md`.
- Decide with Prof. Bellinger: hardware path now or more simulation; which
  controller to carry forward; whether a fair PPO comparison is needed.
- Pushing the branch (origin returns 403: needs his GitHub access or a fork).
- After the 2M run ends: ask Claude to pull and add it, or run
  `cluster/pull_results.sh` and `analyze_all.sh`.

## 15. Safety and reproducibility concerns

- No hardware was touched: no flight, no firmware, no motor command. Every
  hardware step in the documents is conditional on Prof. Bellinger's approval
  and the lab's safety rules.
- GPU training is not bit-for-bit repeatable; results are distributions over
  seeds. The CPU is the exact referee for code changes.
- C1, H2 and the reward pilot have 3 seeds: directions, not sizes.
- The 10 cm success threshold matters; success for every threshold is
  reported alongside.
- Narval scratch is purged after 60 days without access: checkpoints are
  backed up to `/project`; the 2M run must be backed up after it ends.
- The historical PPO results were not changed.
