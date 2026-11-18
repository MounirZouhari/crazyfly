# CLAUDE.md: Crazyflie + TD-MPC2 (Mounir's work)

Context for Claude Code. Current as of Nov 7, 2026 (Batch 2 and the 2M run are done).

## The project in one paragraph

This is the lab repo `uo-mirl-lab/multi-agentCrazyflieLift` (University of Ottawa, supervisor Prof. Colin Bellinger).
The root `README.md` describes the other students' MAPPO work. Leave it as it is.
Mounir Zouhari's work is under `code/src/tdmpc2_training/`. It plugs TD-MPC2 (a world model that plans,
vendored from ShaneFlandermeyer/tdmpc2-jax at `5b05ff4`) into the lab's MJX Crazyflie simulator. The task
is one drone flying to a target and hovering there. It uses the lab's 31-number observation, 4 commands
(collective thrust in [0, 0.35] N plus roll, pitch and yaw moments in [-1, 1]), the lab's reward, and 500-step episodes. Batch 1 (Oct 12-13) used one fixed start.
Batch 2 (Nov 1-2) gave every training flight a random start and was tested on held-out starts. No hardware has been touched.

## Read these first (in order)

1. `code/src/tdmpc2_training/reports/HANDOFF.md`: what was done and where it is. Its "what needs Mounir" section lists his open decisions.
2. `code/src/tdmpc2_training/reports/NEXT_STEPS.md`: the research plan, in order.
3. `code/src/tdmpc2_training/reports/RESULTS_SUMMARY.md`: one answer for each of Prof. Bellinger's ten questions.
4. `code/src/tdmpc2_training/README.md`: commands (train, evaluate, follow-up tests, analysis).
5. `code/src/tdmpc2_training/EXPERIMENTS.md`: the task, the test groups, the metrics and the launch log.
6. For depth: `reports/GUIDE.md` (66 pages, built into `GUIDE.pdf` by `make_guide.py`) and `reports/RESEARCH_LOG.md`.

Older and partly stale: `CONTEXT.md` (Batch 1, Narval setup, Oct 13) and `CLAUDE_PROJECT_CONTEXT.md`
(the first handoff, Oct 12, kept local only). Both are still useful for Narval details and the environment contract.

## Experiment names (Mounir gets lost in abbreviations: show this table before any results)

| Name | What | Seeds | Folder in `outputs/batch2/` |
| --- | --- | --- | --- |
| R1 | Published-size TD-MPC2, random starts, 1M steps | 11 | `random_start_1M` |
| C1 | Published size, single start (= Batch 1 setup), 1M | 3 | `fixed_start_1M` |
| D | Small TD-MPC2, random starts, 1M | 11 | `random_start_small_1M` |
| H2 | Small, plus randomised mass, motors and wind | 3 | `randomized_small_1M` |
| Sharp pilot | Small, extra reward within ~5 cm of the target | 3 | `sharp_small_1M` |
| R1 to 2M | R1 seeds 0-2 continued to 2M | 3 | `random_start_2M` |

The results are success rates within 10 cm, given as training area / held-out block / outside the area.

- Random starts are what make the agent generalise. R1 beats C1 by +20/+34/+20 points.
- D with planning scores 78/83/59. At its validation-chosen snapshot it scores 97/91/70, the best controller so far.
- R1 with planning scores 46/39/20. R1's policy network alone, at its best snapshot, scores 90/81/59 at 0.3 ms per decision. The big planner exploits errors in its world model.
- H2 and the sharp reward are negative results. The sharp reward was learned, but the precision drift stayed.
- R1 at 2M does better inside the area (66 -> 72%) but crashes far more from outside starts (22 -> 58%). The likely cause is that the 1M-transition replay buffer drops the early wide-ranging flights. This is not yet checked, and it is step 0 of `NEXT_STEPS.md`.

Batch 1 caveats found in the Nov 1 audit:
- The PPO learning rate never annealed (step units were wrong).
- PPO auto-reset carried `info` over between episodes.
- The target offset in the observation was divided by the initial distance. Batch 2 uses `metric_target`.

The historical PPO results were left unchanged.

## Code map (`code/src/tdmpc2_training/`)

- `crazyflie_hover.py`: Gymnasium adapter over the lab env (`code/src/gpu_training/crazyflie_env.py`). It has the `random_starts` and `metric_target` options, both off by default so Batch 1 is unchanged. Variants: `crazyflie_hover_dr.py` and `crazyflie_hover_sharp.py`.
- `train_hover.py`: entry point. Configs are Hydra files in `tdmpc2_jax/config_*.yaml`.
- `hover_eval.py`: test starts, metrics and batched GPU test flights. `eval_hover.py` loads and tests a run (`--no-mpc`, `--plan 64x2`, `--snapshot N`).
- `eval_*.py`: follow-up tests that write folders inside a run. `analyze_*.py` analyses them, and `analyze_all.sh` runs every analysis.
- `check_batch2.py`, `check_disturbances.py` and `check_deployment.py`: automatic checks. Run them after touching the env or the tests.
- `tdmpc2_jax/tdmpc2.py` and `world_model.py`: the algorithm. It has not been modified, so keep it that way unless asked.
- `cluster/`: Narval Slurm jobs, the Apptainer Python wrapper (`python.sh`), the locked requirements and `pull_results.sh`.
- `reports/`: all documents, figures, videos and decks.

## Environment and tooling

- The repo lives in WSL Ubuntu at `~/Projects/multi-agentCrazyflieLift`. Claude Code runs on Windows and opens it over `\\wsl.localhost\...`.
  The Read, Write and Edit tools work on that path. Windows git does not ("dubious ownership").
- Run git and Python through WSL. Write a `.sh` file into the scratchpad and run it from **PowerShell** with
  `wsl -d Ubuntu -e bash /mnt/c/Users/Mounir/AppData/Local/Temp/claude/.../scratchpad/x.sh`.
  Git Bash mangles that path, and inline `bash -lc "..."` mangles quoting.
- Python: `.venv/bin/python` (3.11.16, uv). Versions: jax 0.7.2, mujoco/mjx 3.4.0, flax 0.10.7, brax 0.13.0, playground 0.0.5. GPU: RTX 3060 12 GB.
- Narval: user `mounirz`, account `def-cbelling-ab`. Use `ssh narval` (ControlMaster). Mounir must log in first with password + Duo.
  Use 2g.10gb A100 slices (short queue). Runs live in `/scratch/mounirz/tdmpc2/batch2/...`, which is purged after 60 days.
  They are backed up to `/project/def-cbelling-ab/mounirz/tdmpc2_scratch_backup_2026-11-01/`.
- Results at home: `outputs/` (gitignored). A curated copy for git is in `results/tdmpc2_batch1/`.

## Git

- Work on branch `tdmpc2-hover`. Do not touch `main`.
- Pushes to `origin` (the lab repo) are refused with 403. Push to `mine` (`github.com/MounirZouhari/crazyfly`, private) only when Mounir asks.
- Commit as `Mounir Zouhari <zouharimounir1@gmail.com>` (repo-local). Write messages in his voice, with no AI attribution and no long dashes.
- These stay local only, via `.git/info/exclude`: `CLAUDE_PROJECT_CONTEXT.md`, weekly-update `.pptx` files at the top level, `*:Zone.Identifier` and `outputs.zip`.

## How to work with Mounir

- He is new to RL, TD-MPC2 and JAX. Explain from the basics, define terms on first use, and tie every claim to a file, function or number.
- Separate "it runs" from "it is scientifically validated". Correct earlier conclusions openly when they turn out wrong.
- Run anything longer than about 30 s (training, evals, rsync, waits) in the background from the start. Never make him background it himself.
- Ask before starting new training runs, broad rewrites, or changes to existing lab code.
- Every hardware step (flight, firmware, motor commands) needs Prof. Bellinger's approval and the lab's safety rules.
- GPU training is not bit-for-bit repeatable, so report distributions over seeds. The CPU is the exact referee for code changes.
