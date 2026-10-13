# TD-MPC2 Crazyflie hover: project context

Last updated: October 13, 2026. Picks up where `CLAUDE_PROJECT_CONTEXT.md` (Sept 28 handoff, kept local
only) left off. Read this first when continuing the project on any machine.

## 1. Where things stand in one paragraph

TD-MPC2 (JAX version by Shane Flandermeyer) is integrated with the lab's MJX Crazyflie environment and
was trained on a one-drone hover task: start (0, 0, 0.2) m, target (0, 0, 1) m, 500-step (10 s)
flights. Batch 1 is finished: 11-seed experiments for TD-MPC2 (small and published settings), PPO on the
identical task, and a 3-seed 1-million-step TD-MPC2 run, all tested from 6 start positions and written
up in a PDF report. Most experiments ran on the Digital Research Alliance cluster Narval. Main result:
TD-MPC2 hovers within about 6 cm with roughly 100x fewer environment steps than PPO, but every method
trained from a single start only learns that start. Next is Batch 2 (random start and goal each episode).

## 2. What Professor Bellinger asked for

From his reply to the Sept 28 slides:

1. Keep working on understanding TD-MPC2.
2. Test varied starts, then compare with the existing PPO work, before investigating parallel scaling.
3. Do this with multiple seeds (e.g. 11). Plots: mean + standard error, and all seeds as thin lines with
   the median seed as a thick line.
4. Second batch: randomized start and goal locations in every episode.

Status: 1 ongoing, 2 and 3 done (Batch 1), 4 not started.

## 3. Machines and environments

**Home PC (WSL Ubuntu)**: repo at `~/Projects/multi-agentCrazyflieLift`, Python at `.venv/bin/python`
(3.11.16, made with uv), RTX 3060 12 GB. Key versions: jax 0.7.2, mujoco/mujoco-mjx 3.4.0, flax 0.10.7,
brax 0.13.0, playground 0.0.5, gymnasium 1.2.0, hydra-core 1.3.2, orbax-checkpoint 0.12.5. The Windows
copy at `C:\Users\Mounir\Projects\...` is old and separate; always work in WSL.

**Narval (Digital Research Alliance)**: see section 6.

## 4. Git state

- Branch `tdmpc2-hover`, commit `5730489` "Add TD-MPC2 hover experiments, cluster setup and Batch 1
  results" (364 files, about 38 MB). `main` untouched.
- Git identity for this repo only: Mounir Zouhari <zouharimounir1@gmail.com>.
- Remotes:
  - `origin` = `uo-mirl-lab/multi-agentCrazyflieLift` (lab repo). Push was refused (403): the GitHub
    account MounirZouhari has no write access. Ask Prof. Bellinger to add it, then
    `git push -u origin tdmpc2-hover`.
  - `mine` = `https://github.com/MounirZouhari/crazyfly.git` (private). Push with
    `git push -u mine tdmpc2-hover` from the Ubuntu terminal (GitHub sign-in only works interactively).
- WSL git uses the Windows Git Credential Manager (set in `~/.gitconfig`).
- Kept out of git on purpose via `.git/info/exclude` (local only): `CLAUDE_PROJECT_CONTEXT.md`, the
  weekly-update `.pptx`, `*:Zone.Identifier` files, `outputs.zip`. `outputs/` is in `.gitignore`.
- Results that are in git live in `results/tdmpc2_batch1/` (curated copy of `outputs/`, see its README).

## 5. Code map (code/src/tdmpc2_training)

| File | What it does |
| --- | --- |
| `crazyflie_hover.py` | Gymnasium adapter around the MJX env. `place_drone()` sets any start/target. Reset is jit-compiled and each step does one GPU to CPU copy (about 44 steps/s at home, was about 30). |
| `train_hover.py` | Entry point; registers `CrazyflieHover-v0`, uses `config_hover` unless `--config-name` is given. |
| `tdmpc2_jax/train.py` | Vendored trainer. Changes: agent built by `build.py`; test flights every `eval_interval_steps` appended to `learning_curve.csv` (with `train_seconds`, test time excluded); final checkpoint always saved; old end-of-run gym evaluation removed. |
| `tdmpc2_jax/build.py` | Builds the agent from a Hydra config; shared by training and evaluation so checkpoints always restore into matching shapes. |
| `tdmpc2_jax/config_hover.yaml` | Small settings (latent 128, 128 plans x 3 iterations, batch 64, 500 seed steps, 100 seed updates). Adds `eval_interval_steps: 5000`, `eval_flights: 3`, `eval_flight_steps: 500`. |
| `tdmpc2_jax/config_hover_standard.yaml` | Published TD-MPC2 defaults on top of `config_hover`: encoder 256, latent 512, 512 plans x 6 iterations, 64 elites, 24 policy samples, temperature 0.5, batch 256, 2500 seed steps and updates, encoder lr 9e-5. |
| `hover_eval.py` | Test flights. All flights in one call fly in parallel inside one compiled `lax.scan`. Defines `TEST_STARTS` and the metrics (see section 8). |
| `eval_hover.py` | Reloads a run's newest checkpoint from its own `.hydra/config.yaml` and flies 3 flights from each of the 6 starts. Writes `evaluation/summary.csv` and `trajectories.csv`. |
| `run_seeds.py` | Runs a seed list locally (`--parallel 2` works on the 3060), skips finished seeds, records `batch_info.json`, then calls `analyze_seeds.py`. |
| `analyze_seeds.py` | Per-batch plots (mean + SE, all seeds + median seed) and `analysis/summary.md`. Median seed = median late error from the training start. Flight curves stop at the first crash. |
| `compare_methods.py` | Side-by-side plots: by environment steps (log), by training time, by start position, plus a table. |
| `ppo_hover.py` | Trains the lab's Brax PPO (gpu_training/ppo_training.py defaults) on the exact same task: fixed reset, 500-step episodes, same action rescaling as TD-MPC2. `--load ... --raw-actions` tests an existing PPO checkpoint the same way. |
| `cluster/` | Narval scripts, see section 6. |

Run everything from the repo root, e.g.
`.venv/bin/python code/src/tdmpc2_training/run_seeds.py --seeds 0-10 --parallel 2`.

## 6. Digital Research Alliance (Narval)

### Account and login
- CCDB user `mounirz` (CCI bwd-893), group `def-cbelling-ab` (sponsor Colin Bellinger). Access on Fir,
  Narval, Nibi, Rorqual, Trillium, "opportunistic use" (default priority, no RAC). Project storage 1 TB.
- MFA (Duo) and per-cluster access activation (ccdb.alliancecan.ca/me/access_systems) were done on Oct 12.
- `~/.ssh/config` in WSL defines `Host narval` with ControlMaster (`ControlPersist 4h`). The user logs in
  once with `ssh narval` (password + Duo push) in the Ubuntu terminal; other commands can then reuse that
  connection with `ssh -o BatchMode=yes narval ...`. The shared login does not survive overnight; if it
  is stale, `ssh -O exit narval`, remove `~/.ssh/cm-*`, and log in again.
- Host key fingerprint (ED25519): `SHA256:pTKCWpDC142truNtohGm10+lB8gVyrp3Daz4iR5tT1M`.

### Storage on Narval
- Code: `~/multi-agentCrazyflieLift` (copied with rsync, verified byte-identical to home before every
  submission).
- Results: `/scratch/mounirz/tdmpc2/<batch>/seed_XX/` and job logs in `/scratch/mounirz/*.out`.
  **Scratch is purged after about 60 days.** Checkpoints exist only there (and in `outputs/` at home for
  the home runs). Copy anything worth keeping to `/project/def-cbelling-ab/` (backed up, not done yet).

### Python on Narval (why a container)
The Alliance wheelhouse has no MuJoCo 3.4.0 (MJX only up to 3.3.0) and different flax/orbax/brax
versions, and its Python refuses normal PyPI Linux wheels. To keep the physics identical to home, Python
runs inside a plain `python:3.11-slim` Apptainer image:
- `cluster/setup_env.sh` (login node, once): pulls `~/containers/python311.sif`, creates venv
  `~/envs/tdmpc2-ct`, installs `cluster/requirements-lock.txt` (exact `uv pip freeze` of the home venv).
  Verified: all 99 packages identical to home.
- `mujoco_playground` downloads MuJoCo Menagerie on import, which fails on offline GPU nodes. The home
  copy (commit 14ceccf, 1704 files, no .git) was copied into
  `~/envs/tdmpc2-ct/lib/python3.11/site-packages/mujoco_playground/external_deps/mujoco_menagerie`.
- `cluster/python.sh` runs the venv's Python in the container with `--nv --cleanenv`.

### Jobs
- GPU type: 2/7 A100 slices, `--gres=gpu:a100_2g.10gb:1`, 2 CPUs, 8 GB. Full A100s had queues of
  hundreds of jobs (estimated start hours away); slices usually started within minutes.
- `cluster/test_job.sh`: setup check + 3000-step speed test of both configs. Note: its printed speed for
  the standard config is misleading because the 1000-3000 window contains the 2500 pre-training updates.
- `cluster/seed_job.sh`: one seed per array task. Env vars `CONFIG` (default `config_hover_standard`),
  `BATCH`, `STEPS`, `EXTRA` (extra Hydra overrides). Submit from `$SCRATCH`, e.g.
  `cd $SCRATCH && sbatch --array=0-10 ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/seed_job.sh`
- `cluster/ppo_job.sh`: PPO, one seed per task, `STEPS` default 20M (Batch 1 used 60M).
- Useful: `sq`, `squeue -u $USER --start`, `sacct -j <id> -X -o JobID,State,Elapsed`, `partition-stats`.
  The CCDB "group usage" web page updates with a delay of about a day.

### Measured speeds (environment steps per second, steady state)
| Run | Home RTX 3060 | Narval 2/7 slice |
| --- | --- | --- |
| TD-MPC2 small | 44 (34 each with 2 in parallel) | 89 |
| TD-MPC2 standard | not measured | 42-49 |
| PPO, 2048 parallel envs | not measured | 79M steps in about 11.4 min |

### Jobs run on Oct 12-13
| Job ID | What | Outcome |
| --- | --- | --- |
| 5418119 | test, full A100 | cancelled (queue estimate 2 a.m.) |
| 5418330 | test, slice | failed: Menagerie download (fixed) |
| 5418777 | test, slice | completed |
| 5420951 | TD-MPC2 standard, 11 seeds x 50k | completed |
| 5422314 | TD-MPC2 standard, 3 seeds x 1M (8 h limit, about 6.4-7 h each) | completed |
| 5425800 | PPO smoke test + lab checkpoint test | completed |
| 5425898 | PPO, 11 seeds x 60M (Brax rounded to 78.6M) | completed |
| 5425899 | TD-MPC2 small, 11 seeds x 50k (same-hardware rerun) | completed |

About 17.6 slice-hours used by the evening of Oct 12, plus about 20 for the 1M runs.

## 7. Batch 1 results

All runs: same task, reward, observations and action meaning. Every trained model flew 3 test flights
from each of 6 starts (home (0,0,0.2) = training start; plus_x, minus_x, plus_y, minus_y at 0.5 m to the
side, height 0.2; above (0,0,1.5)), target (0,0,1). Values are mean ± standard error across seeds.

| ID | Experiment | Where | Seeds | Late error, training start | Crashes, all 6 starts | Folder |
| --- | --- | --- | --- | --- | --- | --- |
| A | September run, re-tested | home | 1 | 0.30 m (other starts 0.18-0.38 m) | 0/18 | `crazyflie_hover_s0` |
| B | TD-MPC2 small, 50k | home | 11 | 0.54 ± 0.19 m (median 0.34) | 28/198 | `batch1_fixed_start` |
| C | TD-MPC2 small, 50k | Narval | 11 | 0.32 ± 0.09 m (median 0.16) | 11/198 | `batch1_small_narval` |
| D | TD-MPC2 standard, 50k | Narval | 11 | 1.06 ± 0.14 m (median 1.02) | 182/198 | `batch1_standard` |
| G | TD-MPC2 standard, 1M | Narval | 3 | 0.057 ± 0.005 m, 92% of time within 10 cm | 36/54 (0/9 from home, 36/36 from the 4 sideways starts) | `batch1_standard_1M` |
| E | PPO, same task, 78.6M steps | Narval | 11 | 0.17 ± 0.09 m (median 0.028; 7 of 11 within 10 cm all flight) | 157/198 | `ppo_fixed_start` |
| F | Lab's existing PPO model (trained on random tasks, 60M) | tested on Narval | 1 | 0.064 m (3.5-8.8 cm from 5 starts, 21 cm from above) | 0/18 | `ppo_existing_checkpoint` |

Training time per seed on a Narval slice: small 14.0 min, standard 20.2 min, PPO 11.4 min, 1M about 6 h.
B took 26 min per seed at home with 2 seeds sharing the GPU (not comparable).

## 8. Interpretation and things learned

- **Seed variation is large.** Same settings gave 0.08 m to 2.35 m at 50k steps. B and C agree within
  their standard errors (22 runs of the small settings in total).
- **Height is learned quickly, sideways position is not.** At 50k steps the drone reaches 1 m in about
  0.8 s and holds it, then drifts about 3 cm/s sideways in a random direction per seed.
- **The drift was mainly lack of practice.** With 1M steps (G) all 3 seeds hover within about 5-6 cm with
  no crashes from the training start; they first got under 10 cm at 105k, 130k and 175k steps.
- **Sample efficiency vs wall clock.** TD-MPC2 reaches a precise hover after about 150-200k steps; PPO
  needs about 15-20M (roughly 100x more). PPO is far faster on the clock (2048 parallel envs): best
  hover after about 4 min versus about an hour for TD-MPC2.
- **Both methods memorise the single training start.** PPO and the 1M TD-MPC2 crash from almost all
  sideways starts. The 50k TD-MPC2 looked more robust only because it had not specialised yet (this
  corrects a claim made on Oct 12). The lab's PPO trained on random starts works everywhere, which
  motivates Batch 2.
- **Late training instability.** B seed 0 was at 0.27 m with no crashes at 45k steps, then collapsed to
  2.3 m by 50k (confirmed through two independent test paths). PPO peaks around 20-30M steps, then
  degrades. Final-checkpoint scores include this.
- **Not bit-for-bit repeatable.** GPU arithmetic plus the faster reset mean seed 0 today is not seed 0 of
  September (0.30 m then, 2.35 m now).
- **Bigger settings are slow starters**: worst at 50k, best TD-MPC2 result with long training.

## 9. Known caveats in the setup

- The "previous action" in the observation lags one step (built before the action history is updated in
  `crazyflie_env._get_obs`). Same for every method.
- The target direction in the observation is divided by the distance at the start of the flight (0.8 m
  from home, 0.94 m from the sideways starts), so the same physical state looks different depending on
  the start. Relevant for single-start generalisation.
- JAX TD-MPC2 vs official: dropout stays on in planning, Q averaged over all 5 heads, AdamW per network,
  entropy not scaled by action dim, `deterministic=True` takes the best elite (tested: sampling elites
  instead made no difference).
- Late-flight error (mean distance over the last 2 s flown) is not comparable between crashed and
  surviving flights; read it with the crash count.
- Each seed is tested with only 3 flights per start.
- Lab PPO code: hidden sizes passed to `build_ppo_train_fn` are ignored, so stock Brax networks are used
  (policy 4x32, value 5x256). Brax rounds training steps up in chunks of about 1M per curve point.
- `crazyflie_env.reset(random_reset=True)` draws target x/y and z from the same key (and likewise for the
  start), and also randomises orientation, velocity, mass, inertia and motor gain. Batch 2 should use its
  own start/goal sampling through `place_drone`, not that mode.

## 10. Report

`outputs/report_batch1/Crazyflie_TD-MPC2_vs_PPO_Batch1_Report.pdf` (copy in
`results/tdmpc2_batch1/report_batch1/`), 20 pages, written for a reader new to RL. Rebuild with
`make_report.py` in the same folder; reportlab is not in the project venv, so run it with
`~/.local/bin/uv run --no-project --with reportlab --with matplotlib --with numpy python make_report.py`.
It reads `outputs/`.

## 11. Next steps

1. Push `tdmpc2-hover` to `mine` (and to `origin` once write access is granted).
2. Back up `/scratch/mounirz/tdmpc2` (checkpoints) to `/project/def-cbelling-ab/` before scratch is purged.
3. Confirm experiment G properly: 11 seeds x about 200k steps, for both small and standard settings
   (about 1.2 h per seed on a slice).
4. Batch 2: random start and goal every episode, 11 seeds per method, same plots. Proposed ranges (to
   confirm with Prof. Bellinger): x and y in ±1 m, height 0.3-1.5 m, start and goal at least 0.3 m apart,
   positions only (drone level and at rest, no physics randomisation). Needs a new environment option and
   test set; likely more steps than 50k per run.
5. Later: the research question on parallel scaling (TD-MPC2 with many simulated drones), with the
   pipeline profiled first.

## 12. Working with Mounir

New to RL, TD-MPC2 and JAX. Wants explanations from the basics, tied to real files and numbers, in small
steps, with clear separation between "it runs" and "it is validated". Prefers asking before long
training runs, no changes to existing code without saying so, honest corrections when a conclusion turns
out wrong, and no long dashes or AI attribution in commit messages.
