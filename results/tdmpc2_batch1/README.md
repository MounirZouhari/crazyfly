# TD-MPC2 vs PPO, Crazyflie hover: Batch 1 results

Start with **`report_batch1/Crazyflie_TD-MPC2_vs_PPO_Batch1_Report.pdf`**. It explains the task, both
methods, every experiment below, and what the results mean.

Task for every run: one drone starts at (0, 0, 0.2) m and hovers at (0, 0, 1) m for 10 s. After training,
each model flew 3 test flights from 6 start positions (the training start plus 5 it never practised from).

| Folder | Experiment | Runs |
| --- | --- | --- |
| `crazyflie_hover_s0/` | September runs (700, 20k, 50k steps), re-tested | 1 each |
| `batch1_fixed_start/` | TD-MPC2 small settings, 50k steps, home PC | 11 seeds |
| `batch1_small_narval/` | Same as above, repeated on Narval | 11 seeds |
| `batch1_standard/` | TD-MPC2 published settings, 50k steps, Narval | 11 seeds |
| `batch1_standard_1M/` | TD-MPC2 published settings, 1M steps, Narval | 3 seeds |
| `ppo_fixed_start/` | Lab's Brax PPO trained on the same task, Narval | 11 seeds |
| `ppo_existing_checkpoint/` | Lab's earlier PPO model, tested the same way | 1 |
| `comparison_batch1/`, `comparison_with_1M/` | Side-by-side plots and tables | |

Inside each batch folder:

- `analysis/summary.md`: per-seed table, mean ± standard error, results per start position
- `analysis/*.png`: learning curves, flights from the training start, results per start position
- `seed_XX/learning_curve.csv`: test flights every 5,000 training steps
- `seed_XX/evaluation/summary.csv`: one row per final test flight
- `seed_XX/evaluation/trajectories.csv`: every step of every final test flight
- `seed_XX/.hydra/config.yaml`: exact settings of that run

Saved model checkpoints and raw training logs are not in git because of their size. The models are on
the Narval cluster under `/scratch/mounirz/tdmpc2/` (scratch is cleaned after about 60 days).

The plots and summaries were made with `code/src/tdmpc2_training/analyze_seeds.py` and
`compare_methods.py`. `report_batch1/make_report.py` rebuilds the PDF; it reads from `outputs/`, so
point its `OUT` path at this folder if `outputs/` is not present.
