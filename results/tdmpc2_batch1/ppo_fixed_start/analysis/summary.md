# ppo_fixed_start: TD-MPC2 hover, 11 seeds

Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Training steps per seed: 60000000. Extra settings: PPO (gpu_training/ppo_training.py defaults, episode_length=500).
Code version: 62b44e2fc449af760449ef88d14714efae67080f (with uncommitted changes)

Values are mean ± standard error across seeds; each seed is first averaged over its own flights.
Late error = mean distance to target over the last 2 s flown. Time in 10 cm = share of the flight after the first 2 s within 0.10 m of the target.

## Each seed, from the training start

| seed | late error (m) | horizontal (m) | vertical (m) | time in 10 cm | crashes (all starts) | training (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (median) | 0.028 | 0.028 | 0.002 | 100% | 9/18 | 11.5 |
| 1 | 0.924 | 0.788 | 0.399 | 3% | 18/18 | 11.5 |
| 2 | 0.028 | 0.028 | 0.002 | 100% | 12/18 | 11.5 |
| 3 | 0.018 | 0.018 | 0.002 | 100% | 15/18 | 11.4 |
| 4 | 0.009 | 0.008 | 0.003 | 100% | 15/18 | 11.3 |
| 5 | 0.153 | 0.066 | 0.125 | 9% | 18/18 | 11.2 |
| 6 | 0.395 | 0.111 | 0.347 | 0% | 18/18 | 11.2 |
| 7 | 0.283 | 0.050 | 0.265 | 0% | 18/18 | 11.2 |
| 8 | 0.018 | 0.018 | 0.002 | 100% | 5/18 | 11.5 |
| 9 | 0.033 | 0.033 | 0.002 | 100% | 15/18 | 11.5 |
| 10 | 0.008 | 0.007 | 0.003 | 100% | 14/18 | 11.5 |
| **all** | **0.172 ± 0.085** | 0.105 ± 0.069 | 0.105 ± 0.047 | 65 ± 15 % | 157/198 | 11.4 ± 0.0 |

## Each test start

| start | late error (m) | best seed (m) | worst seed (m) | crashes |
|---|---:|---:|---:|---:|
| home | 0.172 ± 0.085 | 0.008 | 0.924 | 12/33 |
| plus_x | 0.840 ± 0.093 | 0.032 | 1.145 | 27/33 |
| minus_x | 0.924 ± 0.050 | 0.640 | 1.123 | 33/33 |
| plus_y | 0.988 ± 0.074 | 0.722 | 1.660 | 30/33 |
| minus_y | 0.876 ± 0.053 | 0.488 | 0.999 | 30/33 |
| above | 0.465 ± 0.107 | 0.027 | 1.357 | 25/33 |

## Findings (computed, not interpreted)

- From the training start, the late-flight error was 0.172 ± 0.085 m (median seed 0: 0.028 m; range 0.008 to 0.924 m).
- 7 of 11 seeds ended their home flights within 0.10 m on average.
- Of that error, the horizontal part was 0.105 ± 0.069 m and the vertical part 0.105 ± 0.047 m.
- Crashes in the final test: 157 of 198 flights across all starts.
- Training time per seed (test flights excluded): 11.4 ± 0.0 min.

Plots: learning_curves.png, home_flights.png, test_starts.png
