# batch1_standard_1M: TD-MPC2 hover, 3 seeds

Seeds: 0, 1, 2. Training steps per seed: 1000000. Extra settings: --config-name=config_hover_standard, buffer_size=1000000, save_interval_steps=50000.
Code version: 62b44e2fc449af760449ef88d14714efae67080f (with uncommitted changes)

Values are mean ± standard error across seeds; each seed is first averaged over its own flights.
Late error = mean distance to target over the last 2 s flown. Time in 10 cm = share of the flight after the first 2 s within 0.10 m of the target.

## Each seed, from the training start

| seed | late error (m) | horizontal (m) | vertical (m) | time in 10 cm | crashes (all starts) | training (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.049 | 0.048 | 0.008 | 89% | 12/18 | 347.5 |
| 1 | 0.065 | 0.065 | 0.006 | 97% | 12/18 | 348.0 |
| 2 (median) | 0.058 | 0.056 | 0.014 | 90% | 12/18 | 382.2 |
| **all** | **0.057 ± 0.005** | 0.056 ± 0.005 | 0.009 ± 0.002 | 92 ± 3 % | 36/54 | 359.2 ± 11.5 |

## Each test start

| start | late error (m) | best seed (m) | worst seed (m) | crashes |
|---|---:|---:|---:|---:|
| home | 0.057 ± 0.005 | 0.049 | 0.065 | 0/9 |
| plus_x | 1.027 ± 0.030 | 0.983 | 1.085 | 9/9 |
| minus_x | 1.048 ± 0.065 | 0.965 | 1.176 | 9/9 |
| plus_y | 0.978 ± 0.019 | 0.941 | 0.998 | 9/9 |
| minus_y | 1.385 ± 0.227 | 0.972 | 1.754 | 9/9 |
| above | 0.233 ± 0.106 | 0.052 | 0.420 | 0/9 |

## Findings (computed, not interpreted)

- From the training start, the late-flight error was 0.057 ± 0.005 m (median seed 2: 0.058 m; range 0.049 to 0.065 m).
- 3 of 3 seeds ended their home flights within 0.10 m on average.
- Of that error, the horizontal part was 0.056 ± 0.005 m and the vertical part 0.009 ± 0.002 m.
- Crashes in the final test: 36 of 54 flights across all starts.
- Training time per seed (test flights excluded): 359.2 ± 11.5 min.

Plots: learning_curves.png, home_flights.png, test_starts.png
