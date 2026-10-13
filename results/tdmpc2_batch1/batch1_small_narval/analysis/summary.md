# batch1_small_narval: TD-MPC2 hover, 11 seeds

Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Training steps per seed: 50000. Extra settings: --config-name=config_hover.
Code version: 62b44e2fc449af760449ef88d14714efae67080f (with uncommitted changes)

Values are mean ± standard error across seeds; each seed is first averaged over its own flights.
Late error = mean distance to target over the last 2 s flown. Time in 10 cm = share of the flight after the first 2 s within 0.10 m of the target.

## Each seed, from the training start

| seed | late error (m) | horizontal (m) | vertical (m) | time in 10 cm | crashes (all starts) | training (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.469 | 0.442 | 0.146 | 5% | 0/18 | 13.7 |
| 1 | 0.626 | 0.624 | 0.049 | 0% | 0/18 | 13.0 |
| 2 | 0.467 | 0.416 | 0.171 | 0% | 6/18 | 13.3 |
| 3 (median) | 0.162 | 0.160 | 0.028 | 30% | 3/18 | 14.2 |
| 4 | 0.065 | 0.063 | 0.013 | 58% | 0/18 | 13.7 |
| 5 | 0.096 | 0.096 | 0.009 | 68% | 0/18 | 12.7 |
| 6 | 1.045 | 1.006 | 0.274 | 0% | 0/18 | 13.8 |
| 7 | 0.101 | 0.099 | 0.018 | 20% | 2/18 | 14.7 |
| 8 | 0.132 | 0.131 | 0.017 | 45% | 0/18 | 12.7 |
| 9 | 0.229 | 0.226 | 0.034 | 0% | 0/18 | 15.7 |
| 10 | 0.129 | 0.128 | 0.009 | 20% | 0/18 | 16.1 |
| **all** | **0.320 ± 0.092** | 0.308 ± 0.088 | 0.070 ± 0.027 | 22 ± 8 % | 11/198 | 14.0 ± 0.3 |

## Each test start

| start | late error (m) | best seed (m) | worst seed (m) | crashes |
|---|---:|---:|---:|---:|
| home | 0.320 ± 0.092 | 0.065 | 1.045 | 0/33 |
| plus_x | 0.491 ± 0.157 | 0.079 | 1.628 | 3/33 |
| minus_x | 0.464 ± 0.154 | 0.065 | 1.762 | 0/33 |
| plus_y | 0.660 ± 0.197 | 0.076 | 2.215 | 5/33 |
| minus_y | 0.499 ± 0.184 | 0.036 | 2.197 | 3/33 |
| above | 0.292 ± 0.090 | 0.028 | 0.923 | 0/33 |

## Findings (computed, not interpreted)

- From the training start, the late-flight error was 0.320 ± 0.092 m (median seed 3: 0.162 m; range 0.065 to 1.045 m).
- 2 of 11 seeds ended their home flights within 0.10 m on average.
- Of that error, the horizontal part was 0.308 ± 0.088 m and the vertical part 0.070 ± 0.027 m.
- Crashes in the final test: 11 of 198 flights across all starts.
- Training time per seed (test flights excluded): 14.0 ± 0.3 min.

Plots: learning_curves.png, home_flights.png, test_starts.png
