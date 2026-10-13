# batch1_standard: TD-MPC2 hover, 11 seeds

Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Training steps per seed: 50000. Extra settings: --config-name=config_hover_standard.
Code version: 62b44e2fc449af760449ef88d14714efae67080f (with uncommitted changes)

Values are mean ± standard error across seeds; each seed is first averaged over its own flights.
Late error = mean distance to target over the last 2 s flown. Time in 10 cm = share of the flight after the first 2 s within 0.10 m of the target.

## Each seed, from the training start

| seed | late error (m) | horizontal (m) | vertical (m) | time in 10 cm | crashes (all starts) | training (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.654 | 1.493 | 0.626 | 0% | 18/18 | 20.2 |
| 1 | 1.725 | 1.603 | 0.588 | 0% | 17/18 | 19.5 |
| 2 | 0.321 | 0.295 | 0.110 | 18% | 13/18 | 21.3 |
| 3 (median) | 1.020 | 0.954 | 0.302 | 3% | 17/18 | 20.0 |
| 4 | 0.676 | 0.563 | 0.283 | 0% | 16/18 | 20.7 |
| 5 | 1.520 | 1.457 | 0.398 | 0% | 18/18 | 19.3 |
| 6 | 1.098 | 0.989 | 0.323 | 3% | 17/18 | 20.3 |
| 7 | 0.770 | 0.501 | 0.540 | 0% | 18/18 | 19.9 |
| 8 | 0.639 | 0.569 | 0.245 | 3% | 14/18 | 20.0 |
| 9 | 0.823 | 0.695 | 0.354 | 1% | 16/18 | 20.5 |
| 10 | 1.423 | 1.375 | 0.313 | 8% | 18/18 | 20.2 |
| **all** | **1.061 ± 0.140** | 0.954 ± 0.140 | 0.371 ± 0.047 | 3 ± 2 % | 182/198 | 20.2 ± 0.2 |

## Each test start

| start | late error (m) | best seed (m) | worst seed (m) | crashes |
|---|---:|---:|---:|---:|
| home | 1.061 ± 0.140 | 0.321 | 1.725 | 24/33 |
| plus_x | 1.259 ± 0.120 | 0.691 | 1.986 | 32/33 |
| minus_x | 1.067 ± 0.065 | 0.797 | 1.465 | 33/33 |
| plus_y | 1.226 ± 0.058 | 0.946 | 1.531 | 33/33 |
| minus_y | 1.182 ± 0.130 | 0.856 | 2.253 | 33/33 |
| above | 0.940 ± 0.110 | 0.354 | 1.565 | 27/33 |

## Findings (computed, not interpreted)

- From the training start, the late-flight error was 1.061 ± 0.140 m (median seed 3: 1.020 m; range 0.321 to 1.725 m).
- 0 of 11 seeds ended their home flights within 0.10 m on average.
- Of that error, the horizontal part was 0.954 ± 0.140 m and the vertical part 0.371 ± 0.047 m.
- Crashes in the final test: 182 of 198 flights across all starts.
- Training time per seed (test flights excluded): 20.2 ± 0.2 min.

Plots: learning_curves.png, home_flights.png, test_starts.png
