# batch1_fixed_start: TD-MPC2 hover, 11 seeds

Seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Training steps per seed: 50000. Extra settings: none (config_hover.yaml).
Code version: 62b44e2fc449af760449ef88d14714efae67080f (with uncommitted changes)

Values are mean ± standard error across seeds; each seed is first averaged over its own flights.
Late error = mean distance to target over the last 2 s flown. Time in 10 cm = share of the flight after the first 2 s within 0.10 m of the target.

## Each seed, from the training start

| seed | late error (m) | horizontal (m) | vertical (m) | time in 10 cm | crashes (all starts) | training (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.345 | 2.265 | 0.596 | 0% | 12/18 | 25.4 |
| 1 | 0.408 | 0.407 | 0.013 | 0% | 6/18 | 25.7 |
| 2 (median) | 0.339 | 0.336 | 0.037 | 8% | 0/18 | 25.6 |
| 3 | 0.265 | 0.256 | 0.066 | 2% | 3/18 | 25.6 |
| 4 | 0.827 | 0.822 | 0.067 | 12% | 0/18 | 27.9 |
| 5 | 0.500 | 0.400 | 0.292 | 0% | 1/18 | 27.9 |
| 6 | 0.078 | 0.075 | 0.015 | 54% | 6/18 | 27.5 |
| 7 | 0.416 | 0.353 | 0.203 | 10% | 0/18 | 27.2 |
| 8 | 0.238 | 0.212 | 0.081 | 8% | 0/18 | 25.6 |
| 9 | 0.304 | 0.293 | 0.080 | 23% | 0/18 | 25.6 |
| 10 | 0.236 | 0.235 | 0.028 | 0% | 0/18 | 22.0 |
| **all** | **0.542 ± 0.189** | 0.514 ± 0.184 | 0.134 ± 0.053 | 11 ± 5 % | 28/198 | 26.0 ± 0.5 |

## Each test start

| start | late error (m) | best seed (m) | worst seed (m) | crashes |
|---|---:|---:|---:|---:|
| home | 0.542 ± 0.189 | 0.078 | 2.345 | 3/33 |
| plus_x | 0.540 ± 0.108 | 0.200 | 1.362 | 5/33 |
| minus_x | 0.579 ± 0.193 | 0.221 | 2.448 | 4/33 |
| plus_y | 0.562 ± 0.156 | 0.138 | 1.918 | 4/33 |
| minus_y | 0.771 ± 0.231 | 0.038 | 2.432 | 9/33 |
| above | 0.512 ± 0.130 | 0.193 | 1.326 | 3/33 |

## Findings (computed, not interpreted)

- From the training start, the late-flight error was 0.542 ± 0.189 m (median seed 2: 0.339 m; range 0.078 to 2.345 m).
- 1 of 11 seeds ended their home flights within 0.10 m on average.
- Of that error, the horizontal part was 0.514 ± 0.184 m and the vertical part 0.134 ± 0.053 m.
- Crashes in the final test: 28 of 198 flights across all starts.
- Training time per seed (test flights excluded): 26.0 ± 0.5 min.

Plots: learning_curves.png, home_flights.png, test_starts.png
