# Method comparison: one-drone hover, start (0, 0, 0.2), target (0, 0, 1)

Mean ± standard error across seeds. Late error = mean distance to target over the last 2 s flown.

| method | seeds | training steps | training time (min) | late error, training start (m) | late error, all starts (m) | crashes, all starts |
|---|---:|---:|---:|---:|---:|---:|
| TD-MPC2 small 50k | 11 | 50,000 | 14.0 ± 0.3 | 0.320 ± 0.092 | 0.454 ± 0.121 | 11/198 |
| TD-MPC2 standard 50k | 11 | 50,000 | 20.2 ± 0.2 | 1.061 ± 0.140 | 1.122 ± 0.052 | 182/198 |
| TD-MPC2 standard 1M | 3 | 1,000,000 | 359.2 ± 11.5 | 0.057 ± 0.005 | 0.788 ± 0.047 | 36/54 |
| PPO | 11 | 78,643,200 | 11.4 ± 0.0 | 0.172 ± 0.085 | 0.711 ± 0.032 | 157/198 |
