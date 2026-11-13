# Beijing velocity scale during diffusion sampling

Coordinate standardization and inversion on all 535 validation requests changed individual coordinate components by at most 0.00391 m and interval speeds by at most 0.00748 m/s. Median chord and original-polyline speeds were 3.14 and 3.22 m/s. A fixed random subset of 96 requests and shared Gaussian noise were used for all four checkpoints. At eight noise levels we compared inferred clean positions from forward-noised observations to those from the model's own reverse chain. The exact injected noise recovered a speed median of 3.16 m/s at every level. Intermediate clean predictions are estimates; the raw noisy states do not represent physical travel.

| Model | Final low-noise prediction given noisy observations | Final free sample | Interval displacement in final sample |
|---|---:|---:|---:|
| Class-only 42 | 5.61 m/s | 76.98 m/s | 276 m |
| Class-only 43 | 5.33 m/s | 133.60 m/s | 466 m |
| Rhythm 42 | 7.21 m/s | 267.93 m/s | 908 m |
| Rhythm 43 | 6.25 m/s | 35.20 m/s | 130 m |

All table entries are medians. Observed displacement was 10.4 m. The discrepancy persists late in the free-running chain and is larger for short time intervals. These controls establish a sampling-quality failure in the expanded Beijing fits. They do not isolate its cause among training, parameterization and reverse-step choices. The fixed request subset is diagnostic. Each JSON records the physical summaries and noise MSE by level.
