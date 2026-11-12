# Beijing vehicle-label switching experiment

470 identical trip headers and common random numbers per intervention. Each of the four fits generated four labels per request. Values below are per-trip features. Speed is measured from model-clock positions.

| Model | Median within-request range of mean speed (m/s) | Range of P90 speed (m/s) | Range of slow time (percentage points) | Range of absolute acceleration (m/s²) |
|---|---:|---:|---:|---:|
| C-42 | 1.12 | 2.12 | 0.50 | 0.17 |
| C-43 | 2.31 | 4.05 | 0.00 | 0.31 |
| VC-42 | 5.88 | 10.43 | 0.00 | 0.87 |
| VC-43 | 4.07 | 7.16 | 0.00 | 0.60 |

## Matched observational contrasts

Signed difference of class means, computed on observed pairs and on label switches on the same request pool. A positive value means the first named class has the larger feature. Slow time is reported in percentage points. Counts of agreement are descriptive, not independent hypothesis tests.

### mean_speed_mps

| Contrast | Observed pairs | Observed difference | C-42 | C-43 | VC-42 | VC-43 |
|---|---:|---:|---:|---:|---:|---:|
| bus minus bike | 94 | +1.35 | +0.26 | -0.22 | -2.68 | +1.50 |
| car minus bike | 6 | +1.65 | +0.45 | -0.25 | +0.77 | -2.70 |
| taxi minus bike | 12 | +0.94 | +0.80 | +0.97 | -6.17 | -4.45 |
| car minus bus | 13 | -0.01 | -0.16 | -0.04 | +4.03 | -3.89 |
| taxi minus bus | 21 | +0.45 | +0.65 | +1.68 | -3.34 | -4.42 |
| taxi minus car | 16 | +0.75 | -0.14 | -0.41 | -5.26 | +0.52 |

### p90_speed_mps

| Contrast | Observed pairs | Observed difference | C-42 | C-43 | VC-42 | VC-43 |
|---|---:|---:|---:|---:|---:|---:|
| bus minus bike | 94 | +4.97 | +0.36 | -0.13 | -4.66 | +2.58 |
| car minus bike | 6 | +5.55 | +1.38 | -0.47 | +1.51 | -4.75 |
| taxi minus bike | 12 | +5.02 | +1.36 | +2.12 | -10.86 | -7.38 |
| car minus bus | 13 | -0.29 | -0.40 | -0.10 | +7.09 | -7.13 |
| taxi minus bus | 21 | +1.46 | +1.19 | +2.79 | -5.97 | -7.51 |
| taxi minus car | 16 | +1.96 | -0.30 | -0.54 | -9.19 | +1.08 |

### slow_fraction

| Contrast | Observed pairs | Observed difference | C-42 | C-43 | VC-42 | VC-43 |
|---|---:|---:|---:|---:|---:|---:|
| bus minus bike | 94 | +14.73 | -0.11 | -0.02 | +0.03 | -0.01 |
| car minus bike | 6 | +16.16 | -0.71 | +0.21 | -0.04 | +0.04 |
| taxi minus bike | 12 | +14.61 | -0.90 | +0.08 | -0.06 | +0.02 |
| car minus bus | 13 | +1.16 | -0.08 | -0.02 | -0.27 | -0.04 |
| taxi minus bus | 21 | -7.13 | -0.16 | +0.01 | +0.04 | +0.01 |
| taxi minus car | 16 | +0.57 | -0.02 | -0.02 | +0.25 | +0.00 |

### mean_abs_accel_mps2

| Contrast | Observed pairs | Observed difference | C-42 | C-43 | VC-42 | VC-43 |
|---|---:|---:|---:|---:|---:|---:|
| bus minus bike | 94 | +0.02 | +0.08 | +0.10 | -0.70 | +0.43 |
| car minus bike | 6 | -0.01 | +0.13 | -0.06 | +0.12 | -0.47 |
| taxi minus bike | 12 | -0.07 | +0.21 | +0.41 | -1.51 | -1.13 |
| car minus bus | 13 | -0.07 | -0.01 | -0.01 | +0.66 | -0.79 |
| taxi minus bus | 21 | -0.03 | +0.20 | +0.47 | -0.77 | -1.19 |
| taxi minus car | 16 | +0.06 | -0.07 | +0.00 | -1.08 | +0.09 |

C adds class to the original header. VC also uses interval-speed supervision in training. Real trips are paired across labels within prespecified tolerances for header attributes. A real trip can appear in different class-pair comparisons. Exact support counts, unrestricted means, all paired effects and replay checks are in the aggregate JSON. See [protocol](PROTOCOL.md).
