# Vehicle-label contrasts at similar requested average speeds

The original matched pairs were retained when their requested average speeds differed by at most 0.50 m/s. Each class contributed 600 validation trips. Positive differences refer to the second class minus the first.

| First class | Second class | Pairs before | Pairs retained | Mean signed speed gap before → after (m/s) | Mean absolute speed gap before → after (m/s) |
|---|---|---:|---:|---:|---:|
| bicycle | bus | 113 | 53 | -0.541 → -0.074 | 0.655 → 0.233 |
| bicycle | motorcycle | 41 | 1 | 1.814 → -0.080 | 1.818 → 0.080 |
| bicycle | passenger | 18 | 2 | 1.532 → 0.289 | 1.532 → 0.289 |
| bicycle | taxi | 21 | 4 | 1.553 → 0.271 | 1.556 → 0.288 |
| bus | motorcycle | 20 | 12 | 0.503 → -0.027 | 0.673 → 0.257 |
| bus | passenger | 24 | 4 | 1.182 → 0.128 | 1.230 → 0.138 |
| bus | taxi | 14 | 2 | 1.270 → 0.193 | 1.276 → 0.240 |
| motorcycle | passenger | 577 | 253 | 0.224 → 0.013 | 0.744 → 0.232 |
| motorcycle | taxi | 579 | 254 | 0.280 → 0.003 | 0.780 → 0.237 |
| passenger | taxi | 580 | 245 | 0.071 → 0.005 | 0.697 → 0.212 |

Means and percentile 95% intervals appear for pairs with at least 30 matches and an absolute standardized observed difference of at least 0.20. Each generated difference comes from changing the label on the same request and initial noise. The observed reference compares two matched trips.

| Second minus first | Feature (unit) | Observed | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 |
|---|---|---:|---:|---:|---:|---:|
| bus minus bicycle | slow_time_percent (percentage points) | 39.6 [38.5, 40.8] | 2.54 [2.02, 3.03] | 2.9 [2.43, 3.38] | 24.1 [21.9, 26.3] | 22.5 [20.5, 24.3] |
| bus minus bicycle | slow_entries_per_10min (entries/10 min) | 7.87 [7.55, 8.18] | 0.537 [0.268, 0.773] | 0.74 [0.53, 0.945] | 4.91 [4.38, 5.42] | 4.57 [4.09, 5.04] |
| bus minus bicycle | p90_speed_mps (m/s) | 7.38 [7.24, 7.53] | 0.995 [0.86, 1.12] | 0.886 [0.739, 1.04] | 5.15 [4.81, 5.47] | 4.59 [4.22, 4.95] |
| bus minus bicycle | mean_abs_acceleration_mps2 (m/s²) | 0.363 [0.352, 0.375] | 0.0742 [0.0624, 0.0861] | 0.0501 [0.0427, 0.0565] | 0.253 [0.232, 0.275] | 0.224 [0.203, 0.247] |
| passenger minus motorcycle | mean_abs_acceleration_mps2 (m/s²) | -0.0929 [-0.118, -0.0675] | 0.0102 [0.0067, 0.0137] | 0.0171 [0.0144, 0.0201] | -0.073 [-0.0826, -0.0632] | -0.0703 [-0.0796, -0.0603] |
| taxi minus motorcycle | mean_abs_acceleration_mps2 (m/s²) | -0.112 [-0.137, -0.0866] | 0.0107 [0.00556, 0.0166] | 0.0137 [0.0109, 0.0168] | -0.0778 [-0.0875, -0.0688] | -0.0736 [-0.0836, -0.064] |

Intervals resample matched pairs within this fixed validation sample, holding the models, latent noise, matches and eligibility screen fixed. They ignore possible dependence between trips from the same simulation episode and variation between fits.
