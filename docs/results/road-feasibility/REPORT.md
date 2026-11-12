# Directed-road and speed feasibility on Porto and Beijing

Porto includes 3,000 previously scored mixed-population requests per seed. Beijing includes 470 held-out labelled segments per arm and seed. B is the original model, V adds interval-speed supervision, C adds vehicle class, and VC adds both. A trip passes only if every interval passes. Interval rates pool original intervals, counting matching failures in the denominator.

| Cohort | Continuous trips | Continuous and speed-compatible trips | Connected intervals | Connected and speed-compatible intervals | Point projections within 50 m | Median / P95 offset of projected points (m) |
|---|---:|---:|---:|---:|---:|---:|
| Porto-real | 2676/3000 (89.2%) | 2233/3000 (74.4%) | 97.6% | 97.1% | 97.9% | 3.3 / 18.8 |
| Porto-gen-42 | 273/3000 (9.1%) | 156/3000 (5.2%) | 73.1% | 69.0% | 84.2% | 21.2 / 46.3 |
| Porto-gen-43 | 265/3000 (8.8%) | 131/3000 (4.4%) | 72.7% | 68.4% | 84.5% | 21.3 / 46.4 |
| Beijing-real | 233/470 (49.6%) | 177/470 (37.7%) | 95.6% | 95.1% | 96.6% | 7.1 / 34.9 |
| Beijing-B-42 | 3/470 (0.6%) | 0/470 (0.0%) | 55.3% | 41.8% | 69.1% | 22.9 / 46.8 |
| Beijing-B-43 | 0/470 (0.0%) | 0/470 (0.0%) | 39.9% | 19.4% | 60.8% | 22.4 / 46.9 |
| Beijing-V-42 | 2/470 (0.4%) | 0/470 (0.0%) | 51.9% | 37.8% | 66.1% | 22.5 / 46.7 |
| Beijing-V-43 | 2/470 (0.4%) | 0/470 (0.0%) | 57.5% | 51.4% | 67.8% | 23.4 / 46.6 |
| Beijing-C-42 | 0/470 (0.0%) | 0/470 (0.0%) | 55.9% | 42.0% | 69.4% | 22.7 / 46.7 |
| Beijing-C-43 | 0/470 (0.0%) | 0/470 (0.0%) | 34.5% | 10.3% | 57.8% | 21.9 / 46.8 |
| Beijing-VC-42 | 1/470 (0.2%) | 0/470 (0.0%) | 38.0% | 22.4% | 56.3% | 22.4 / 46.8 |
| Beijing-VC-43 | 0/470 (0.0%) | 0/470 (0.0%) | 26.9% | 7.9% | 51.0% | 22.3 / 47.0 |

## Beijing by vehicle class

The same 200 points per request give 199 intervals per trip. Entries are the percentage of original intervals simultaneously connected and speed-compatible. The two values in each model column correspond to seeds 42 and 43.

| Class | Trips | Real | B | V | C | VC |
|---|---:|---:|---:|---:|---:|---:|
| bike | 234 | 94.4% | 45.6% / 19.9% | 38.5% / 55.5% | 44.4% / 9.8% | 20.3% / 7.0% |
| bus | 141 | 96.3% | 36.8% / 16.2% | 39.9% / 47.2% | 37.6% / 8.7% | 23.3% / 7.0% |
| car | 40 | 90.3% | 33.9% / 22.1% | 33.1% / 47.9% | 38.8% / 14.6% | 24.3% / 10.4% |
| taxi | 55 | 98.4% | 44.6% / 23.4% | 33.2% / 47.2% | 45.6% / 13.3% | 27.4% / 12.4% |

## Coverage and speed sensitivity

| Cohort | Trips with points beyond extract box | In-box trips passing / all in-box | Passing with 0.8× / 1× / 1.2× speed ceiling |
|---|---:|---:|---:|
| Porto-real | 6 | 2233 / 2994 | 2072 / 2233 / 2384 |
| Porto-gen-42 | 6 | 156 / 2994 | 112 / 156 / 187 |
| Porto-gen-43 | 3 | 131 / 2997 | 101 / 131 / 166 |
| Beijing-real | 4 | 177 / 466 | 147 / 177 / 190 |
| Beijing-B-42 | 6 | 0 / 464 | 0 / 0 / 0 |
| Beijing-B-43 | 15 | 0 / 455 | 0 / 0 / 0 |
| Beijing-V-42 | 14 | 0 / 456 | 0 / 0 / 0 |
| Beijing-V-43 | 17 | 0 / 453 | 0 / 0 / 0 |
| Beijing-C-42 | 6 | 0 / 464 | 0 / 0 / 0 |
| Beijing-C-43 | 13 | 0 / 457 | 0 / 0 / 0 |
| Beijing-VC-42 | 36 | 0 / 434 | 0 / 0 / 0 |
| Beijing-VC-43 | 55 | 0 / 415 | 0 / 0 / 0 |

## Interpretation

Porto generated trips reproduce parts of the road network, but their whole-trip pass rate is 4–5% versus 74% for observations. Failing a strict trajectory test means at least one transition failed. On Beijing, requiring all 199 transitions to pass leaves zero passing generated trips, even when increasing speed ceilings by 20%. The real Beijing control retains 177 passing trips. This motivates reporting both transition and whole-trip scores.

Speed-only supervision reduces the Beijing passing-interval share for seed 42 and increases it for seed 43. Class alone does not consistently improve it. The joint extension reduces passing intervals at both seeds. These descriptive comparisons use the same requests; 470 segments come from ten held-out participants.

Fractions measure support under this matcher and snapshot. Alternative matching paths, historical changes, multiple modes, 200-point interpolation and observation errors can affect scores. Offsets summarize projected points; projection coverage is reported separately. Speed thresholds are coarse plausibility bounds. The results do not establish individual traffic violations.

See [protocol](PROTOCOL.md), [coverage and paired counts](coverage.json), and [matcher controls](controls.json).
