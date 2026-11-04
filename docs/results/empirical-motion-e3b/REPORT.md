# Empirical motion versus learned path diffusion

Same 33 validation conditions, K=8, 512 points, candidate paths and shared decoder. Empirical baseline draws entire class-matched training arc-increment sequences, with replacement, then maps them onto the same paths. No validation future is supplied. Frozen test untouched.

Diffusion has three training seeds; empirical motion has three resampling seeds and no learned generator. Standard deviations are descriptive, not a significance test. Both use the same frozen original-Hilbert GAF and raw TCN selectors/auditors. Teacher agreement does not establish utility.

| Generator | Selection | Graph % | Speed W1 m/s | Acceleration W1 m/s2 | Travel W1 m | GAF audit % | TCN audit % | Cap % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| route-diverse-e4r | all_candidates | 100.00 +/- 0.00 | 1.02 +/- 0.09 | 0.22 +/- 0.01 | 145.73 +/- 6.62 | 42.47 +/- 0.94 | 76.11 +/- 0.80 | 0.00 +/- 0.00 |
| route-diverse-e4r | random | 100.00 +/- 0.00 | 1.00 +/- 0.03 | 0.20 +/- 0.01 | 135.04 +/- 3.63 | 37.22 +/- 2.75 | 76.94 +/- 1.96 | 0.00 +/- 0.00 |
| route-diverse-e4r | gaf | 100.00 +/- 0.00 | 1.13 +/- 0.18 | 0.25 +/- 0.01 | 138.19 +/- 24.55 | 63.33 +/- 1.18 | 72.78 +/- 3.42 | 0.00 +/- 0.00 |
| route-diverse-e4r | tcn | 100.00 +/- 0.00 | 1.12 +/- 0.14 | 0.25 +/- 0.00 | 164.30 +/- 4.24 | 56.67 +/- 2.04 | 91.94 +/- 4.16 | 0.00 +/- 0.00 |
| route-diverse-e4r | shuffled_gaf | 100.00 +/- 0.00 | 1.21 +/- 0.10 | 0.23 +/- 0.01 | 152.06 +/- 11.73 | 31.39 +/- 2.75 | 75.56 +/- 6.17 | 0.00 +/- 0.00 |
| empirical-motion-e3b | all_candidates | 100.00 +/- 0.00 | 1.18 +/- 0.18 | 0.25 +/- 0.01 | 152.72 +/- 14.91 | 45.14 +/- 1.70 | 70.45 +/- 2.14 | 0.31 +/- 0.00 |
| empirical-motion-e3b | random | 100.00 +/- 0.00 | 1.43 +/- 0.15 | 0.23 +/- 0.01 | 189.26 +/- 22.94 | 46.67 +/- 5.31 | 74.44 +/- 3.07 | 0.00 +/- 0.00 |
| empirical-motion-e3b | gaf | 100.00 +/- 0.00 | 1.55 +/- 0.22 | 0.29 +/- 0.04 | 190.68 +/- 8.83 | 75.00 +/- 5.44 | 71.39 +/- 2.75 | 0.00 +/- 0.00 |
| empirical-motion-e3b | tcn | 100.00 +/- 0.00 | 1.66 +/- 0.11 | 0.25 +/- 0.02 | 200.52 +/- 16.67 | 51.67 +/- 4.14 | 85.56 +/- 3.87 | 0.00 +/- 0.00 |
| empirical-motion-e3b | shuffled_gaf | 100.00 +/- 0.00 | 1.61 +/- 0.32 | 0.25 +/- 0.02 | 201.73 +/- 48.64 | 40.56 +/- 4.83 | 67.78 +/- 5.11 | 0.00 +/- 0.00 |

## Interpretation limits

The empirical baseline deliberately reuses complete training motion profiles; geometry is new but motion is memorized. It is a strong no-training baseline, not an independent simulator. Diffusion needs evidence beyond topology, which both methods inherit from the same path decoder. Absolute-coordinate nearest-training distances cannot alone detect copied speed profiles.

Only 425 strict moving training tracks and 33 evaluation conditions are represented. All retained candidates and fallback counts are reported, including caps and invalid trajectories; no failed samples are silently replaced. This is a post-E5 validation comparison, not a final test or downstream augmentation experiment.

## Development decision

On the all-candidate comparison, learned diffusion has lower point estimates for speed, acceleration and travel W1 (1.02/0.22/145.73 versus 1.18/0.25/152.72). This is not a significance claim. Empirical candidate-pair distances are 180-193m, versus approximately 147m for diffusion; the evidence does not show uniform dominance or better diversity for diffusion. Each empirical seed has one route cap (3/792 pooled); table cap percentages are class-macro averages rather than sample-pooled rates.

Keep the negative downstream and cross-family GAF findings. Before a final locked test, add paired vehicle uncertainty and motion-domain nearest-training diagnostics using these existing candidates. Do not initiate another classifier search or proxy-gradient guidance on these results.
