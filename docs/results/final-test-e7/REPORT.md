# Locked final held-out-ID evaluation

Test IDs: 270; encodable 512-point windows: 182; strict moving requests: 53; complete K=8 planned pools: 53; unserved conditions: 0. No missing pools are replaced with real future routes.

Generation class counts before planning: {'pedestrian': 0, 'bus': 7, 'motorcycle': 4, 'private_car': 33, 'bycicle': 7, 'taxi_or_uber': 2}. Missing served classes: []. Macro generation values average supported classes only. All eligible test conditions are used, unlike the 33-condition validation pilot.

| Generator | Selection | Graph % | Speed W1 | Acceleration W1 | Travel W1 | GAF audit % | TCN audit % | Cap % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion | all_candidates | 100.000 | 1.244 | 0.269 | 177.414 | 47.477 | 77.683 | 0.000 |
| diffusion | random | 100.000 | 1.402 | 0.280 | 206.004 | 51.854 | 89.863 | 0.000 |
| diffusion | gaf | 100.000 | 1.410 | 0.286 | 191.519 | 70.455 | 78.102 | 0.000 |
| diffusion | tcn | 100.000 | 1.444 | 0.291 | 209.295 | 47.316 | 93.182 | 0.000 |
| diffusion | shuffled_gaf | 100.000 | 1.501 | 0.279 | 195.798 | 39.553 | 76.948 | 0.000 |
| empirical | all_candidates | 100.000 | 1.446 | 0.239 | 201.119 | 40.472 | 71.070 | 0.000 |
| empirical | random | 100.000 | 1.692 | 0.257 | 233.887 | 45.613 | 78.175 | 0.000 |
| empirical | gaf | 100.000 | 1.887 | 0.284 | 252.549 | 63.110 | 68.413 | 0.000 |
| empirical | tcn | 100.000 | 2.281 | 0.308 | 282.142 | 42.720 | 91.364 | 0.000 |
| empirical | shuffled_gaf | 100.000 | 1.488 | 0.267 | 195.018 | 35.411 | 63.045 | 0.000 |

## Paired vehicle uncertainty

Positive empirical-minus-diffusion W1 favors diffusion. Fixed models, routes and pools; 2,000 paired class-stratified vehicle resamples, not training variability or multiplicity-adjusted inference.

| Metric | Difference [95% interval] |
|---|---:|
| speed_w1_mps | 0.2027 [-0.0181, 0.9926] |
| acceleration_w1_mps2 | -0.0304 [-0.0805, 0.0460] |
| travel_w1_m | 23.7055 [7.8574, 132.9121] |

## Downstream synthetic-data utility

Same 36 fixed E5 learners, six classes, no test-driven retraining. Macro-F1 is averaged across the three learner seeds. Intervals are conditional on one augmentation generator and pool.

| Learner | Group | Macro-F1 % | GAF minus this group pp [95% interval] |
|---|---|---:|---:|
| tcn | real_only | 76.92 | -5.23 [-9.01, -1.65] |
| tcn | repeat | 72.77 | -1.09 [-4.02, 1.93] |
| tcn | timewarp | 73.32 | -1.63 [-4.42, 1.33] |
| tcn | synthetic_random | 70.29 | 1.40 [-1.54, 5.16] |
| tcn | synthetic_gaf | 71.69 | 0.00 [0.00, 0.00] |
| tcn | synthetic_tcn | 72.08 | -0.39 [-4.41, 4.26] |
| gru | real_only | 64.38 | -1.70 [-6.42, 3.13] |
| gru | repeat | 59.70 | 2.98 [0.28, 5.95] |
| gru | timewarp | 63.14 | -0.46 [-2.77, 2.16] |
| gru | synthetic_random | 60.82 | 1.86 [-1.70, 5.47] |
| gru | synthetic_gaf | 62.68 | 0.00 [0.00, 0.00] |
| gru | synthetic_tcn | 61.20 | 1.48 [-1.72, 4.82] |

## Auxiliary classifier audit on all encodable test windows

| Self-trained family | Accuracy % | Six-class macro-F1 % |
|---|---:|---:|
| gaf | 64.29 | 53.38 |
| tcn | 78.57 | 72.92 |

## Limits and interpretation

GAF classifiers are self-trained on inherited Hilbert/GASF/GADF/MTF, not supplied group CNNs. Same-family auditor agreement is biased toward its selector and cannot establish utility. Road consistency mostly follows from the common path decoder, not diffusion learning or GAF. Pedestrians are not synthesized; no traffic-light, collision, city-transfer or precise SUMO-version claim is made.

Full aligned motion nearest-training diagnostics are in summary.json. Near-copy tolerance is 0.001m mean increment difference, with empirical motion as a known-copy control; this does not detect every form of memorization. Larger candidate distance need not mean more realistic diversity. All failures and coverage restrictions remain in the report. Final tests do not authorize further result-driven model selection.

Fixed map examples are generated separately: seed42, first served condition per class, every method from the same pool.

## Figures and reusable artifacts

- [Chinese thesis summary and chapter outline](../../THESIS_RESULTS_SUMMARY.md).
- [Method diagram](method.png) ([PDF](method.pdf)).
- [Fixed map examples](map_examples.png) ([PDF](map_examples.pdf), [selection record](map_examples.json)). These use the first served condition per class, not hand-picked best cases.
- [Paired generation effects](paired_w1.png) ([PDF](paired_w1.pdf)).
- [All augmentation effects vs real-only](utility.png) ([PDF](utility.pdf)).
- [Training-only synthetic export catalog](training_synthetic_catalog.json): three alternative 425-trajectory CSV variants; exact synthetic extras from E5, not real training rows.
- [Locked-test synthetic export catalog](synthetic_catalog.json): 1,272 candidates and four overlapping selected variants. Do not reuse these test conditions as training inputs and claim an untouched final test.

Archives remain on the experiment server and local deliverable folder, outside Git. Each export is read back to verify five columns, finite coordinates, trajectory counts and 512 quarter-second timestamps; catalog hashes identify exact files. Coordinates are SUMO metres and time is relative within each trajectory. Separate trajectories do not constitute a jointly simulated traffic scene.
