# Long trajectory diffusion and original GAF candidate selection

Three diffusion seeds, 20,000 updates each, 512 points (127.75 s), five moving non-pedestrian classes. 425 training / 57 available validation tracks; generation uses 33 selected validation conditions and K=8 draws each. Planned validation paths use only start, class and the training-only transition prior. Test untouched.

All selection methods use the same candidate pool and shared validity mask. Teacher seed42; auditors seeds43/44. GAF uses inherited Hilbert/squeeze/GASF/GADF/MTF, with duration-matched self-trained ResNet18. The raw trajectory TCN uses the same real training cohort and 50 epochs. Different network families and ImageNet initialization are disclosed, not claimed to be identical architectures.

| Method | Graph consistency % | GAF audit % | Raw TCN audit % | Speed W1 m/s | Acceleration W1 m/s2 | Route cap % |
|---|---:|---:|---:|---:|---:|---:|
| all_candidates | 100.00 +/- 0.00 | 41.94 +/- 1.38 | 77.26 +/- 0.35 | 1.12 +/- 0.04 | 0.22 +/- 0.01 | 0.00 +/- 0.00 |
| random | 100.00 +/- 0.00 | 37.22 +/- 5.11 | 79.44 +/- 1.57 | 1.24 +/- 0.09 | 0.21 +/- 0.02 | 0.00 +/- 0.00 |
| gaf | 100.00 +/- 0.00 | 41.39 +/- 4.16 | 76.94 +/- 1.57 | 1.32 +/- 0.11 | 0.21 +/- 0.02 | 0.00 +/- 0.00 |
| tcn | 100.00 +/- 0.00 | 38.06 +/- 5.29 | 81.11 +/- 2.39 | 1.23 +/- 0.10 | 0.21 +/- 0.02 | 0.00 +/- 0.00 |
| shuffled_gaf | 100.00 +/- 0.00 | 38.89 +/- 6.17 | 78.61 +/- 2.39 | 1.28 +/- 0.08 | 0.21 +/- 0.02 | 0.00 +/- 0.00 |

## Representation and coverage diagnostics

- Seed 42: valid candidates 0.159; identical GAF pools 0.000; mean distinct images per eight draws 8.00; candidate-pair distance 61.78 m; encoding failures 0.
- Seed 43: valid candidates 0.155; identical GAF pools 0.000; mean distinct images per eight draws 8.00; candidate-pair distance 66.98 m; encoding failures 0.
- Seed 44: valid candidates 0.155; identical GAF pools 0.000; mean distinct images per eight draws 8.00; candidate-pair distance 66.05 m; encoding failures 0.

## Interpretation limits

Road consistency belongs mainly to the path decoder, not to GAF. Same-family auditor agreement can favor its selector; inspect both families and motion distributions. Real-reference auditor accuracy is recorded in summary.json and each seed report to expose evaluator limitations. A high classifier score is not synthetic-data utility.

Taxi has only three evaluation conditions, motorcycle six. Seeds share these conditions and are not independent vehicles; mean/std is descriptive, not a statistical significance test. Full-trajectory and short-window historical experiments are different populations. Each condition produces one selected sample, so retained within-condition diversity is not established; only original candidate-pool diversity is reported.

Next decision must combine independent class agreement, motion realism, selection fallback and image distinguishability before downstream augmentation. Preserve negative findings and assess broader route coverage; do not expand classifier architecture search.


## Audit search-limit issue discovered in this round

The low shared-validity fraction is dominated by the lane auditor's default limit of 12 candidate lane positions per point, not by failed geometry or route-end caps. Most conditions therefore fall back to the same random candidate (seed42: 26/33); this greatly limits the amount of actual semantic selection. Finding a consistent path among retained candidates still proves existence within the audit envelope, but the preregistered conservative mask rejects any truncation warning. Keep these results as the original strict run.

A separate post-hoc sensitivity run expands only the lane candidate search limit to 64. The generated trajectories, network weights, 5m distance envelope, class conditions and common selection rule are unchanged. It must be reported separately and cannot retroactively replace this run. Current numbers do not demonstrate a robust cross-family benefit of GAF selection: its same-family audit rises, while raw-TCN audit and speed distribution do not improve.
