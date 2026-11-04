# Long trajectory diffusion and original GAF candidate selection

Three diffusion seeds, 20,000 updates each, 512 points (127.75 s), five moving non-pedestrian classes. 425 training / 57 available validation tracks; generation uses 33 selected validation conditions and K=8 draws each. Planned validation paths use only start, class and the training-only transition prior. Test untouched.

All selection methods use the same candidate pool and shared validity mask. Teacher seed42; auditors seeds43/44. GAF uses inherited Hilbert/squeeze/GASF/GADF/MTF, with duration-matched self-trained ResNet18. The raw trajectory TCN uses the same real training cohort and 50 epochs. Different network families and ImageNet initialization are disclosed, not claimed to be identical architectures.

| Method | Graph consistency % | GAF audit % | Raw TCN audit % | Speed W1 m/s | Acceleration W1 m/s2 | Route cap % |
|---|---:|---:|---:|---:|---:|---:|
| all_candidates | 100.00 +/- 0.00 | 41.94 +/- 1.38 | 77.26 +/- 0.35 | 1.12 +/- 0.04 | 0.22 +/- 0.01 | 0.00 +/- 0.00 |
| random | 100.00 +/- 0.00 | 37.22 +/- 6.43 | 77.78 +/- 1.42 | 1.21 +/- 0.08 | 0.21 +/- 0.02 | 0.00 +/- 0.00 |
| gaf | 100.00 +/- 0.00 | 55.56 +/- 3.87 | 75.00 +/- 1.18 | 1.26 +/- 0.05 | 0.24 +/- 0.01 | 0.00 +/- 0.00 |
| tcn | 100.00 +/- 0.00 | 48.06 +/- 2.19 | 81.94 +/- 1.57 | 1.20 +/- 0.09 | 0.22 +/- 0.01 | 0.00 +/- 0.00 |
| shuffled_gaf | 100.00 +/- 0.00 | 45.83 +/- 3.54 | 78.61 +/- 2.39 | 1.06 +/- 0.08 | 0.23 +/- 0.02 | 0.00 +/- 0.00 |

## Representation and coverage diagnostics

Lane candidate search limit per point: 64. Expanded-limit runs are post-hoc sensitivity checks on the same saved candidates, not new training or resampling.
- Seed 42: valid candidates 1.000; identical GAF pools 0.000; mean distinct images per eight draws 8.00; candidate-pair distance 61.78 m; encoding failures 0.
- Seed 43: valid candidates 1.000; identical GAF pools 0.000; mean distinct images per eight draws 8.00; candidate-pair distance 66.98 m; encoding failures 0.
- Seed 44: valid candidates 1.000; identical GAF pools 0.000; mean distinct images per eight draws 8.00; candidate-pair distance 66.05 m; encoding failures 0.

## Interpretation limits

Road consistency belongs mainly to the path decoder, not to GAF. Same-family auditor agreement can favor its selector; inspect both families and motion distributions. Real-reference auditor accuracy is recorded in summary.json and each seed report to expose evaluator limitations. A high classifier score is not synthetic-data utility.

Taxi has only three evaluation conditions, motorcycle six. Seeds share these conditions and are not independent vehicles; mean/std is descriptive, not a statistical significance test. Full-trajectory and short-window historical experiments are different populations. Each condition produces one selected sample, so retained within-condition diversity is not established; only original candidate-pool diversity is reported.

Next decision must combine independent class agreement, motion realism, selection fallback and image distinguishability before downstream augmentation. Preserve negative findings and assess broader route coverage; do not expand classifier architecture search.


## Decision after the completed sensitivity check

All candidates are eligible after increasing the search capacity; every variant has zero fallback conditions. The warning was a computational search limit, not a demonstrated road violation. The unchanged candidate pools have 100% graph consistency within the 5m audit envelope and zero route caps.

GAF selection increases its independent-seed same-family agreement from 37.22% to 55.56%, but raw-trajectory audit decreases from 77.78% to 75.00% and speed W1 rises from 1.21 to 1.26 m/s. This is not robust evidence of useful data synthesis or superiority over plain trajectory feedback. Raw TCN selection reaches 81.94% on its own family and 48.06% on GAF audit; same-family preferences must not be mistaken for ground truth.

Proceed to a bounded downstream augmentation pilot with real-only, class-matched repeats, simple perturbations, unguided synthetic, original-GAF-selected synthetic and TCN-selected synthetic groups. Do not start teacher-gradient guidance or another classifier architecture search on these results. Use training-only generation conditions, fixed synthetic counts and equal downstream optimization budgets; no validation conditions may enter augmentation training. Report that the current generator covers only 425 strict training tracks and five classes, while the real downstream task has six classes.
