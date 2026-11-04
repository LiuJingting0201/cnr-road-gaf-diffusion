# E1 inherited Hilbert GAF with a self-trained ResNet18

Full-vehicle participant classification diagnostic. Fixed epoch 50 is primary. No test evaluation; not the supplied group CNN or a reproduction of unavailable journal details. No claim of short-trajectory guidance effectiveness.

Status: all 12 runs complete.

| Grid and initialization | Completed seeds | Accuracy % | Macro-F1 % |
|---|---:|---:|---:|
| grid32-random | 3/3 | 51.80 +/- 2.74 | 38.81 +/- 5.17 |
| grid32-imagenet | 3/3 | 72.74 +/- 3.90 | 58.80 +/- 3.36 |
| grid64-random | 3/3 | 62.95 +/- 1.15 | 44.44 +/- 2.40 |
| grid64-imagenet | 3/3 | 75.71 +/- 1.23 | 60.09 +/- 0.92 |

The next gate is duration-matched validation and candidate-image distinguishability, before original-GAF selection of generated trajectories. Final synthesis evaluation must use independent models and downstream real held-out data.


## Interpretation and next decision

The selected grid64 pretrained model has mean per-class F1 of pedestrian 85.46%, bus 83.65%, motorcycle 49.28%, private car 83.18%, bicycle 43.36%, and taxi/uber 15.63%. Thus the 75.71% overall accuracy does not mean reliable six-class feedback; rare and confusable classes remain weak. ECE is 0.183, so raw confidence is not a calibrated probability of correctness. Three 50-epoch runs took 58.6/58.8/58.5 seconds of measured training/evaluation each, excluding preprocessing and initial weight download.

Freeze grid64 + pretrained ResNet18 for E2 rather than expanding the architecture search. E2 checks full-trajectory transfer versus duration-matched training on 64/224/512-point real continuous segments, retaining class coverage and a common-validation-ID view. These diagnostics determine whether original-GAF feedback is informative at the generator's time scale; they do not establish synthetic-data utility. The next generation experiment must compare equal candidate pools and independent evaluation, and must not hide taxi or bicycle failures.
