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
