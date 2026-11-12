# Does trip-average speed mask Beijing vehicle-class differences?

Each signed effect is bus minus bicycle. For generated trajectories, the request and random draws are identical on both sides. Real trajectories use 94 disjoint observational pairs. Each pair contributes its two headers to the generated comparison. All variants share the same class-only architecture, data and fit budget. Changes in the no-mean and no-distance-cues rows include retraining from the corresponding seed.

| Header and seed | Mean speed (m/s) | P90 speed (m/s) | Slow share (pp) | Absolute acceleration (m/s²) | Slow entries / 10 min |
|---|---:|---:|---:|---:|---:|
| Observed | +1.351 | +4.974 | +14.728 | +0.024 | +3.951 |
| full-42 | +0.255 | +0.362 | -0.107 | +0.085 | -0.037 |
| full-43 | -0.221 | -0.130 | -0.019 | +0.096 | +0.002 |
| no_mean-42 | -0.977 | -1.725 | -0.045 | -0.229 | -0.012 |
| no_mean-43 | -4.351 | -7.635 | +0.003 | -1.029 | -0.005 |
| no_distance_cues-42 | -1.232 | -2.321 | -0.078 | -0.224 | -0.012 |
| no_distance_cues-43 | -1.319 | -2.176 | +0.024 | -0.124 | +0.003 |

A slow entry starts after a non-slow interval and requires two consecutive intervals below 1 m/s. Initial slow periods do not count as entries. This metric depends on the observation clock; a low-speed interval does not establish a full stop.

## Actual-label speed distribution error

Lower W1 is better. These values use each original bike or bus request with its observed class, independent of the switch contrast.

| Variant | Bike W1 (m/s) | Bus W1 (m/s) |
|---|---:|---:|
| full-42 | 17.76 | 24.27 |
| full-43 | 171.12 | 202.03 |
| no_mean-42 | 54.21 | 70.35 |
| no_mean-43 | 126.49 | 142.42 |
| no_distance_cues-42 | 32.78 | 24.80 |
| no_distance_cues-43 | 119.08 | 138.88 |

## Additional observed segments

| Class | Previous | Added | Expanded |
|---|---:|---:|---:|
| bike | 1486 | 242 | 1728 |
| bus | 1959 | 357 | 2316 |
| car | 699 | 130 | 829 |
| taxi | 338 | 53 | 391 |

The separate pack includes 4,729 training and 535 validation segments across 44 participants (originally 34). The added 782 spans contain 60–119 native observations; 169 recording files were previously unrepresented. The expanded pack was kept outside this ablation. [Protocol](PROTOCOL.md)
