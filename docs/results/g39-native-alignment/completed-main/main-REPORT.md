# G39 direct synthesis results

SV does not improve both aggregate primary endpoints; no broad physical-quality improvement claim.

Primary: first direct draw, no terminal GAF ranking. Lower W1 is better.

| Arm | Speed W1 | Acceleration W1 |
|---|---:|---:|
| B | 1.283834 | 0.186744 |
| S | 1.297086 | 0.188423 |
| V | 1.294724 | 0.185917 |
| SV | 1.293373 | 0.187615 |
| P | 1.287757 | 0.187902 |
| SH | 1.288549 | 0.185868 |
| RAW | 1.294157 | 0.188364 |

All classes, scenes, seeds and secondary metrics are retained in `all-class-metrics.csv` and `complete.json`.
The classifiers are self-trained; confidence alone is not a quality result. RAW differs in architecture and pretraining.
No unseen-city, real-world or universal GAF claim follows from this simulator study.
