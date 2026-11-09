# G40 direct synthesis results

J-SV does not improve both aggregate primary endpoints; no broad physical-quality improvement claim.

Primary: first direct draw, no terminal GAF ranking. Lower W1 is better.

| Arm | Speed W1 | Acceleration W1 |
|---|---:|---:|
| B | 1.437940 | 0.176001 |
| F-SV | 1.435433 | 0.176107 |
| J-S | 1.439085 | 0.176944 |
| J-V | 1.437401 | 0.179206 |
| J-SV | 1.440104 | 0.180289 |
| J-RAW | 1.443677 | 0.180682 |
| J-SH | 1.434486 | 0.174503 |
| C | 1.438504 | 0.175781 |

All classes, scenes, seeds and secondary metrics are retained in `all-class-metrics.csv` and `complete.json`.
The classifiers are self-trained; confidence alone is not a quality result. RAW differs in architecture and pretraining.
No unseen-city, real-world or universal GAF claim follows from this simulator study.
