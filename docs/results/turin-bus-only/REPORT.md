# Conditional DiffTraj on observed Turin bus journeys

The single-day input contained 3,262 terminal-covered bus journeys with consistent fleet labels. Vehicle-disjoint partitions contain 2,555 training, 366 development and 341 reserved journeys. Both 200-epoch fits use the same conditional architecture and generation requests. No class input is used. The score is distributional and the trip header includes distance and OD cells.

| Sample | Point density JS ↓ | OD JS ↓ | Length JS ↓ | Top-20 F1 ↑ |
|---|---:|---:|---:|---:|
| Turin, seed 42 | 0.0607 | 0.6513 | 0.1387 | 0.8500 |
| Turin, seed 43 | 0.4931 | 0.6889 | 0.3718 | 0.1500 |
| Turin, real development vs real reserve | 0.0587 | 0.3136 | 0.0895 | 0.7500 |
| Porto, long trips, equal-sized subsets, seed 42 | 0.0222 | 0.0124 | 0.0157 | 0.7500 |
| Porto, long trips, equal-sized subsets, seed 43 | 0.0176 | 0.0126 | 0.0121 | 0.7000 |

Porto rows are medians over 50 paired subsets of 341 old requests sampled from the 3,000 saved evaluations; the aggregate JSON reports the 10th and 90th percentiles. The real/real row uses 341 actual trips on each side. The absolute number of optimizer updates differs because the training sets differ in size. Turin reports one service day and a coarser observation clock.

| Seed | Selected epoch | Real median (km) | Generated median (km) | Length W1 (km) | Points outside training bounds |
|---|---:|---:|---:|---:|---:|
| 42 | 194 | 11.13 | 13.48 | 8.86 | 0.70% |
| 43 | 197 | 11.13 | 45.76 | 95.80 | 70.41% |
