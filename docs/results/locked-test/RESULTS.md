# Frozen SUMO test results

All 5,000 held-out journeys are included. Each class contributes 1,000 requests. The controls share noise and request headers; a combined draw reuses the coordinate draw as its path.

| Measure | Coordinate 42 | Combined 42 | Coordinate 43 | Combined 43 |
|---|---:|---:|---:|---:|
| Speed W1 (m/s) | 1.2041 | 0.7753 | 1.0859 | 0.6349 |
| Acceleration W1 (m/s²) | 0.1783 | 0.0436 | 0.1888 | 0.0435 |
| Destination cell hit | 72.28% | 72.28% | 71.74% | 71.74% |
| Density JS | 0.0063 | 0.0068 | 0.0069 | 0.0073 |
| OD JS | 0.2099 | 0.2099 | 0.2212 | 0.2212 |
| Length JS | 0.0256 | 0.0382 | 0.0233 | 0.0322 |
| Top-20 hotspot F1 | 0.960 | 0.960 | 0.930 | 0.930 |

## Speed distribution by class

| Vehicle | Coordinate 42 | Combined 42 | Coordinate 43 | Combined 43 |
|---|---:|---:|---:|---:|
| bicycle | 0.665 | 0.485 | 0.662 | 0.389 |
| bus | 1.548 | 0.553 | 1.355 | 0.510 |
| motorcycle | 1.187 | 0.957 | 1.049 | 0.754 |
| passenger | 1.237 | 0.920 | 1.097 | 0.726 |
| taxi | 1.383 | 0.962 | 1.266 | 0.796 |

## Bus minus bicycle

117 disjoint pairs of observed trips met the established calipers, including at most 0.50 m/s difference in requested mean speed. For each member of each pair, the generated path, header and noise are held fixed while the speed generator receives the two different vehicle labels. Differences are averaged over both members.

| Measure | Matched SUMO | Combined 42 | Combined 43 |
|---|---:|---:|---:|
| slow_time_percent | +40.003 | +24.623 | +24.520 |
| slow_entries_per_10min | +7.990 | +4.917 | +4.818 |
| p90_speed_mps | +7.379 | +4.485 | +4.384 |
| mean_abs_acceleration_mps2 | +0.360 | +0.207 | +0.202 |

Both seeds used the predeclared checkpoint hashes. The frozen test episodes are disjoint from training and validation. Endpoint coordinates were checked for exact equality on every combined and coordinate draw. Endpoint retention follows from retiming the same polyline.

[Lock](PROTOCOL.md) · [Seed 42 maps](map-42.png) · [Seed 43 maps](map-43.png) · [Seed 42 label switch](swap-42.png) · [Seed 43 label switch](swap-43.png)
