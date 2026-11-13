# DiffTraj on the original SUMO vehicle export

1,478 train, 198 development, 177 reserved vehicles. All four distributional metrics use training-defined grids and bins.

| Sample | Density JS | OD JS | Length JS | Pattern F1 | Median observed/generated km | Length W1 km | Outside |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.5918 | 0.6760 | 0.6058 | 0.10 | 5.86 / 224.62 | 372.06 | 93.54% |
| 43 | 0.6343 | 0.6931 | 0.6097 | 0.00 | 5.86 / 429.54 | 697.01 | 98.14% |
| Real/real | 0.0224 | 0.5144 | 0.0295 | 0.9000 | — | — | — |

One observed track per vehicle. At batch 1,024 the model received 400 updates. Both fits score the final epoch on the same requests. [Protocol](PROTOCOL.md)
