# Original DiffTraj on SUMO taxi trips and Porto long trips

Each model receives the original eight trip attributes. Equal-sample Porto scores are medians across 50 paired subsets of 1,000 generated and observed trips. The local Porto full-population values use 3,000 requests. JS divergence is lower for closer distributions. Hotspot F1 increases with overlap.

| Data | Seed | Density JS ↓ | OD JS ↓ | Length JS ↓ | Hotspot F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Porto, published Table 5 | — | .0087 | .0132 | .0242 | .847 |
| Porto, all requests | 42 | 0.0136 | 0.0059 | 0.0064 | 0.7500 |
| Porto, equal sample size | 42 | 0.0176 | 0.0079 | 0.0084 | 0.7000 |
| SUMO, taxi only | 42 | 0.0139 | 0.3904 | 0.0252 | 0.9500 |
| Porto, all requests | 43 | 0.0094 | 0.0070 | 0.0039 | 0.8000 |
| Porto, equal sample size | 43 | 0.0138 | 0.0088 | 0.0058 | 0.7000 |
| SUMO, taxi only | 43 | 0.0251 | 0.4767 | 0.0281 | 0.9500 |
| SUMO observed vs observed | — | 0.0039 | 0.4870 | 0.0036 | 0.9000 |

| SUMO seed | Observed median km | Generated median km | Length W1 km | Outside training bounds |
|---|---:|---:|---:|---:|
| 42 | 5.64 | 4.84 | 0.82 | 0.03% |
| 43 | 5.64 | 6.18 | 0.80 | 0.20% |

[Data and training protocol](PROTOCOL.md)
