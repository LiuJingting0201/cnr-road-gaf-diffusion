# Turin bus batch-size comparison

Both fits use 200 epochs and the same 2,555 training journeys. The smaller batch makes 9,200 optimizer updates versus 600 for the original fit.

## Development requests

| Seed | Global batch | Requests | Density JS | OD JS | Length JS | Pattern F1 | Observed / generated median km | Length W1 km | Outside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1024 | 256 | 0.0828 | 0.6459 | 0.1823 | 0.80 | 12.29 / 15.17 | 8.69 | 2.13% |
| 42 | 56 | 256 | 0.0201 | 0.3496 | 0.0712 | 0.80 | 12.29 / 11.47 | 0.70 | 0.44% |
| 43 | 1024 | 256 | 0.4966 | 0.6931 | 0.4114 | 0.25 | 12.29 / 49.81 | 99.50 | 71.29% |
| 43 | 56 | 256 | 0.0262 | 0.3843 | 0.0698 | 0.75 | 12.29 / 11.69 | 0.65 | 0.77% |

## Reserved requests

| Seed | Global batch | Requests | Density JS | OD JS | Length JS | Pattern F1 | Observed / generated median km | Length W1 km | Outside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1024 | 341 | 0.0607 | 0.6513 | 0.1387 | 0.85 | 11.13 / 13.48 | 8.86 | 0.70% |
| 42 | 56 | 341 | 0.0120 | 0.3187 | 0.0515 | 0.95 | 11.13 / 10.01 | 0.54 | 0.28% |
| 43 | 1024 | 341 | 0.4931 | 0.6889 | 0.3718 | 0.15 | 11.13 / 45.76 | 95.80 | 70.41% |
| 43 | 56 | 341 | 0.0171 | 0.3375 | 0.0611 | 0.95 | 11.13 / 10.34 | 0.51 | 0.36% |
