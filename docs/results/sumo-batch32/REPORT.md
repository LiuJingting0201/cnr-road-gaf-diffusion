# SUMO batch-size comparison

Fixed training and sampling protocol; changed batch size and resulting update count. Scores are from the same 177 reserved vehicles.

| Seed | Batch | Density JS | OD JS | Length JS | Hotspot F1 | Generated median km | Length W1 km | Outside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1024 | 0.5918 | 0.6760 | 0.6058 | 0.10 | 224.62 | 372.06 | 93.54% |
| 42 | 32 | 0.1110 | 0.5597 | 0.2058 | 0.65 | 10.52 | 6.97 | 1.33% |
| 43 | 1024 | 0.6343 | 0.6931 | 0.6097 | 0.00 | 429.54 | 697.01 | 98.14% |
| 43 | 32 | 0.0690 | 0.5564 | 0.0536 | 0.80 | 5.66 | 0.49 | 0.73% |

Observed median 5.86 km. Real development versus real reserve: density, OD, length JS and hotspot F1 0.0224, 0.5144, 0.0295, 0.9000.
