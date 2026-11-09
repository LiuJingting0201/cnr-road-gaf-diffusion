# G39 completed independent augmentation stage

All26 fixed fits are complete. Frozen historical tests were not used. This stage uses a separately trained raw-series CNN; no GAF input or teacher weights enter its evaluation.

| Regime | F1 seed3901 (%) | F1 seed3902 (%) | Mean (%) | Nonbus parent-equal mean (%) |
|---|---:|---:|---:|---:|
| real | 89.52 | 92.97 | 91.25 | 89.71 |
| only-empirical | 90.30 | 89.07 | 89.69 | 87.71 |
| only-B | 85.46 | 85.93 | 85.70 | 84.09 |
| only-S | 85.44 | 85.96 | 85.70 | 84.23 |
| only-V | 84.87 | 87.78 | 86.33 | 84.72 |
| only-SV | 84.82 | 86.25 | 85.54 | 83.87 |
| only-RAW | 85.38 | 86.90 | 86.14 | 84.71 |
| plus-empirical | 91.23 | 90.48 | 90.85 | 89.21 |
| plus-B | 92.27 | 94.07 | 93.17 | 91.94 |
| plus-S | 91.98 | 94.62 | 93.30 | 91.95 |
| plus-V | 93.57 | 94.63 | 94.10 | 93.02 |
| plus-SV | 93.31 | 93.83 | 93.57 | 92.36 |
| plus-RAW | 93.53 | 94.59 | 94.06 | 92.96 |

`real`:1000 balanced real training parents; `empirical`:1000 resampled rows from that same base; each synthetic source:1000 matched unscored candidate identities. `only` trains on that source alone; `plus` adds it to the real base. All regimes receive2000 updates, final checkpoint, two downstream seeds.

The base diffusion augmentation improves mean macroF1 by1.92 percentage points versus real-only; empirical duplication does not provide that gain. This supports synthetic-data utility on these held-out conditions. Synthetic-only remains below real-only.

Speed-GAF augmentation adds0.93 points over base augmentation, but raw-feature augmentation is numerically very close (94.10% versus 94.06%). This does not establish GAF-specific superiority over an alternative learned representation. The dual-GAF gain over base is smaller and not consistent in sign across the two seeds.

Evaluation uses two held-out OD cohorts repeated across windows, with common bus service templates and the same map. Do not turn two seeds into a population significance claim. Nonbus parent-equal results and every class/episode confusion are retained. Additional synthetic public OD conditions are part of the augmentation; this is not an isolated test of fine-scale dynamics.

Physical-quality results remain unchanged: the G39 main comparison does not show an overall native speed/acceleration improvement. Utility is a meaningful separate success criterion for data synthesis, not permission to describe all generated motion as better. Factorial mechanism review is still pending at this stage.

Sources: complete.json contains every fit and per-episode/per-class outcome; verification.json records independent artifact and prediction-order/F1 checks.

Independent verification recomputed all26 pooled/per-class F1 values and confirmed identical prediction ID/label/episode order for384 held-out reference journeys. This includes all admissible references; main physical pairing additionally requires candidate-catalog support, so its paired count is different.
