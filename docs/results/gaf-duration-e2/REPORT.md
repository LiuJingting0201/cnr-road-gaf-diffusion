# E2 duration-matched inherited Hilbert GAF diagnostic

64x64 Hilbert grid, unchanged squeeze/mirror to 224 RGB. One centered window of longest continuous segment per ID; all three lengths share a common center, up to half a sample. No test access. New ResNet18 models, fixed epoch 50. Not generated-data evaluation.

Common validation cohort: 164 vehicles. Common-cohort evaluation controls validation identity, but training eligibility still differs with duration; duration and training population effects cannot be fully separated.

| Points | Model training domain | All eligible accuracy % | All eligible macro-F1 % | Common-ID macro-F1 % |
|---|---|---:|---:|---:|
| 64 | full_vehicle | 26.78 +/- 0.34 | 11.39 +/- 1.25 | 5.28 +/- 0.86 |
| 64 | duration_matched | 46.30 +/- 0.52 | 28.63 +/- 1.23 | 21.66 +/- 1.32 |
| 224 | full_vehicle | 23.23 +/- 2.06 | 13.61 +/- 1.56 | 10.82 +/- 1.47 |
| 224 | duration_matched | 58.25 +/- 2.12 | 42.37 +/- 0.94 | 38.81 +/- 0.48 |
| 512 | full_vehicle | 31.30 +/- 5.01 | 18.26 +/- 1.69 | 18.26 +/- 1.69 |
| 512 | duration_matched | 66.67 +/- 1.44 | 59.12 +/- 3.51 | 59.12 +/- 3.51 |

Next: select a generation horizon considering road reconstruction coverage, class coverage, diversity and feedback information. Do not pick by CNN accuracy alone. Keep the existing diffusion baseline; evaluate GAF selection on identical candidate pools against random and trajectory-classifier selection.
