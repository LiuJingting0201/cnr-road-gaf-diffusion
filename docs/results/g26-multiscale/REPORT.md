# G26 · Joint diffusion of block distances and local progress

Each of four arms trained from scratch on 6,033 complete trips with two seeds, 12,000 updates each. H jointly denoises duration, 16 coarse distance shares and 16 local shares per block. HM adds a six-way physical mode condition. HV and HG add direct speed-curve or speed angular-field supervision respectively. The original spatial Hilbert scorer is separate. Seeds 277001–277008 provided 384 new balanced OD/class requests, four common noises per model seed, 3,072 candidates per arm. No frozen test data were used.

| Arm | Longest steady interval W1 (s) | Duration W1 (s) | Four-second speed change W1 (m/s) |
|---|---:|---:|---:|
| G22 original | 12.43 | 14.82 | 0.253 |
| H | 26.83 | 15.16 | 0.395 |
| HM | 26.74 | 15.35 | 0.430 |
| HV | 26.85 | 15.37 | 0.429 |
| HG | 26.83 | 15.41 | 0.428 |

Cruise discrepancy for H rose by 14.40 seconds vs G22 (paired episode-bootstrap interval 11.97 to 16.43). On 336 supported requests, switching only the within-band mode under HG shortened the longest cruise by 19.12 seconds. Pedestrians lacked support for this contrast. In a subsequent diagnostic, the speed jump at a coarse block boundary was 2.78 times the interior jump for H versus 1.00 for SUMO. The seed-42 first-draw road audit passed 383/384 for each block arm, including a private-car failure, and 384/384 for G22. Pedestrians have a sidewalk proximity check only. Preserve this layout as a negative result. See G27 for a matched replication removing local normalization.
