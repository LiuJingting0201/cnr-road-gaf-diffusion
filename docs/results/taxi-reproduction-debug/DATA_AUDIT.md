# Audit of the 10,000-taxi input pipeline

The audit reads 12,000 vehicle trajectories from their episode archives, departure/arrival records, packed native observations and prepared DiffTraj arrays. It independently reconstructs the five continuous conditions, two endpoint cells, departure bin, training normalization and 200-point paths. No fit, draw selection or evaluation parameter was changed by this inspection.

| Check | Result |
|---|---:|
| Train / development / reserved | 10,000 / 1,000 / 1,000 |
| Episodes used by selected IDs | 148 / 15 / 14, pairwise disjoint |
| Taxi records below 120 native samples, incomplete or discontinuous journeys | 0 |
| Exact duplicate 200-point coordinate sequences | 0 |
| Maximum discrepancy between archived raw component and packed coordinate, metres | 0.00047 |
| Maximum absolute discrepancy on independent 200-point interpolation, metres | 0.00049 |
| Maximum error after independently recomputing the continuous header, standardized units | 0.000033 |
| Endpoint ID discrepancies against independent bin-edge calculation | 0 of 24,000 |
| Stored versus recomputed training coordinate mean and standard deviation | identical |
| Median path length retained after 200-point interpolation and restoration to native sample count, reserved | 98.53% |

The original simulation is in local metric x/y coordinates with one-second steps. The representation requires at least 120 observations before interpolation, so 120 points span 119 seconds. Porto's 120 15-second observations span 1,785 seconds. The per-city datasets therefore describe different trip populations. Date/clock filtering and length matching require separate decisions from the original point-count rule.

## Artificial split-specific departure times

The original generator set `clock_start = (episode_seed % 1440) * 60 + local_departure_seconds`. The packer reproduced this value. Episode seeds for train, development and test start near 410000, 420000 and 430000, respectively. Their offsets place the three splits in entirely different five-minute departure bins. Original and repeated simulations share the same local departure schedule; the inter-split time-of-day gap is an artifact of the synthetic offset.

| Split | Original bins | Covered by training | Local simulation clock bins | Covered by training |
|---|---|---:|---|---:|
| Train | 208–240 | 100% | 0, 1, 2 | 100% |
| Development | 192–197 | **0%** | 0, 1, 2 | 100% |
| Reserved | 176–181 | **0%** | 0, 1, 2 | 100% |

DiffTraj looks up departure time by discrete embedding. The condition rows selected by these held-out bins received no examples during training. This is a concrete defect in the training population construction. It does not by itself quantify the fraction of generation error caused by clock conditioning. Global UTC is only a numerical convention for this artificial header; neither clock version is authentic time-of-day demand. For a standard baseline remove the episode offset and use the actual simulated departure seconds on all partitions.

## OD support

The local full grid has 16 × 16 cells, approximately 623.5 × 392.6 metres. The 10,000 training journeys cover 3,698 distinct directed OD cells pairs. Development and reserved each cover 817 distinct pairs in 1,000 trips; 86.0% and 86.7% of their individual trips have a joint OD seen in training. All individual endpoint cell IDs occur in training. The most frequent training pair accounts for 0.31% of trips. Sparse joint distributions and training support affect sampling error even with correctly encoded endpoints.

[Independent numerical audit](DATA_AUDIT.json). Reproduce by running `taxi_preprocessing_forensics.py /workspace/cnr-experiments` with NumPy and access to the private episode/fit files.

### How much additional support does 20k buy?

Holding the 10k grid fixed, 86.0% of the 1,000 development requests have their exact OD pair in the 10k training set, 68.4% have at least two examples, 37.0% at least five and 12.5% at least ten. The nested 20k cohort raises these fractions to 92.7%, 84.9%, 60.8% and 34.2%. Reconstructing the expanded pack's unstandardized endpoints produced exactly the same OD IDs for all 10,000 shared training journeys. Training frequency is a measure of data support; prediction improvement must be evaluated separately. [Measured coverage](DEV_SUPPORT.json).
