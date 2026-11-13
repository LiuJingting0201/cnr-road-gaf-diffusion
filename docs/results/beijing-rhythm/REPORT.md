# Does training on stop–go rhythm improve vehicle label control?

All 535 held-out requests were drawn with both labels. The comparison below uses the two headers from each of 108 disjoint observed bus/bike pairs. Signed differences are bus minus bicycle. Both seeds use identical validation requests.

| Model | Slow share (pp) | Slow entries per 10 min | Mean speed (m/s) | P90 speed (m/s) |
|---|---:|---:|---:|---:|
| Observed pairs | +16.285 | +3.936 | +1.108 | +4.249 |
| C-42 | +0.007 | -0.005 | +0.783 | +1.361 |
| C-43 | +0.012 | +0.000 | -3.853 | -6.707 |
| CR-42 | +0.002 | -0.006 | +2.624 | +4.543 |
| CR-43 | +0.002 | -0.000 | +2.942 | +5.187 |

C is the class-conditioned baseline. CR adds soft slow-time and entry-rate supervision. Hard thresholds are used for evaluation. Entries represent sustained low speed on the 200-point clock, not confirmed stops at bus stops.

| Seed | Change in absolute slow-share error (pp) | Change in absolute entry-rate error (per 10 min) | Bike speed W1, C → CR (m/s) | Bus speed W1, C → CR (m/s) |
|---|---:|---:|---:|---:|
| 42 | +0.005 | +0.000 | 124.52 → 542.10 | 147.07 → 595.14 |
| 43 | +0.009 | +0.000 | 222.38 → 84.80 | 214.14 → 74.19 |

Negative error changes indicate a closer signed contrast. The original-label speed W1 uses all held-out requests for the indicated mode. See [protocol](PROTOCOL.md) and [full metrics](summary.json).
