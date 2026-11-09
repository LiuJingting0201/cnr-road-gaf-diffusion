# G39 completed main experiment — interpretation

**Main112/112 verified; utility and factorial are still pending. This is not full-batch closure.**

Eight simulator episodes, four repeated-window OD cohorts, seven arms, two seeds, five participant classes;560 aggregate class rows. Counts do not imply800 independent routes: windows share cohorts and bus templates. All112 generated artifact manifests, source model/catalog links, evaluation links, diagnostic links and exact route/duration pairing were checked.

| Arm | Speed W1 | Change vs B | Acceleration W1 | Change vs B |
|---|---:|---:|---:|---:|
| B | 1.283834 | +0.00% | 0.186744 | +0.00% |
| S | 1.297086 | +1.03% | 0.188423 | +0.90% |
| V | 1.294724 | +0.85% | 0.185917 | -0.44% |
| SV | 1.293373 | +0.74% | 0.187615 | +0.47% |
| P | 1.287757 | +0.31% | 0.187902 | +0.62% |
| SH | 1.288549 | +0.37% | 0.185868 | -0.47% |
| RAW | 1.294157 | +0.80% | 0.188364 | +0.87% |

Lower is better; primary is the first direct output with no GAF selection. Equal class/seed/window/cohort weighting is unchanged.

SV does not improve both primary endpoints. Its speed distance improves in4/16 scene-seed cells; acceleration improves in7/16. The descriptive four-cohort bootstrap intervals for SV−B cross zero for both endpoints. These are small, mixed differences, not evidence of catastrophic failure or population significance. V acceleration improves0.44% for the first draw, while SH improves0.47%; weighted-all-four acceleration worsens0.33% for V and1.06% for SV. A robust GAF-specific motion benefit is not established.

| Participant | SV speed change | SV acceleration change |
|---|---:|---:|
| bus | +3.09% | +6.35% |
| bycicle | +0.01% | -1.41% |
| motorcycle | +0.83% | -2.03% |
| private_car | +1.01% | +4.85% |
| taxi_or_uber | -0.15% | +0.01% |

SV longest-stop W1 improves1.44%, stopped-fraction W1 improves0.32%, stop-frequency W1 improves0.15%; mean-absolute-acceleration W1 worsens4.86%, and four-second speed-change W1 worsens5.23%. Retain all endpoints; do not replace native speed/acceleration primaries with a favorable stopping statistic.

Route and total-duration distributions are paired and unchanged by construction. Of800 declared episode-requests,782 have catalog support and754 have paired physical references. The catalog contains the exact reference route in470/754 paired cases; an alternative route need not be invalid. These coverage limits apply to every arm.

Mean within-request normalized-progress diversity changes from0.053541 to0.053328 (about−0.40%). No constant-speed fallback, upper clamp or near-shape match under the fixed512-training-examples/class diagnostic was observed. This is neither a proof of realism nor a full memorization/privacy audit. Mean negative decode fraction is12.75% for B and12.27% for SV; negative-to-zero decoding also represents legitimate stops and must not be interpreted alone as failure.

The justified methodological conclusion is narrow: this global hidden-feature integration has not demonstrated an incremental overall motion-quality gain. It does not prove that GAF is useless as an explicit condition, output-level constraint or upstream route/duration signal. Existing P/SH/RAW controls caution against attributing small changes specifically to correctly paired GAF semantics.

Wait for independent utility and factorial evidence before locking the next protocol. No current training, sampling, metric definition or thesis conclusion is changed. The candidate mechanism audit remains provisional.

Source: `main-complete.json`, `main-all-class-metrics.csv`, `verified-aggregates.json`. All individual data and weights remain outside this public result directory.

## Direct trajectory inspection

`visual-review/complete-journeys.png` shows all five classes using the first lexicographic jointly supported request in the predeclared390201/seed42/first-draw cell. No visual or metric-based example selection was used. All generated route/time conditions match between B and SV. The plot uses full geometry, full elapsed-time progress and unsmoothed native speed during the first60seconds. It is illustrative, not a population estimate or requirement to reproduce a single reference.

For these examples, B/SV motion is usually close, consistent with a weak output effect from auxiliary hidden alignment. The bus example has1756.5s generated duration versus2392.0s for the reference; the taxi example483.3s versus380.0s. A fixed-total-duration motion intervention cannot remove such timing differences. This does not make every alternative duration invalid; joint distribution and conditional context must be evaluated. The spatial panels are trajectory views, not a full network-legality certificate.

## Auxiliary optimization versus output improvement

All14 final training logs are retained in `training-diagnostics.json` as aggregate first/last1000-update summaries. For SV, the alpha-weighted cosine target loss decreases from0.04556 to0.02518 for seed42 and from0.04607 to0.02992 for seed43 (means of ten logged updates per interval, before the0.1multiplier). Speed-only alignment likewise decreases0.03752→0.01567 and0.03749→0.01887. Class-centroid alignment reaches about0.000084/0.000086. Thus the auxiliary objective is being optimized; its optimization alone is not delivering the required physical improvement. These are training observations, not an independent feature reconstruction test or proof that projection heads explain the whole failure.
