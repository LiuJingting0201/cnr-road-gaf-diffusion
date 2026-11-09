# G39 interim: first two complete OD cohorts

**Incomplete experiment: descriptive interim only, not a thesis success/failure verdict.**

Four scenes, two complete OD cohorts, seven arms, two model seeds, five classes:56cells/280class rows.
This subset was fixed from completion status before inspecting metric values. The fifth complete scene
was excluded because its paired window/control grid was not yet complete at initial inspection.
Equal class/seed/window/cohort means; primary is the first direct candidate, no GAF ranking.
All arms have identical route and drawn-duration pairing hashes; route/duration metrics agree exactly.

| Arm | Speed W1 | Change vs B | Acceleration W1 | Change vs B |
|---|---:|---:|---:|---:|
| B | 1.301576 | +0.00% | 0.174361 | +0.00% |
| S | 1.316666 | +1.16% | 0.176109 | +1.00% |
| V | 1.317586 | +1.23% | 0.175553 | +0.68% |
| SV | 1.312431 | +0.83% | 0.176524 | +1.24% |
| P | 1.304403 | +0.22% | 0.176318 | +1.12% |
| SH | 1.307760 | +0.48% | 0.174182 | -0.10% |
| RAW | 1.313705 | +0.93% | 0.176098 | +1.00% |

Lower W1 is better; a positive change is worse. S:spatial GAF; V:speed GAF;
SV:both; P:class-centroid targets; SH:another same-class trip feature;
RAW:raw-series auxiliary encoders, different architecture/pretraining.

Complete secondary, class, seed and cohort results remain in summary.json/all-class-metrics.csv.
No training, sampling, metric or follow-up protocol changes are made from this interim look.


## Interpretation of this interim subset

The GAF alignment arms do not improve either aggregate primary endpoint in this
subset. SV speed is worse in7/8scene-seed cells; acceleration is worse in6/8.
Seed42 acceleration changes+2.49%, seed43 approximately−0.01%, so the latter is
essentially neutral. The all-four-candidate secondary metrics also worsen for
S/V/SV. These are descriptive differences; two OD cohorts do not support a
population-level significance conclusion.

SV improves stop-frequency distribution W1 by2.06%, longest-stop W1 by1.38%
and stopped-fraction W1 by0.59%, while longest-steady W1 worsens1.19% and
four-second speed-change W1 worsens4.84%. RAW improves stop-frequency W1 by2.62%,
and SH has similar stopped-fraction change, so the small stopping changes do not
establish a GAF-specific benefit. Class-centroid and shuffled-trip controls also
fail to improve both primary endpoints. RAW differs in encoder architecture and
pretraining; this is not an encoding-only causal comparison.

SV native acceleration W1 improves2.99% for motorcycles, but worsens6.39% for
buses and5.91% for private cars. RAW's motorcycle change is−4.67% and SH's−2.93%;
that isolated class improvement cannot establish a uniquely useful GAF signal.

Mean within-request progress diversity changes−0.69% for SV; no wholesale
collapse is apparent in this diagnostic. No constant-speed fallback or finite-
library near-copy under the specified1e−4threshold appears in the included
outputs, but this is not proof of privacy, universal realism or full route
coverage. Negative decoded bins still require clamping: class/scene/seed-macro
fraction12.54% for B versus11.92% for SV. A reduction is not automatically a
physical improvement because zero-speed output also represents real stops.

There are400declared scene requests,390with route-supported output and378with
physical/route references in this subset. These references are reused across
model arms/seeds; they are not independent replications. Catalog contains the
exact reference route for228/378, and an alternative route need not be invalid.

The experimentally supported scope is this motion-layer auxiliary-feature
alignment implementation under fixed route/duration draws. It has not yet
converted the teachers' classification representations into improved generated
motion. Input-route/class information and model capacity could be explanations,
but this interim analysis does not establish a mechanism. Finish the fixed
remaining cohorts, utility and factorial controls before further design choices
or thesis claims. No interim result has been used to alter the live experiment.
