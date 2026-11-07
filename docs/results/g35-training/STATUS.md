# G35 expanded training completed

Verified 2026-11-07 11:58 Beijing. The recovered cloud pipeline completed all simulations, extraction, merging, four 10,000-step diffusion fits, and both fixed 50-epoch GAF teachers. Completion receipts and all six model/config hashes were checked; teacher saved prediction artifacts also match their hashes. Disk has 21.23 GB free. The first read-only status query failed because of shell quoting and was corrected; this did not affect the experiment.

Training: 29,895 full vehicle legs from 24,073 unique parents. Development: 7,230 legs from 5,842 disjoint parents. The five-class training counts are bus685, motorcycle6547, private car18381, bicycle2744, taxi/uber1538. These are activity legs, not all separate vehicles; very short eligible legs are retained. Native training contains65,091,746 coordinate observations. Public native demand, map and split rules are unchanged.

| Self-trained GAF teacher | Development accuracy | Macro F1 |
|---|---:|---:|
| Spatial original Hilbert–GAF | 86.17% | 76.47% |
| Speed GAF | 89.13% | 90.34% |

These scores use all7,230 development legs. Spatial per-class F1 is95.65/73.57/95.20/72.10/45.85% in the class order above; taxi/uber remains weak. Training accuracy is97.22% spatial and92.39% speed. Class imbalance, shared public map/lines and train–development gap remain material limits. These raw scores cannot establish an improvement over G33 on its different validation set, nor do they establish guidance or generation quality.

Both native diffusion fits took approximately46 minutes each; both257-point controls approximately5.3 minutes each, parallelized over four GPUs. All are10,000 steps; generation evaluation is not yet performed for the expanded models. No new thesis quality claim is added.

The preregistered same-development-cohort old/new teacher audit has been launched in the cloud, using original fixed weights and2,000 paired parent bootstrap replicates. This is analysis, with no refitting or test-set selection. G36 independent generation evaluation remains to be fixed and executed under the existing analysis protocol.


Same-validation audit completed: both old/new models evaluated on identical7,230 legs/5,842 parents. Spatial accuracy60.65→86.17%, macroF1 50.10→76.47%; speed accuracy63.61→89.13%, macroF1 61.82→90.34%. Paired parent-bootstrap95% intervals for macroF1 gain are spatial[24.54,28.16]pp and speed[27.27,29.73]pp. This supports improved classifier generalization on this public-demand development cohort. Scale, full-day coverage and number of optimization updates changed together; it is not a pure data-size causal effect, unseen-city test or demonstrated generation benefit. Full fixed analysis: paired-teacher-audit.json.
