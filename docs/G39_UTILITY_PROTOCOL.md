# G39 utility stage, fixed before generation outcomes

2026-11-09. Follow-up to G39, no main experiment outcomes have been opened.
Question: do synthetic data from the feature-supervised generator help a separate
raw time-series model on previously unobserved SUMO trajectories?

Generator arms B/S/V/SV/RAW, all retained; no best-arm selection. Combine generator
seeds42/43 with equal selection opportunity. Utility training synthetic conditions
are the first two OD cohorts (episodes390201–390204); evaluation uses only the
last two cohorts (390205–390208). Nonbus source-parent sets must be disjoint.
Bus service templates are shared; simulator episodes and actual vehicle identities
differ, and this is not unseen-line generalization. No evaluator receives any
GAF representation or teacher weights. Frozen historical tests remain unopened.

Real base set: fixed SHA-ranked200 distinct training parents per class from G35,
one leg each, total1000. No validation samples. Empirical control:1000 additional
draws with replacement from the exact same1000 legs, equal200/class. Synthetic
control:200/class, fixed SHA rank on episode/model-seed/request/candidate identity,
shared identities across all five generator arms; all four unscored draws are
eligible, without class-confidence or reference-quality filtering. Require enough
common eligible generated identities, otherwise record failure (no smaller
post-hoc budget). These counts describe downstream learning, not new simulator
observations or independent sampled routes.

Regimes: real-only; each of empirical/B/S/V/SV/RAW alone; real plus each of those
six sources. Total13 regimes × downstream seeds3901/3902 =26 bounded fits. The
empirical-only regime checks resampling variability; real+empirical controls extra
presentation of the original1000 cases. Real+synthetic has2000 stored rows but the
same update budget as every comparator, not twice the optimization budget.

Independent evaluator: four-layer temporal residual CNN,64channels, dilations
1/2/4/8, global mean/max pooling and five-way head, scratch initialization.
Inputs are257 relative-time XY samples plus observed arc-speed resampled on that
clock; XY global coordinate mean/SD from the real training subset only, speed/35.
This evaluator can use route and motion; no GAF classifier confidence is a label
or metric. Same initial state, balanced class sampling64/update, AdamW2e-4/1e-4,
2000 updates, final checkpoint, no validation-based checkpoint/hyperparameter search.

Evaluation: all admissible last-two-cohort SUMO references, with exact coverage
and every per-class confusion matrix/F1, macroF1 primary and accuracy secondary.
Report each downstream seed, each simulator episode, pooled macroF1 and
source-parent-weighted nonbus results. Bus is also reported separately because
service templates repeat. Only two held-out OD cohorts: inference intervals are
descriptive and no broad population significance claim is made. The expected
comparison is GAF synthetic versus B synthetic and empirical resampling, followed
by real+GAF synthetic versus real-only/real+empirical/real+B. RAW tests whether
GAF exceeds a different auxiliary representation. Physical quality remains the
main G39 outcome; downstream classification does not override degraded dynamics.

Run only after all required G39 models, references, candidate pools and pairing
checks complete. New isolated g39u-job, wait bounded24h for G39 then bounded4h
utility stage; four GPU workers only after G39 releases its GPUs. Output cap4GiB
and free-space reserve12GiB. Keep failure/coverage records. Do not modify thesis
claims until physical and utility evidence have been reviewed together.
