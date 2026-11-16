# Terminal supervision for free-sampled DiffTraj buses

This development pilot changes where training feedback is measured. A pure Gaussian draw and the standard trip request pass through the existing single-class bus A model for the first 97 DDIM updates. The same 2,048 training requests and one newly sampled training noise per request are used by both experimental objectives and both seeds. The remaining three updates use the A weights with a trainable copy of the existing output convolution. Backpropagation through all three links measures the objective on the resulting positions. All other weights and the initial 97 transitions remain frozen. Each adapted model receives 128 AdamW updates (batch 32, learning rate 0.0001 with cosine decay to 0.00001, gradient norm clipped to one). This is a short fine-tune; the two full-training reference models are the existing 50-epoch single-class A and B.

The pointwise terminal control uses smooth L1 on the 199 log(1+speed) values aligned by interval. The distribution objective sorts these 199 values within each trip, compares the order statistics by the same loss, and additionally matches the fraction of contiguous 20-second windows spent at low speed (sigmoid of the window maximum speed, threshold 1 m/s, softness 0.25 m/s). Targets come from the training trajectory matching the request, on the shared 200-point coordinate clock. Both adaptations anchor the displacement of positions and endpoints to the paired free sample from A. Inference reads only the original request and noise. Regularisation and the short suffix favour local motion changes; the experiment does not address transit line selection.

The validation cohort consists of all 1,000 bus trips. Reference A, historical B, terminal pointwise and terminal distribution draws share the same request IDs and initial noise. The original B training target was time-aggregated native SUMO speed. Its contrast with terminal fine-tuning includes the stage, training data/exposure and target measurement. The paired pointwise vs distribution fine-tunes isolate the change in terminal objective as a package. The effects of sorting and the 20-second term are not separately estimated. The reserved test split is excluded.

## Complete validation results

Each row uses the same 1,000 requests and noise draws. Slow-time and entry errors are signed generated-minus-observed differences. The native cohort is measured on the same 199 physical time intervals for motion metrics. An entry requires a preceding faster interval and at least two consecutive low-speed intervals.

| Training | Seed | Speed W1, m/s | Slow time error, pp | Entry error per 10 min | Median slow spell, s | Density JS | OD JS | Length JS | Hotspot F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 42 | 0.7527 | -13.2643 | 0.1719 | 20.2312 | 0.0038 | 0.0564 | 0.0565 | 0.9500 |
| B | 42 | 0.8951 | -16.8246 | -1.5362 | 19.6206 | 0.0050 | 0.0459 | 0.0673 | 0.9000 |
| pointwise | 42 | 0.7338 | -12.3910 | 0.4077 | 20.5930 | 0.0037 | 0.0558 | 0.0583 | 0.9500 |
| distribution | 42 | 0.7209 | -12.0950 | 0.4392 | 20.7437 | 0.0037 | 0.0558 | 0.0528 | 0.9500 |
| A | 43 | 0.7923 | -13.6166 | 0.1432 | 20.1256 | 0.0021 | 0.0290 | 0.0461 | 0.9500 |
| B | 43 | 0.8707 | -15.3121 | -0.2993 | 17.3266 | 0.0027 | 0.0228 | 0.0554 | 0.9500 |
| pointwise | 43 | 0.7793 | -12.7779 | 0.2941 | 20.5930 | 0.0021 | 0.0292 | 0.0569 | 0.9500 |
| distribution | 43 | 0.7654 | -12.6799 | 0.3112 | 20.6080 | 0.0021 | 0.0286 | 0.0472 | 0.9500 |

The distribution objective lowered generated bus speed W1 by 4.2% and 3.4% relative to A and also beat the matched terminal pointwise variant on both seeds. The low-speed-time deficit decreased by 1.17 and 0.94 percentage points. Median spell duration increased by approximately half a second and remained well below the 29.0 s reference median. Both adaptations increased overestimation of slow entries. Hotspot F1 remained 0.95. The length-JS change relative to A differed in direction across seeds. These results support a modest speed-distribution improvement in this simulator validation cohort.

## What the motion objective measures

On 1,000 validation buses a within-trip permutation of the observed speed sequence gives exactly zero pooled speed W1 and zero sorted-speed loss against the original sequence. Yet median low-speed spell duration falls from 29.01 to 13.11 s and entries rise from 6.92 to 12.18 per ten minutes. This negative control acts on speed sequences only. The original 1 Hz coordinates give a 32 s median versus 29.01 s on the 200-point grid. The share of counted spells lasting at least 30 s is 83.5% versus 42.4%; the two-interval minimum and the interval speeds both depend on the clock.

Against equally budgeted terminal pointwise supervision, distribution supervision additionally improves pooled W1 by 0.0129/0.0139 m/s and the median spell by 0.151/0.015 s. Across trips the Spearman correlations of the decrease in the held-out motion objective (excluding coordinate anchoring) with improvement in within-trip speed W1 are .775/.584, compared with .342/.105 for within-trip spell-duration W1 (994/998 evaluable journeys with observed and generated spells in both methods). The output-coordinate gradient norm for the 20 s window loss is .240/.226 times that of the sorted-speed loss on the adapted outputs. This is an output sensitivity, not a parameter-gradient measurement during fitting. Timing, small terminal corrections and geometry anchoring also matter; their separate effects are untested.
