# GAF integration audit — hypotheses, not the next locked experiment

2026-11-09. User direction: finish the entire current G39 batch, then design the
next round around road-conditioned diffusion trajectory synthesis and genuine
GAF utility. This note records code-level mechanisms while results are pending.
It neither selects a favorable result nor authorizes a new scientific run.

## The thesis question stays unchanged

Given map/OD, participant class and applicable public service conditions, synthesize
plausible complete trajectories: route choice, journey timing, and movement along
the route. Test whether representations derived from original spatial Hilbert–GAF
and separately defined speed-GAF add value beyond class conditioning and ordinary
trajectory losses. The delivered model directly generates trajectory motion;
there is no image-to-trajectory inversion and no supplied group CNN.

## What the implementation actually constrains

G39 globally pools the denoiser's middle hidden state and learns two small projection
heads to match512D frozen classifier features. The heads are removed at inference.
The nonzero auxiliary gradient into the backbone has been tested. That proves a
gradient path, not that the decoded trajectory acquires better dynamics. A projection
head may predict class/route features without preserving detailed temporal behavior.
Class, route and duration are already provided as conditions, so conditional
predictability and redundancy are plausible explanations to test.

G39 deliberately pairs route and sampled total duration across arms. Spatial GAF
information about route geometry therefore cannot change route preference in this
comparison. Nor can a better motion loss repair a wrong total-duration prior when
that total is fixed. The factorial experiment separates route and duration marginal
changes from within-route/duration motion reweighting; wait for it before assigning
the next intervention to the route, duration or motion component.

The G35 speed teacher first obtains approximately1s arc-speed, then resamples a
complete journey to128values. Its classification success does not establish
sensitivity to native0.25s acceleration or short stop/start events. Time compression
can weaken those signals. A single global teacher feature and native physical
quality metrics may therefore respond to different aspects of a journey.

## Differentiability is a real implementation boundary

Original spatial Hilbert quantization, squeeze/duplication and uint8 image conversion
are discrete. Do not pretend an unchanged original spatial encoder supplies exact
coordinate gradients. It can evaluate discrete route candidates, supply frozen
training targets, or motivate an explicitly validated surrogate; those are different
methods and must be named separately.

Speed GASF/GADF admit continuous derivatives inside their domain. The existing RGB
speed representation also includes hard-binned MTF, whose transitions depend on
discrete bin indices. Backpropagating through this input is only a partial-gradient
method, not differentiation through the entire original three-channel construction.
Saturation/clipping, endpoint derivatives and gradient scale need explicit checks.
Replacing MTF by a soft version is a new representation requiring its own control.

## Candidate improvement to decide after full G39

First perform a bounded training-only sensitivity audit: alter stop/start timing,
smooth or add acceleration irregularities, and preserve endpoints, route and duration.
Measure changes in frozen spatial/speed feature distances and independent physical
signals. Compare same-class trajectory variation with controlled corruption. Do not
equate a classifier-confidence increase with a quality increase, or choose thresholds
from the current held-out outcomes. Verify gradients on output coordinates/speed.

If the appropriate GAF representation detects the relevant errors, consider placing
the auxiliary constraint on the decoded clean-trajectory estimate rather than only
the denoiser's globally pooled hidden vector. Construct the speed sequence from that
output using explicit physical time; compare whole-journey versus fixed-duration
local windows only under a predeclared limited protocol. Include a direct raw-speed
or raw-series auxiliary control with the same examples, updates and conditioning.
This asks whether the GAF transformation adds more than extra supervision.

The next protocol must account for earlier G24 sampled guidance and G27 continuous
angular-loss experiments. Repeating them under a new name is not a new contribution.
Native physical-time data, direct decoded-output supervision and a measured teacher
sensitivity mechanism would be substantive differences only if actually implemented
and tested. Avoid searches for a special denoising timestep and blind increases in
class-confidence guidance strength.

Spatial information belongs in a component where geometry can change; speed
information belongs in a component where motion can change. This is a candidate
hierarchical design, not evidence that both views improve both components. The full
factorial evidence may instead show insufficient within-route timing sensitivity.

## Evidence required before a stronger story

Read all112 main cells, all26 independent utility fits and all4 factorial cells,
including adverse classes, stop/cruise behavior, acceleration, diversity and coverage.
Keep primary speed/acceleration distances and their existing weighting. Any new
diagnostic is secondary and explicitly motivated, not a replacement chosen to win.
An auxiliary gain must beat both the base diffusion and relevant non-GAF controls.

Current G39 outcomes become development evidence for subsequent method changes;
confirm on fresh OD cohorts/independent simulator episodes after locking the next
protocol. Historical frozen tests stay unopened. Update thesis claims only after
completed results support them. A useful final story is not guaranteed by a useful
classifier, and this audit does not claim that a proposed change will succeed.

## Relevant research analogies, with limited scope

[REPA](https://arxiv.org/abs/2410.06940) motivates noisy-hidden/clean-feature alignment,
but its image-model setting does not validate our global class-feature targets.
[DOODL](https://arxiv.org/abs/2303.13703) discusses limits of guidance through approximate
denoised outputs; it does not prove a particular trajectory guidance schedule works.
The local ablations and output-quality measurements remain the deciding evidence.
