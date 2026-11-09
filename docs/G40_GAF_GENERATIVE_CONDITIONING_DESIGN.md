# G40 design: generation-trained GAF conditions for trajectory diffusion

2026-11-09. Technical design following completed G39, not a launched experiment
or a claim of improvement. The user currently prioritizes the thesis mechanism;
the separate performance benchmark remains unstarted behind its review gate.
An implementation/configuration manifest must be fixed before new fits or fresh
reference outcomes. Do not silently substitute this design for completed G39.

## Question and selected change

Can the inherited spatial Hilbert–GAF and a separate speed-GAF provide information
about within-class movement that a trajectory diffusion model actually uses?
Replace frozen-feature hidden alignment with **GAF-conditioned trajectory
diffusion**, and train the GAF encoders partly through the trajectory denoising
objective. Learn a conditional distribution over the resulting codes so they are
available for genuinely new synthesis. This is the selected next design; do not
simultaneously launch output-gradient guidance, special-timestep searches, or
another classifier architecture sweep.

This extends the earlier tentative idea of retaining a predicted feature head.
That idea would still ask the generator to reproduce fixed classification features.
Here the representation itself is trained for generation and an explicitly sampled
code carries variation beyond the public condition. The added prior and its possible
distribution mismatch are material costs, not details to hide.

## Evidence motivating the design

G39 main112cells show no overall physical gain from global hidden alignment.
Synthetic augmentation is useful, but speed-GAF94.10% and RAW94.06% downstream
macroF1 do not establish a GAF-specific advantage. Completed factorial4cells use
the same10,784candidates across10rules. After preserving the exact joint route/T
distribution, speed-score reweighting changes speedW1−0.121% and accelerationW1
−0.133%, with only1/4 and2/4cells improving. Spatial changes are+1.598/+1.243%;
dual-view changes+0.904/+0.606%. These exposed-scene diagnostics do not justify
stronger classifier-score guidance or demonstrate that full latent features are
useless. Both distinctions matter.

Class-equal variance decomposition of existing training targets attributes54.42%
of spatial and67.21% of speed-feature variance to class means; remaining variation
has not yet been shown to encode useful motion conditional on route and duration.
Classification accuracy, feature variance, and trajectory quality are different.

## Exact inheritance and separate views

Spatial input remains the original frozen XY→Hilbert→squeeze/mirror→GASF/GADF/MTF
construction, with its inherited ordering and train-fitted scales. It receives no
speed channels. Speed input remains the separately documented G35 speed sequence
and angular/transition image, derived from the same timed positions. Changing its
temporal resolution would be a separately named representation extension.

Initialize the encoders from our existing G35 CNNs; no supplied CNR CNN is claimed.
Use separate encoders and class heads. Start with two32D bottlenecks, one per view,
concatenated for dual-view conditioning. Their axes are not presumed to be named,
disentangled physical attributes. Preserve the five participant classes and report
each separately.

## Training: GAF must help recover motion

For each training trajectory, compute its two fixed input images and encode them
as zS,zV. Condition the native-clock motion denoiser on route, class, duration,
noise level, and these codes. Inject the code through learned modulation of the
denoising blocks at every denoising step; its contribution is learned during training.
Keep the same injection at synthesis. There is no hand-selected intervention step.

The proposed objective is unchanged physical-time-weighted noise MSE plus a fixed
class-preservation loss on the GAF encoders. Backpropagate the denoising loss into
their final residual blocks and bottlenecks; keep earlier layers and batch-norm
statistics frozen initially. Thus an encoder is rewarded for carrying information
that helps predict the trajectory, while retaining the inherited classification
task. Compare this with an otherwise identical frozen-CNN conditioning arm.

The images are observed training inputs. Gradients pass through the CNN parameters
and continuous code, not through Hilbert quantization or image construction back
to XY. No claim of differentiating the exact discrete encoder is needed. The
model directly produces trajectory motion, with the existing road decoder to XY;
there is no generated-image inversion or unique inverse-GAF assumption.

## Synthesis: where does the code come from?

After encoder/denoiser training closes, freeze them and extract codes for training
parents only. Fit a small conditional latent diffusion prior p(zS,zV | route,class,T),
using only available route geometry/context, participant class and sampled duration.
Fit code standardization on the training side; inverse-transform prior samples
before sending them to the trajectory decoder. Both codes are sampled jointly to
retain their learned dependence. A prior of the same architecture/budget is used
for the RAW-code comparison.

For a new request: sample a legal route using G38 preferences and a duration using
the unchanged training-only prior; sample a code from the learned conditional code
prior; sample native-clock motion from diffusion conditioned on that code. Different
code/noise draws produce candidate trajectories. The main output is the first draw,
without GAF ranking; all draws also contribute to secondary distribution/diversity
evaluation.

Never compute a code from the held-out trajectory being synthesized. A known-trip
code can be used only in a clearly separated train-parent-split reconstruction
diagnostic. Its performance is not a valid synthesis result. Quantify the gap between
such reconstruction and code-prior sampling; a method that succeeds only with the
known-future code has failed the generation question.

## Bounded implementation and experiment sequence

1. Training-only input sensitivity audit, at most100distinct parents/class under
   a fixed identity ordering. First replay observed motion through the existing
   path decoder and compare raw-FCD versus replayed GAF/features and physical
   signals, so unachievable source/decoder differences are not mistaken for a
   missing generative constraint. Preserve route/endpoints/T while applying two fixed
   severities of monotone time warp, native subsecond mass redistribution, and
   temporal smoothing. Retain exact-copy controls and every class. Measure image
   and feature differences alongside native/one-second motion changes; do not treat
   classifier confidence as a realism score. Coarse image information cannot be
   assumed to recover dynamics it has discarded. No new classifier search.
2. Four bounded implementation pilots (B, frozen-SV, joint-SV, joint-RAW), one seed,
   at most1000updates each, before fresh reference generation. Verify data isolation,
   gradients through the intended encoder layers, exact zero-adapter initialization,
   sampler code availability, code/noise diversity, and runtime/memory. Record
   failed checks. These pilots do not select a best checkpoint or prove quality.
3. Proposed main comparison: B; frozen-SV; joint-S; joint-V; joint-SV; joint-RAW;
   joint-SV with wrong-trip images shuffled within class and approximate length/T;
   class-centroid-code control. Eight arms×two seeds, fixed final checkpoints and
   equal5000continuation updates from the same completed G39 B backbone per seed.
   Keep batch64, native clock, path decoder, route policy, duration prior and data
   unchanged. All controls instantiate matched code adapters. Report additional
   encoder/prior training cost and parameter differences; equal updates are not
   equal FLOPs. Joint RAW vs GAF is not a pure encoding-only comparison because
   CNN architecture/pretraining differ. Exact loss weights, prior steps and source
   hashes must be fixed in the execution protocol before any formal launch.
4. Lock fresh OD-parent cohorts and independent SUMO episodes before their outcomes.
   Exclude exposed G36/G38/G39 nonbus cohorts; disclose shared bus services/map.
   Pair route/T and diffusion-noise keys across arms. Historical frozen tests remain
   untouched. G39 is now development evidence for this design.

This design does not add a route/T reformulation in the same round. Spatial GAF may
encode relative occupancy/temporal traversal along a fixed route, but geometry it
cannot change is not an expected gain here. The existing learned route policy still
provides the full pipeline's route-choice component. No claim that both GAF views
must improve both components is made.

## What would establish the missing thesis link?

- **Useful representation:** same-class, similar-route/T codes vary with actual
  movement, beyond the information in the public condition. Joint vs frozen encoding
  tests whether generation training is needed. A high class score alone is insufficient.
- **Used by the generator:** hold route/T/noise fixed and exchange eligible same-class
  codes; measure changes in stopping, cruising and speed change. Hold the code fixed
  and change noise to test retained variation. Changes alone are not quality gains.
- **Better synthesized trajectories:** main comparison uses prior-sampled codes only;
  retain native speed/acceleration W1, one-second acceleration, stopping, diversity,
  coverage, legality and nearest-training-motion checks for every class/episode/seed.
  Add observation-precision diagnostics without replacing existing primary endpoints.
- **Useful data:** repeat matched-budget independent raw-series augmentation on
  separate reference cohorts, against real-only, resampling, B and RAW synthesis.
  This cannot override degraded physical motion.

A broad GAF benefit requires independent physical improvement over B and appropriate
controls without collapse or major adverse classes. If RAW matches the gain, describe
GAF as a useful representation option, not an exclusive or superior mechanism.
If only reconstruction or GAF self-scores improve, the missing link remains unproven.
Do not keep increasing guidance strength or replace the primary endpoint to win.

## Thesis claim and research precedent

The intended contribution is the tested transfer from inherited GAF classification
representations to generation-trained, sampleable conditions for road-constrained
trajectory diffusion. Its success is an empirical hypothesis, not a promised result.
Generative semantic codes plus a diffusion decoder and a learned code prior have
precedent in [Diffusion Autoencoders, CVPR2022](https://arxiv.org/abs/2111.15640)
and the [authors' project](https://diff-ae.github.io/). Those image-domain findings
motivate an adaptation, not a trajectory result or an algorithmic-priority claim.
Our specific research questions concern original spatial GAF versus speed GAF,
topology-constrained complete trajectories, usable sampling conditions, and
independent physical/downstream gains.
