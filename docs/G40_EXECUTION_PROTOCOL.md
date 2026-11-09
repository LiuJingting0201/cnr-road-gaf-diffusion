# G40 finite execution protocol

2026-11-10 Asia/Shanghai. User explicitly resumed experiments and requested the
thesis be revised after results. This protocol fixes the implementation choices
left open in the design. Preparation/pilots/formal outcomes are separate receipts;
none is a completed quality result merely because a controller starts.

## Data, preparation and isolation

Reuse the immutable G35 native five-class dataset, original spatial GAF cache,
speed-GAF transform and G35/G39 self-trained teachers. Train29,895legs/24,073parents;
no historical frozen-test access. Parent/identity/label arrays must agree exactly.
Prepare a lossless FP32 cache of each frozen encoder prefix through its penultimate
residual stage. Its final stage remains trainable in joint arms. Recompute every
training feature with the source tail and require agreement with the archived
teacher features (atol=rtol=2e−4); verify lossless cache round trips. This numerical
tolerance is for frozen CNN inference under possibly different batch grouping,
not a replacement for the G39 performance benchmark's exact-state requirement.
Running batch-norm statistics stay frozen throughout G40.

Four prefix views: spatial/speed GAF and raw XY/speed. New `g40a-job` prepares them,
then performs the bounded training-only sensitivity audit in the design. At most
100distinct parents/class, SHA-ranked IDs, one leg/parent. Exact copy, native
encoder/decoder replay, monotone sinusoidal time warp amplitudes.1/.3, within-one-
second four-bin mass redistribution amplitudes.1/.3, and4/16-bin moving averages.
Physical clocks/endpoints and the continuous observed polyline stay fixed; report
changes in sampled chord length. This measures sensitivity, not realism ranking.
No adaptive teacher search or exclusion of adverse classes.

Wait for the existing G39 performance queue to complete. Adopt cached native inputs
only if the preregistered training state/RNG equality passes and every paired slot
has a whole-run speedup>1. Otherwise keep original input packing and retain failed
performance evidence. Adopt evaluation/utility single-load IO only after exact
output checks. Preparation cap2h/16GiBnew/free12GiB; no takeover of unrelated jobs.

## Fixed model and optimization

Eight arms: B, F-SV, J-S, J-V, J-SV, J-RAW, J-SH, C. F freezes the source CNN;
J trains its last residual stage/class head; C supplies class-mean original GAF
features. Every active view has512→128→32→LayerNorm bottleneck. The dual code is64D;
inactive halves are zero. A64→128→128 adapter adds the code to the denoiser's
conditioning at every block and timestep. Its final linear layer starts at zero,
so initial trajectory predictions equal the common B warm start. B bypasses it.

Every formal arm/seed restarts from completed G39 B final EMA, seeds42/43. Pilots
are not warm starts or checkpoint candidates. Formal continuation5000updates,
effective batch64, max microbatch32/token budget32768, FP32, AdamW weight decay1e−4,
denoiser LR2e−4, encoder LR5e−5, global gradient clip1, EMA.001. Balanced class
sampling and identical per-seed batch/timestep/motion-noise streams across arms.
Loss is original physical-time noise MSE plus.1×mean per-view class CE for joint
arms. Frozen/class-centroid encoders receive no class-update loss. Preserve all
final results; no development checkpoint, loss-weight or sampler search.

J-SH uses the existing deterministic same-class/approximate length/T derangement.
Its code prior is trained on the **same wrong-trip associations** used in training;
do not silently repair the pairings at code extraction. Record same-parent matches.
RAW initialization and architecture differ from GAF; it is a representation
alternative control, not perfect encoding-only isolation. Cost/parameter counts
and warm-start training budget must be reported.

After fitting, freeze final EMA encoders and extract all training codes. Learn
a conditional64D latent diffusion prior: four256D residual MLP blocks; inputs are
class, route geometry and existing5D route/duration context. Train5000updates,
balanced batch256, AdamW2e−4/1e−4, clip1, EMA.001, fixed final weights; train-only
per-dimension standardization with SD floor1e−3. Use the same1000-step linear noise
schedule(.0001→.02) and50DDIM sampling steps. B uses no code; C uses its deterministic
class-code lookup and has no prior optimization. All other arms use a learned prior.
Sampling creates the code from public/generated conditions, never from target GAF.

## Pilots and checks before formal experiments

Four pilots B/F-SV/J-SV/J-RAW, seed42,1000motion updates each; active prior fits
are capped at1000updates. Check exact zero-adapter initialization, padding, EMA
ownership, encoder-gradient routing, frozen-prefix equivalence, future-input
rejection, consistent route/T/noise keys, finite output and retained stochastic
variation. These are implementation/sensitivity checks using training-side data,
not evidence of independent generation benefit. Retain any failure and fix it in
a new isolated snapshot before retrying. Do not alter running scientific snapshots.

## Independent generation and usefulness

Eight fresh SUMO episodes400201–400208: four disjoint nonbus OD-parent cohorts
(selection seeds40001–40004), paired departure windows18000/32400s, unchanged public
bus-service templates and background rules. Exclude G36/G38/G39 nonbus cohorts.
Freeze requests before new outcomes; retain coverage/failures without replacement.
Verify training-parent disjointness and cohort separation before simulation.

Use frozen G38 neural route preferences and unchanged training-only duration prior.
Four direct draws/request, matched route/T/motion-noise keys across all8arms and
both seeds. Main evaluation128cells/640class rows. K1 is primary; all-four weighted
distribution and diversity are secondary. No GAF terminal ranking. Extra code-prior
forwards are reported separately from100motion-denoiser forwards/draw.

Keep native speed/acceleration W1 co-primary, with all stopping/cruise, one-second
acceleration, route coverage, class/scene/seed, diversity and nearest-training-shape
diagnostics. Cohort-level intervals are descriptive with four OD cohorts. A known-
trajectory GAF code may appear only in a separate reconstruction diagnostic; never
count such results as prospective synthesis. The source FCD precision limitation
is retained; do not replace original primary endpoints after seeing outcomes.

Downstream utility: balanced1000real training parents and1000matched synthetic rows,
same200/class, first2ODcohorts for synthesis and last2for evaluation. Include B,
F-SV,J-S,J-V,J-SV,J-RAW, empirical resampling and real-only; synthetic-only and
real+source for each source, total15regimes×seeds4001/4002. Reuse independent raw-
series evaluator,2000updates/fit, identical budgets, final checkpoint. All results,
including adverse physical outcomes, stay visible. Shared bus services limit
generalization. No new success claim until generation and utility are reviewed.

## Delivery

Use isolated snapshots, four owned GPU workers, at most two SUMO workers,24h bounded
main queue,24GiBnew G40 output budget plus shared preparation,12GiB free reserve.
Preserve failures and exact source/config/artifact hashes. Public GitHub gets code,
small aggregate metrics, plots and prose; raw data/weights remain server/private.
After complete evidence review, revise thesis methods, results, figures, abstract
and discussion consistently. Main figures must include all five participant classes,
generated trajectories and both GAF views, alongside independent quality results.

## Pilot finding and bounded prior check — 2026-11-10

All four1000-update motion fits and diagnostics completed, but the pilot supervisor
failed while checking a diagnostic's source model hash as a local model artifact.
Preserve g40p-job/failed.json and every completed fit. Fix only provenance checking;
independently verify all saved motion artifacts, tokens, RNGs and initialization.

The1000-update EMA priors produced code maxima119–178, far outside ordinary
LayerNorm encoder support, and increased training-side denoising MSE versus known
codes. Do not release formal generation on those priors. In a new g40pr-job, reuse
all four motion fits and fit the declared formal prior budget5000updates from its
original initialization (B remains deterministic), then repeat the identical25-parent
checks. This is a single bounded training-only prerequisite, not a held-out quality
sweep. It changes no formal5000-update budget, architecture or endpoint. Original
1000-update prior results remain visible. Formal release depends on actual review;
if extrapolation persists, diagnose before more generation.

## Registered final-model mechanism audit — 2026-11-10

Before any formal generation-quality outcomes are available, fix one secondary
training-side condition intervention. After the full scientific and figure queues
close, audit all eight arms and two seeds on the same 25 training parents selected
by the pilot rule (five per class, stable SHA order). At fixed diffusion noise
indices 50/250/500/750/950, compare assigned training code, a cyclic same-class
other-parent code, zero code, and the sampled prior, holding route, duration and
noisy motion input fixed. Report every noise MSE and prediction-change RMS with
physical-time weights. These indices diagnose the denoiser; they are not selected
guidance intervention times. SH retains its assigned wrong-trip code.

No parameter fitting, synthesis reranking, endpoint changes or future evaluation
GAF inputs are allowed. These are in-sample mechanism checks, not independent
quality evidence or a checkpoint-selection rule. Bound the new isolated audit to
four GPU workers, 30 active minutes, 1 GiB output and a 12 GiB reserve; it waits
for the existing figure queue to finish, avoiding resource interference.
