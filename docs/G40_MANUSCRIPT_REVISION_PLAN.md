> Current status, 2026-11-10: full G40 evidence has been reviewed and integrated into the actual v20 manuscript. Final rendering/packaging QA is complete; delivered artifacts are in `deliverables/thesis-v20` and the build receipt is `docs/results/thesis-v20-build.json`. The preparation notes below are retained as historical decisions, not pending scientific tasks.

# G40 manuscript revision plan — evidence first

Registered while the formal queue is running. This is a revision map, not a result
or a promise of a GAF advantage. The v19 manuscript remains unchanged until the
formal generation, utility and condition-use evidence have been reviewed.

## Two research questions

1. Can a trained conditional diffusion system synthesize complete, variable-duration
   journeys with plausible route preferences and movement on a permitted road graph?
2. Does inherited spatial Hilbert–GAF and/or derived speed-GAF information improve
   the synthesized data beyond class, route, duration and non-GAF controls?

The delivered object is timed XY data, not images, class scores or simulator demand.
Route choice, duration, motion and the geometric decoder have distinct responsibilities.
G40 is learned conditioning throughout denoising, not classifier-gradient guidance.

## Evidence hierarchy

- CNR source audit and native-demand reconstruction motivate data choice; the expanded
  corpus is newly simulated, not a reconstruction of the supplied tables.
- G33 establishes a native-clock motion representation with explicit computation and
  stopping costs. It is not a new large-corpus temporal-resolution comparison.
- G35 establishes development recognition by our own spatial/speed CNNs. Its scores
  describe those frozen readers before G40 joint fine-tuning.
- G38 tests learned route preference versus shortest and uniform alternatives. Correct
  alternatives between the same endpoints are not errors against a unique route.
- G39 fixed-route/duration score controls and frozen-feature alignment explain why
  class recognition alone does not establish useful generation supervision. Utility
  shows whether synthetic data help, separately from a GAF-specific advantage.
- G40 tests trainable generative conditions plus a training-only conditional code prior,
  all eight arms and both seeds, eight fresh reference episodes/four OD cohorts,
  native speed/acceleration co-primary and all movement/diversity/coverage endpoints.
- G40 utility uses an independent raw-series task and disjoint reference cohorts.
  Final-model code interventions are explicitly in-sample mechanism diagnostics.

## Consistent manuscript changes after evidence review

Abstract and introduction: full map+OD+class synthesis task, two questions, actual
supported findings and limitations. Do not headline post-hoc best-class percentages.

Method: next-edge policy and constrained catalog; duration prior; native-time motion
diffusion; deterministic decoder. Explain alternative valid routes and missing
background-state conditioning. Preserve all normalization and sampling definitions.

GAF chapter: original spatial construction and separate speed construction; frozen
ranking/alignment as controlled precursors; joint final CNN stage, bottlenecks,
class-preservation term, conditional latent diffusion prior and fresh-code sampling.
Credit Diffusion Autoencoders for the general learnable-code/conditional-decoder
idea. Include controls, initialization, prior failure and bounded budget recovery.
No future trajectory/GAF is available at prospective synthesis.

Protocol: distinguish old exposed evidence from new prospective references; preserve
all request/cohort denominators, shared public bus services, seeds, physical-time
weighting and compute. Explain K1 primary versus all-draw weighted distributions.
Report matched controls, downstream model/label budgets and descriptive intervals.

Results: lead with trajectory generation and independently measured physical quality,
then explain whether GAF contributes. Condense earlier chronological sections while
keeping their underlying results and artifacts. Replace the main five-class example
set with the fixed-order G40 set; never choose examples by quality or teacher score.
Show output spatial/speed images as visualizations, not synthesis inputs.

Discussion: distinguish representation sensitivity, class recognition, actual code
use, physical quality and augmentation. Preserve conflicts and adverse endpoints.
Explain limits of subsecond sensitivity, source FCD precision, finite route catalogs,
shared templates and single-agent simulation; do not claim unseen-city or safety.

## Delivery and checks

Use all actual outputs, not placeholders. Verify every source/artifact receipt,
recompute downstream metrics, and inspect class/scene effects before writing claims.
Update all chapters, abstract, subtitle, cover version and README consistently.
Keep administrative metadata unconfirmed. Add only the two already verified relevant
primary references (Diffusion Autoencoders and path-choice tutorial), with substantive
citations. Existing 52 entries are not all peer-reviewed papers.

Build a clean self-contained Overleaf package and PDF, resolve references, render all
pages, inspect figure legibility and tables, compare clean ZIP build page text, record
hashes and counts. Keep raw trajectories, weights and credentials out of public code
and manuscript packages. Push code, aggregate evidence, figures and final sources.

Packaging preflight: `thesis/tools/package_thesis.py` collects actual LaTeX inputs
from the compiler recorder, builds a clean source ZIP, checks rebuilt PDF page text,
counts citations/figures/tables and records every input/output hash. Tested against
unchanged v19: 109 PDF pages/75 main pages/31 figures/22 tables/52 cited entries,
zero overfull/undefined warnings and identical clean-build page text. This is a
build check only; it does not mark the forthcoming manuscript scientifically or
visually reviewed. Temporary preflight output is not a new delivered thesis version.

Source-level writing corrections found during waiting: G40 reuses the complete G35
speed cache (`g35c_teachers.speed_profile`), which interpolates available one-second
arc increments and uses a constant mean arc speed for a subsecond leg. Do not carry
the old terminal-reader paragraph's five-increment failure rule into the G40 training
description. The 2,454 training/527 development subsecond cases remain included.
The class-preservation heads act on the pooled 512D CNN features; the separate
512→128→32 bottlenecks feed the denoiser. The learned route score is the sum of
masked branching-choice log probabilities, renormalized over the finite catalog;
there is no length-normalization or learned full-graph equilibrium claim.

Interpretation boundaries to retain: J jointly changes the CNN tail through both
denoising and class-preservation losses; there is no CE-only tail-adaptation arm,
so a J versus frozen advantage cannot isolate denoising gradients alone. Zero-code
interventions retain the learned adapter and its biases; they are not removal of
the complete adapter module. The augmentation generator was trained on all29,895
labelled legs, while the downstream real-only learner sees1,000 balanced parents.
This is not a strict1,000-label learning comparison. Utility empirical resampling
bootstraps those same1,000 observed rows; distinguish it from G33 empirical motion
transfer along a new route. G40 holds route choice common across arms, so any GAF
effect in this round concerns motion conditional on the sampled route, not better
route selection.

While the first20/128 generation cells had closed, before any of their quality
outcomes were inspected, add a read-only final metadata audit of recorded
`code_abs_max`, decoder truncation and constant-progress fallback counts across
all arms/classes/cells. `scripts/g40_sampling_audit.py` reports fixed quantiles
and exceedance of the observed training maximum; the latter is descriptive, not
a validity threshold, sample exclusion or new primary endpoint. Run only after
all-stages closure, in a new small isolated postprocessing directory. No fits,
sampling or protocol decisions depend on its output. This checks whether the
pilot's prior-range issue reappears under prospective route/duration conditions.

Methods, data, protocol, introduction and literature edits are prepared under
`/tmp/cnr-v20-draft` only; the actual manuscript is unchanged. These drafts contain
no new G40 quality conclusions and need integration/review after complete evidence.

The additional presentation-only builder `thesis/tools/build_g40_supplement.py`
will render the fixed ten physical endpoints as both class-marginal W1 and
request-paired MAE, both native endpoints by all five classes, both fitted seeds,
per-episode coverage and diversity/decoder accounting. It reads only the verified
closed export. These are existing endpoints, not a new metric selection or
post-hoc exclusion rule. Short-trip support remains endpoint-specific.

Publication-layout preparation before outcome review: preserve the original six-panel
class overview, and add three page-width pairs containing the same six endpoints,
every class/arm, exact numbers and fixed color limits. Journey/GAF labels use larger
fonts and wrapped headings. The running immutable review snapshot is unchanged;
if needed these presentation-only sources run later in a new layout directory.
Selection of journey cases and all scientific results remain unchanged. Working
sources are also preserved under the local review workspace's
`work/thesis-v20-pre-evidence`, explicitly marked as unintegrated drafts.
