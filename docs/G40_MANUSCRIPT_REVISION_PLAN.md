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
