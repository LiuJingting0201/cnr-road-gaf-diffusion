# G40 current state — complete and delivered

2026-11-10 Beijing time. All 16 motion fits, 12 learned priors, eight references, 128 generation/evaluation cells and 30 utility fits are complete. Full evidence has been reviewed; see [REPORT](results/g40-generative-conditioning/REPORT.md). Independent recomputation and condition/cohort summaries are retained under `completed/`. No additional training is queued. Thesis v20 PDF/ZIP are complete under `deliverables/thesis-v20`: 113 PDF pages, 69 main pages, 34 figures, 29 tables and 54 cited entries. All-page and detailed figure/table inspection passed; clean ZIP rebuild has identical page text. This round is closed; no further model search or watcher is queued.

Primary conclusion: no overall GAF motion-quality gain; spatial-only augmentation has a small pooled, cohort-dependent effect. Preserve all controls and adverse findings. Do not restart old queues, open old frozen tests or search parameters for a favorable outcome.

---
Historical execution checkpoints (superseded by the state above):

# G40 resumption entry — 2026-11-10 Beijing

Latest user instruction: continue experiments, revise the thesis after evidence review.
G39 scientific batch and performance benchmark are closed; do not relaunch them.
Read G40_EXECUTION_PROTOCOL.md and G40_GAF_GENERATIVE_CONDITIONING_DESIGN.md.

## Immutable cloud snapshots

Base `/workspace/cnr-experiments/gaf-inloop-v1`:

- `g40a-job/g40a-data`: preparation completed302.204s,500training parents audited.
  Sourcee960322,launchPID240193. All four prefix feature errors0; source byte fidelity
  and lossless archives verified. See results/g40-preparation/REPORT.md. Review
  receipt releases finite pilots only; no formal generation-quality conclusion.
- `g40p-job/g40p-data`: four1000-update pilots B/F-SV/J-SV/J-RAW, seed42,
  three1000-update priors plus B zero prior, implementation/training-side diagnostics.
  Source2d3efc0,launchPID240691. Four fits/priors/diagnostics completed, but supervisor failed on a source-hash/local-artifact mix-up. Preserve failure; do not relaunch.
  Inspect queue.log, failed.json, model training.jsonl, prior and diagnostic receipts.
- `g40pr-job`: newly implemented prior-only recovery, source4821c56; completed70.685s as PID241619. Four verified motion fits reused; declared5000-update prior budget restores ordinary code ranges on identical25-parent diagnostics. See results/g40-pilots/REPORT.md. The1000-update prior sampled code magnitudes119–178 and worsened diagnostic noise MSE; full-budget prerequisite has subsequently passed; weak condition-use remains a limitation, not a success claim.
- `g40-job`: formal snapshot launched source29f34d7,PID242214,manifest6da6942cbda67007fe7941a84ca5db33760a53bbd930658b5b4b63d64503573a. Launch free40.41GiB. Queue includes
  eight arms×two seeds×5000updates,12priors×5000,8newSUMOreferences,128direct
  generation/evaluation cells and30independent utility fits. Has actual pilot
  review tied to the formal source manifest; fresh G39B start, not pilot checkpoints.

- `g40-review-job/g40-review-data`: sourcef434847,PID242645; waiting for formal
  all-stages-complete.json, then verifies every source artifact and recomputes
  downstream predictions, produces fixed five-class journey/GAF figures plus
  all-arm aggregate plots/tables. This does not interpret results or edit thesis
  prose automatically; a review is still required. Max24h wait/1h active/free12GiB.
- `g40m-job/g40m-data`: source26e8090,PID243107,manifest
  afcd4bb7c339e25b5c0c3bcb2eaf534aec0e38a946d33fe214d81ff43698ceb2;
  361 files verified, observed waiting_for_closed_science_and_visuals. Fixed final
  condition-use audit (25 training parents, five noise levels, four code variants,
  all16models), no fitting. Starts only after g40-review complete; max30active
  minutes/1GiB/free12GiB. Review separately from independent synthesis results.

No desktop milestone automation. Existing detached cloud supervisors own their
children and stop on bounds/failure. Preserve all old work trees/failure receipts.
Do not start a duplicate queue. Historical frozen tests stay closed. Public repository
gets code, aggregate metrics, figures and prose; raw data/weights remain private.

Native input caching is not adopted; exact evaluation IO optimization is adopted.
All five classes and adverse endpoints remain visible. Main synthesis uses sampled
codes, never the true target GAF. Training reconstruction/conditioning diagnostics
must be labelled in-sample and kept separate from prospective synthesis results.

After completion, independently verify generated/source/metric receipts, compare
primary speed/acceleration and all motion/diversity controls, then update LaTeX,
trajectory/GAF figures, abstract and discussion together and render-review the PDF.

Formal launch receipt does not prove progress or completion. Inspect live logs before reporting. The cloud queue includes utility after128cells close; all-stages-complete.json is required before final scientific interpretation and thesis revision. Main complete.json closes the primary stage only and remains immutable.

Latest live check at UTC2026-11-09T18:32:49: all16motion fits and16prior receipts
complete. training-complete.json confirms common initialization/tokens/RNG/stats
and no pilot checkpoint reuse. 6/8references and all8catalogs complete, no failures,
35.43GiBfree; the remaining late-window references are400206/400208. Main supervisor
waits for both before generation. Do not start another sampler while it owns the
queue. Training curves/configs/costs are public in results/g40-generative-conditioning/
training (83628e3). Fit22.90–27.46minutes each under concurrent L20 operation; this
is continuation cost, not total pretraining or sampling. All eight requests were
frozen before outcomes. Training closure is not generation-quality evidence.

Manuscript revision map is G40_MANUSCRIPT_REVISION_PLAN.md. Its source-level
clarifications include G35 subsecond speed fallback and exact route-policy scoring.
Reproducible package builder passed the unchanged v19 clean-build preflight;
thesis prose, figures and delivered version are still unchanged pending evidence.

UTC2026-11-09T19:30:19: all8reference episodes and8catalogs are complete;
38/128 generation/evaluation cells closed, no failures,34.75GiBfree. The same
four-GPU sampler queue continues. No generation-quality values have been read
for scientific decisions; utility remains downstream of primary closure.

Staged only, not launched: `g40-sampling-audit-job` contains the registered
metadata audit (source script978d8b9d...); run after all-stages closure.
`g40-layout-job` contains three presentation-only sources from2f2f1ce: larger
journey/GAF labels, three fixed endpoint-pair class panels, and the full numerical
supplement. Its dependencies are the immutable g40-review-job snapshot, whose
manifest SHA is2a5e508d139ab3f9baf3e1e1907129e75419dafe5f51665977412b8917766e8e.
Run only after review closure, into new layout outputs; compare all fixed-case
track/image hashes and numerical matrices with the original review assets. Both
staging receipts are in results/g40-postprocessing. No additional model fits.

UTC2026-11-09T20:29:50:106/128evaluationcells,0utilityfits,no failures,33.73GiBfree.
The existing read-only SSH watcher remains active in desktop process session80619.
