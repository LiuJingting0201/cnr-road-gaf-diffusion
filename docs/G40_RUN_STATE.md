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

Latest live check at formal elapsed~18minutes: B42/B43 3700/3600updates; F-SV42/43 3200/3300. Four GPU processes active,2/8references and1/8catalog complete, no failures. First eight request manifests are frozen and train-parent checks passed. These are progress counts, not completed-model quality evidence.
