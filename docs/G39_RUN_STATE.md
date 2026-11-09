# G39 continuation entry point

Latest override,2026-11-09: the entire recovered scientific batch is complete and
reviewed (112main cells,26utility fits,4factorial cells). See
`results/g39-native-alignment/completed-batch/INTERPRETATION.md` and the full
`verified-aggregates.json`; do not relaunch scientific queues. The user now
prioritizes the thesis/GAF mechanism. Performance has not been released or measured;
inspect the existing waiting controller before resuming it. The selected next
technical design is `G40_GAF_GENERATIVE_CONDITIONING_DESIGN.md`, not yet implemented
or launched. It extends the earlier tentative head-retention idea with encoders
trained for generation and a sampleable conditional code prior.

This is the current authorized experiment round, superseding old E2/G35/G38
launch instructions. Read G39_PROTOCOL.md and G39_UTILITY_PROTOCOL.md first.

## Live, finite cloud ownership

Relative to the existing isolated experiment base:

| Directory | Supervisor at launch | Work | Dependency |
|---|---:|---|---|
| g39r-job/g39-data |228063|Reuse14fits/2teachers/8references; complete112generation/evaluation cells|Completed and verified|
| g39ur-job/g39u-data |228066|26 independent raw-series utility fits|Completed and verified|
| g39fr-job/g39f-data |228067|4 expanded route×prior-duration×motion generation/evaluation cells|Completed and verified|
| g39m-job/g39m-data |226380|Existing-pool route-marginal mechanism|**Completed and reviewed**|

PIDs are launch receipts, not evidence of current liveness. Inspect complete.json,
failed.json, queue.log, training-progress.json, per-fit training.jsonl and workers
before acting. Do not start duplicates or rewrite the deployed source snapshots.
The supervisors are detached and their scientific dependencies run on the server;
they do not require an open local SSH connection. They stop on failures and bounded
resource/time limits. No desktop job-watching automation was created.

Original snapshots3fbf0d9(main) and34311f1(follow-ups) are preserved in the old
failed g39/g39u/g39f directories. Recovery orchestration is commit6bebd6a;
scientific scripts/protocols are copied and hash-checked unchanged. Do not rerun
old failed supervisors or delete their failed.json markers. New utility uses
--source ../g39r-job/g39-data; new factorial uses
--dependency ../g39ur-job/g39u-data. Free-space reserve12GiB; output caps
main24GiB/utility4GiB/factorial6GiB. See RECOVERY.md and recovery-live.json in
G39 results for failure, deployment, source checks and launch receipts.

## What is already established

- Latest stored check:14/14diffusion fits and2RAW teachers complete, artifact
  hashes and paired training checks verified. All8SUMO references completed.
  Original queue stopped on an artifact/provenance hash mix-up after4pools.
  Recovered main is now112/112complete, with generated artifacts, source models,
  catalog/evaluation/diagnostic links and exact route/duration pairing independently
  rechecked12:21UTC. Full560-class-row tables and interpretation are in
  `results/g39-native-alignment/completed-main`. SV speedW1+0.743%, acceleration
  +0.466% versus B; V's small acceleration gain is matched by SH. No overall
  GAF-specific motion-quality gain is established. No new failures or retraining.
  Utility26/26 fits have now completed; all model/prediction hashes, shared384-row
  evaluation identity/label order and recomputed macro/per-class F1 were verified.
  Mean macroF1 real91.25%, real+B93.17%, real+V94.10%, real+RAW94.06%: synthetic
  augmentation has utility here, but a GAF-specific advantage over RAW is not
  established. Full outcomes: `results/g39-native-alignment/completed-utility`.
  Factorial4/4 is complete; all10,784candidates and evaluation sources verified.
  Fixed joint route/T speed-score weighting changes speedW1−0.12% and acceleration
  −0.13%, improving1/4 and2/4cells respectively. Spatial/dual weighted endpoints
  worsen overall. No stable physical benefit from class-score ranking established.
  Performance still waits for its unreleased review gate; no benchmark has run.
  User-requested interim look on the first2complete OD cohorts is recorded in
  interim-two-cohorts/REPORT.md: GAF primary distances are slightly worse, with
  mixed small stopping effects. Full-batch interpretation is now recorded; original
  protocol and scientific snapshots remain unchanged.
- Full raw-series downstream-cache fixture passed without opening fresh outcomes.
- G35 train-parent vs planned utility-evaluation source-parent metadata overlap0
  in each class; nonbus OD cohorts separated. Public bus templates still shared.
- Route-marginal diagnostic reproduced old R/global controls exactly. Fixed-route
  spatial feedback: speedW1+2.73%, accelerationW1−2.77%, only2/4acceleration cells
  improve. This is mixed exposed-data evidence; duration remains uncontrolled.

## Next evidence review

1. Verify all completion receipts and hashes. If a stage fails, preserve its
   logs/outputs, diagnose before launching a new attempt; don't reinterpret partial
   results as full completion. Parent supervisor errors may surface after running
   child threads close, so inspect individual failed.json too.
2. Main report/complete.json and all-class-metrics.csv: K1 primary speed and
   acceleration, all-four secondary, every class/cohort, stopping/cruise,
   diversity/near-copy diagnostics, decode clamps and exact route/T pairing.
3. Compare B/S/V/SV/P/SH/RAW, recognizing RAW pretraining/architecture differences.
   Don't use teacher confidence or training loss to announce generation benefit.
4. Utility has13regimes×2downstream seeds and held-out last-two OD cohorts. Evaluate
   real+GAF against real-only, real+empirical, real+B and real+RAW; retain synthetic-
   only and all adverse per-class results. It cannot override bad physical quality.
5. Factorial is exposed G38 development data, frozen G35 generator,8candidates per
   available route, same expanded pool for all10rules. FixedRouteT holds the joint
   route/drawn-T distribution, not just its means. Do not compare its larger pool
   with historical K8 as though budgets matched, or call it in-loop guidance.
6. Commit/push checked aggregate receipts, full metric tables, plots and reasoning.
   Raw trajectories/checkpoints remain server/private artifacts only. Update thesis
   claims only after reviewing completed generation and utility evidence. Additional
   adaptive fits require a new bounded, documented mechanism hypothesis and a fresh
   confirmatory cohort; do not silently optimize on the now-exposed references.

The cloud scripts compute fixed summaries and proceed through predeclared controls.
Scientific interpretation, paper changes and publication of newly completed results
still require the next evidence-review pass; those are not already completed.

## Authorized next action after this batch

2026-11-09 user instruction: leave this batch running, then optimize performance.
Keep the recovered g39r/g39ur/g39fr snapshots and their existing order unchanged.
The receipt-verifier repair was necessary recovery, not a throughput optimization.
The scientific supervisors do not implement performance optimization. Candidate
code is now prepared separately in `g39perf-job`; its CPU-only synthetic input
fixture passed exact value/layout and RNG checks, with307 original Python source
hashes verified. No performance fit has run at this preparation checkpoint.
The waiting controller was launched12:14:25UTC as PID235671; live progress confirmed
`waiting_for_entire_G39_batch`. Receipt: `results/g39-performance/launch.json`.
This process is only waiting; the review gate remains closed.
`docs/G39_PERFORMANCE_PROTOCOL.md` defines a finite controller gated on all three
scientific completions **and an actual completed-batch review receipt**. Inspect
its launch/progress markers before any start; do not duplicate it. Preparation is
not a measured speedup or adoption. No desktop automation is installed.

1. Verify main, utility and factorial completion receipts, failures, artifact
   hashes and generated-quality results before starting the optimization stage.
2. Create a separate implementation/benchmark directory. Profile a fixed,
   training-only subset; never use held-out quality outcomes to tune performance.
   Bound the first benchmark round to60minutes, with no full scientific retraining.
3. First test caching invariant clock/route/context features and reducing repeated
   device-scalar reads and host/device transfers. Preserve the effective batch,
   selected examples, microbatch grouping, timestep/noise stream, loss, optimizer
   and EMA. Compare baseline and optimized outputs, gradients and parameter
   updates on identical inputs; document tolerances and any differences.
4. Measure warmed-up end-to-end updates/s, total wall time, peak device/host memory
   and GPU activity under matched conditions. A higher utilization percentage
   alone is not success. Retain baseline and failed/neutral variants.
5. Larger microbatches, mixed precision or compilation require separate numerical
   checks and explicit reporting; do not mix them into the unfinished G39 arms or
   assume they preserve exact RNG allocation. Select further work from measured
   bottlenecks, with bounded comparisons rather than an open-ended tuning sweep.
6. Commit/push implementation, benchmark settings and measured results. Adopt only
   validated changes for future runs; preserve all original G39 evidence.

Baseline observation: `docs/results/g39-native-alignment/UTILIZATION_AUDIT.md`.

2026-11-09 latest user direction: examine the GAF integration mechanism, complete
this batch, then design the next scientific round. Code-level hypotheses and
representation/differentiability limitations are in
`docs/GAF_INTEGRATION_REVIEW_PENDING_G39.md`. That note is not a locked protocol;
no new scientific arm or thesis claim is authorized by the note itself.
