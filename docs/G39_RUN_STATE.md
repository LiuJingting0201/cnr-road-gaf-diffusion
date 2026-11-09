# G39 continuation entry point

This is the current authorized experiment round, superseding old E2/G35/G38
launch instructions. Read G39_PROTOCOL.md and G39_UTILITY_PROTOCOL.md first.

## Live, finite cloud ownership

Relative to the existing isolated experiment base:

| Directory | Supervisor at launch | Work | Dependency |
|---|---:|---|---|
| g39-job/g39-data |225896|14 diffusion fits,2RAW teachers,8SUMO references,112generation/evaluation cells|Running|
| g39u-job/g39u-data |226500|26 independent raw-series utility fits|G39 complete and GPUs free|
| g39f-job/g39f-data |226501|4 expanded route×prior-duration×motion generation/evaluation cells|Utility complete and GPUs free|
| g39m-job/g39m-data |226380|Existing-pool route-marginal mechanism|**Completed and reviewed**|

PIDs are launch receipts, not evidence of current liveness. Inspect complete.json,
failed.json, queue.log, training-progress.json, per-fit training.jsonl and workers
before acting. Do not start duplicates or rewrite the deployed source snapshots.
The supervisors are detached and their scientific dependencies run on the server;
they do not require an open local SSH connection. They stop on failures and bounded
resource/time limits. No desktop job-watching automation was created.

Main runtime snapshot is commit3fbf0d9 (321 hashed source/protocol files), follow-up
snapshots34311f1 (328 files). Later reporting scripts and prose are separate; don't
silently replace running source with latest Git. Old server project tree remains
untouched. Free-space reserve12GiB; output caps main24GiB/utility4GiB/factorial6GiB.

## What is already established

- Actual four-card progress, not just PIDs: first fits beyond5,400updates at the
  latest stored snapshot; no failure marker. Exact short-test initialization and
  denoising loss equality, nonzero auxiliary gradient, padding/EMA/export checks.
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
Keep the deployed g39/g39u/g39f snapshots and their existing order unchanged.
The current cloud supervisors do not implement performance optimization; this is
the next agent action after their completion and evidence review, not an already
launched job or an installed desktop automation.

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
