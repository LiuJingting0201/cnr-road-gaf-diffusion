# G38 running: route models trained, generation-quality evaluation pending

2026-11-09. Protocol and full finite dependency queue were committed/pushed as2199ad4 before execution. Isolated cloud directory `g38-job`; old worktrees and frozen tests unchanged. The queue launches independent CPU route construction/reference simulation and four GPU fits, then four diffusion-generation cells and evaluation without a local connection.

## Completed route training

The input has29,895 training legs from24,073 parents. The first-passage road-choice interface accepts23,696 legs/parents with546,363 branching decisions. Development support is5,748 legs/parents and132,341 decisions, of which2,994 legs have a class/OD edge pair absent from accepted training. All five classes remain. Rejections are fully retained in `policy-data/complete.json`:2,763 training legs have fewer than two edges,3,403 have non-first-passage segments and33 have no branching decision. This policy does not cover journeys that reach the same goal edge and then leave it before the final arrival; that coverage limitation is material and not a claim that those journeys are invalid.

Four fixed fits (linear/neural × seeds42/43) ran on separate L20 devices,4,000 updates each. Their measured training times were16.12–17.92seconds; these are small road-choice networks, not new diffusion fits. Existing full-trajectory diffusion and GAF models remain frozen to isolate route learning. Input preparation took79.53seconds. Every fit has a final checkpoint receipt, SHA256 and complete scalar training trace; no validation-based checkpoint selection.

Class-macro development branching-decision results, equal parent then leg then decision within class:

| Policy | Accuracy | NLL | Brier score | ECE (10 bins) |
|---|---:|---:|---:|---:|
| Uniform reachable successor |45.28%|0.7574|0.5259|0.0347|
| Static shortest remaining cost |94.90%|0.5867|0.3772|0.3757|
| Linear42 |90.59%|0.2752|0.1573|0.0748|
| Linear43 |90.58%|0.2747|0.1568|0.0756|
| Neural42 |96.76%|0.0966|0.0518|0.0096|
| Neural43 |96.78%|0.0959|0.0515|0.0072|

The strong shortest-cost accuracy is important: most held-out decisions already align with static routing. The neural improvement is about1.9percentage points, not evidence that every route is better. ShortestCost probabilities use a fixed1km softmax scale, so its NLL/calibration gap is not a temperature-matched learned-baseline comparison; the fitted linear control is reported alongside it. Uniform accuracy uses deterministic tie-breaking among equal-probability successors. These are road-choice prediction metrics, **not GAF classifier accuracy, whole-route accuracy or trajectory-quality results**. Per-class and seen/unseen-OD values are in `policy-evaluation/complete.json`; buses share public service templates across splits.

## Running and resource scope

Two new SUMO episodes and two map-only route catalogs proceed on CPU. Four sampling cells are queued after catalogs complete; physical and route-distribution evaluation follows completed generation and references. Ten selectors share each8-candidate pool, including learned routing without GAF and its spatial/speed/combined GAF extensions. Unequal candidates per route are corrected before stochastic selection. No evaluation result or GAF benefit claim is available at this checkpoint.

At launch, approximately50GiB was free. The entire new root has a12GiB output limit and12GiB minimum-free guard, plus a12hour timeout. The user-facing initial runtime estimate is1–2hours for the whole queue, informed by the previous65minute experiment; it is an estimate, not a guarantee. GPU idle intervals during CPU work are expected. The two simulation clocks have been observed advancing; background teleport counts are present and must not be misreported as zero incidents or automatically attributed to target trajectories. Reference acceptance retains the existing explicit quality checks and all request denominators.

Runtime hashes of all five new scripts match local committed sources. Checks passed for masked/ordered goal decisions, permutation invariance, candidate-multiplicity correction, coupled random selection, common fallback, edge-occurrence mass/JS/native weighting, eight inherited reference cases and33 geometry cases. Raw arrays, per-participant metadata, candidate tracks and model weights remain on the cloud. The checked resource/stage snapshot is `current.json`; its timestamp is explicit and is not a live dashboard.
