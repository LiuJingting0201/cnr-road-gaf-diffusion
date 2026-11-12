# Porto scoring audit and repeat fit

Freeze the original seed-42 long and mixed weights. Replay all four train/evaluation pairs with their original 3,000 requests and noise. Verify the archived four scores. Evaluate both training arms on identical reference trajectories using shared bounds, bins and request counts. Use the previously specified full and training 1%–99% grids at 16 by 16, with top 20 as the primary local pattern diagnostic. Include overflow in core density and exclude it from core hotspot rankings. Compare two disjoint real samples of equal size from each evaluation pool, and compare generated versus real at that same size. Summarize lengths in kilometres and overlay empirical distributions. Plot observed and generated point densities on the same scale.

Repeat the stabilized mixed-population fit (the broader taxi population) with seed 43. Keep data, 200 epochs, global batch 1,024, four GPUs, validation noise, checkpoint selection, holdout request IDs and generation noise fixed. Evaluate it on both original populations. All evaluation cohorts were previously scored. This is a repeatability check, with no selection by these scores.

On a fresh isolated server directory copy porto.py, porto_validation.py and run_porto_metric_review.sh, then run bash run_porto_metric_review.sh. The script reads the archived datasets and weights through symlinks.

## Measurement provenance

| Quantity | Published description | Local operation and unresolved details |
|---|---|---|
| Density | Point occupancy on a 16 by 16 grid, JS divergence | Same grid resolution. Boundaries affect the score. The common audit grid takes bounds from the mixed training cohort. |
| Trip | Joint origin and destination frequencies, JS divergence | Count one ordered cell pair per journey. Both samples use the same mapping. |
| Length | Distance accumulated between consecutive coordinates, described as Euclidean | Geodesic metres, 20 training-quantile bins. Also calculate planar metres on a local equirectangular projection using the same bin edges. Paper does not specify Porto bin edges or projection. |
| Pattern | F1 of the most frequent n grid cells | Single-draw top 20. With 20 cells per set the score changes in units of 0.05. The paper does not identify n or averaging for Porto. |
| Generation requests | Original conditions include distance, time, sample count and OD cells | The conditional length and OD scores measure adherence when supplied this information. They cannot measure unconditional demand generation. |

The cited LDPTrace study specifies the top 100 frequent **patterns**, reports repeated synthesis, and defines hotspot cells separately. Its settings do not identify the n or aggregation used in DiffTraj Table 5. [DiffTraj, Appendix B.2 and C.1](https://proceedings.neurips.cc/paper_files/paper/2023/file/cd9b4a28fb9eebe0430c3312a4898a41-Paper-Conference.pdf) · [LDPTrace, §6.2–6.3](https://arxiv.org/pdf/2302.06180) · [DiffTraj release](https://github.com/Yasoz/DiffTraj/tree/27d25a9eb4cb94bcbe76068d5c8825f713a7ac2c)

The paired diagnostic splits the same 3,000 trips twenty times into two disjoint halves of 1,500. It compares observed A with observed B and observed A with generated A. Report medians and 10th–90th percentiles over these splits. These describe within-cohort sampling variation. The independent training repeat uses the broader mixed arm only. Both training arms use the mixed-arm training bounds and mixed-arm length bins for the cross-arm comparison. In the full grid, out-of-range coordinates clip to the edge as in the historical implementation. In the core grid they map to an overflow cell in distribution metrics and are excluded from hotspot rankings.

For each completed score directory run the server environment's Python with PYTHONPATH including the DiffTraj release and run `porto_metric_review.py DIRECTORY`. Copy its aggregate audit.json as DIRECTORY.json to the report folder and run `python plot_porto_review.py REPORT_FOLDER` locally. Private draws remain on the server.

Each training arm contains 46,470 trips at equal compute. The mixed training set is a uniform sample of 1,391,486 eligible training trips. Both models use the same 512 disjoint development trips from long-trip taxis for validation noise loss. The checkpoint choice therefore weights long-trip validation cases. The GPS sample interval is 15 seconds: 120 positions span 29 min 45 s, and applying the cutoff changes the travel population. [Cutoff protocol and author clarifications](../porto-cutoff/PROTOCOL.md) · [Stabilized training protocol](../porto-stabilized/PROTOCOL.md)
