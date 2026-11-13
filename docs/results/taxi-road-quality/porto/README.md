# Direct road audit of the matched Porto taxi draws

The two 10,000-training-trip Porto taxi fits used disjoint training and held-out taxis. Each seed generated one trajectory for the same 1,000 requests. Observed GPS positions retain the native 15-second clock. The saved 200-point generation is restored to the request's original point count by the DiffTraj evaluation interpolator. The experiment uses every request.

Valhalla 3.9.1 matches the original input coordinates to the archived directed Porto road graph with taxi costing, 50 m search radius and 10 m GPS accuracy. A supported interval has both endpoints matched within 50 m, no intervening matcher break flag and forward progress along the matched path (1 m rounding tolerance). A complete trip requires support on every interval. The middle 80% uses intervals with midpoints between 10% and 90% of elapsed time. The additional speed screen uses the distance along the matched directed path with a 40 m/s ceiling. The shape match ignores observed time during path search. Coverage uses the extract's rectangular extent; roads and survey dates can differ from conditions during GPS collection. Continuous support means the matcher found a plausible directed path between observations. Sparse GPS does not identify the exact intervening path.

| Measures (% unless counts are shown) | Observed | Generated 42 | Generated 43 |
|---|---:|---:|---:|
| Entire trip connected, out of 1,000 | 650 | 2 | 0 |
| Middle 80% entirely connected, out of 1,000 | 707 | 7 | 2 |
| Connected intervals, all 188,173 | 89.61 | 58.79 | 50.10 |
| Connected intervals, middle 150,647 | 88.74 | 58.10 | 49.78 |
| Entire trip connected and speed-compatible, out of 1,000 | 489 | 2 | 0 |
| Intervals connected and speed-compatible | 89.25 | 56.00 | 46.86 |
| Original points projected within 50 m | 90.02 | 71.57 | 64.24 |
| Trips entirely within map bounds | 951 | 937 | 943 |
| In-box trips connected from first to last | 650/951 | 2/937 | 0/943 |

Request order is zero-based and identical in all three cohorts. The compressed CSV records only indices and measures. In the CSV, `walk` denotes whole-trip continuity, `reasonable` counts speed-compatible connected intervals, and `passed` denotes whole-trip continuity with speed compatibility. Whole-trace matcher errors receive zero supported intervals. The paired table and its aggregate counts were checked independently. A second run reproduced all 3,000 per-request records.

The Porto map source is the archived BBBike extract, SHA-256 `dec551eb4178c97593d97af1aafe3c5b702c1acd6baabe830f9b609b146d8676`. Use the released Valhalla graph's configuration and original PBF from `road-check-v1/maps/Porto.pbf`. Run `taxi_porto_road_quality.py` with the matched Porto observation/draw pack, graph configuration, source PBF and an output directory. Valhalla and the upstream DiffTraj interpolation function must be available on the module search path. [Road evaluation protocol](../../road-feasibility/PROTOCOL.md)
