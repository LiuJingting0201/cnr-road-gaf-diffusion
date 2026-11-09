# G38 completed: learned routing helps edge-use agreement; GAF remains a trade-off

2026-11-09. The finite queue completed in 65.22 minutes. All four generation/evaluation cells are present, with ten matched selectors per cell. The original pre-execution protocol is unchanged. Full class/endpoint results are in `generation-complete.json`; `generation-macro.csv` contains every cell rather than only an average. Launch-time training/resource evidence remains in `LAUNCH_STATUS.md`.

## What changed

The four route-policy fits completed as previously reported (neural development decision accuracy96.76/96.78%). Existing native-clock diffusion and spatial/speed GAF networks were frozen. This isolates the learned route preference and its GAF reweighting; it is not new diffusion training or sampling-time guidance. Two new SUMO scenes share public OD/service plans. Models42/43 pair route and diffusion seeds; four cells are not four independent scenes.

## Complete comparison

The table is a descriptive, equally weighted average across the two scenes and two model seeds, after class-macro averaging within each cell. Weighted native W1 measures the full probability-weighted candidate distribution; it is distinct from the single sampled output. Both are retained in the full results. Lower W1/JS is better; effective route count describes retained route diversity rather than correctness.

| Selector | Speed W1 (m/s) | Acceleration W1 (m/s²) | Edge-use JS | Effective routes |
|---|---:|---:|---:|---:|
| Uniform | 1.0511 | 0.1758 | 0.10177 | 3.454 |
| Shortest | 1.2033 | 0.1744 | 0.04000 | 1.000 |
| Linear | 1.1438 | 0.1724 | 0.04931 | 1.656 |
| R | 1.1673 | 0.1719 | 0.03458 | 1.176 |
| S | 1.1150 | 0.1770 | 0.09677 | 2.748 |
| V | 1.2442 | 0.1747 | 0.10120 | 3.015 |
| SV | 1.1952 | 0.1721 | 0.09422 | 2.672 |
| R+S | 1.2125 | 0.1677 | 0.03732 | 1.159 |
| R+V | 1.2871 | 0.1716 | 0.03626 | 1.195 |
| R+SV | 1.2237 | 0.1663 | 0.03473 | 1.175 |

R is the neural route policy. S/V are spatial/speed GAF, SV averages their log scores. All methods share the same eight-candidate pool, fixed coefficient1, and candidate-count correction per route. No strength or checkpoint is selected after viewing these outcomes.

## Interpretation and coverage

R reduces average edge-use JS versus the shortest-route control by13.57%, while retaining about1.18 effective routes versus1.00 for Shortest. Against the same Shortest control, mean speed W1 decreases2.99% (all four cells) and acceleration W1 decreases1.40% (three of four cells). This is finite-catalog distributional agreement, not proof of correct individual route choice or population route-frequency calibration. The uniform catalog policy has substantially more route diversity and lower aggregate speed W1, so learned routing is not an across-the-board improvement.

Adding spatial GAF to R lowers aggregate acceleration W1 by2.43% but raises speed W1 by3.87% and edge-use JS by7.95%. The averages conceal a scene reversal: acceleration improves8.09/8.34% in scene380201 and worsens4.09/8.96% in380202. R+SV likewise improves acceleration8.05/10.21% in the first scene but worsens0.86/10.80% in the second. Consequently this experiment does not establish a stable incremental GAF gain beyond learned routing. Full adverse endpoints and all five classes remain reported; no new GAF-weight sweep is started.

Each scene requests100 journeys:24 public bus plans and19 for each other class. The route catalog supports98, including22 buses; the two unsupported bus plans remain in the denominator. Paired physical/route support is97 in380201 and92 in380202; the later scene has19 supported buses and16 taxi/uber references. The catalog contains an exact reference route for64/97 and61/92 paired requests. Valid alternate routes are not errors solely because they differ from the reference. Per-class nearest-catalog overlap and all support counts remain in the full JSON.

The old frozen test is not used. These independent simulator realizations still share the same map and public service/OD templates; the simulator references do not provide a general stochastic within-OD route-frequency ground truth. The existing v19 thesis is not silently rewritten as universal GAF success. This result and the full closed snapshot are included in the2026-11-09 GitHub archive.
