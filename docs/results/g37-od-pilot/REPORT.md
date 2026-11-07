# G37: OD route-and-motion development pilot

Completed 2026-11-07. The finite cloud queue finished in **350.07 seconds**, including route construction (150.78 s), generation/scoring (143.59 s), evaluation and supervisor overhead. No new simulation or model fitting was performed. This is an OD feasibility and mechanism experiment using existing diffusion/GAF models, not a new independent test.

## What completed

All 20 predeclared OD requests (four per vehicle class) produced candidates. Nineteen requests had multiple supported routes within the bounded search. The catalog contains 72 routes: bus 15, motorcycle 15, private car 16, bicycle 15, taxi/ride-hailing 11. The one single-route taxi request does not prove no alternative exists outside the search budget.

Generation received only endpoints, class, map and ordered public stops for buses; interior reference edges, future motion and reference duration were removed. Directed permitted-road proposals and the geometry constructor enforce route connectivity, endpoint conditions and bus stop anchors. These are structural checks, not proof of collision avoidance, traffic-rule timing or service dwell compliance.

The single-route arm generated eight motion candidates on the minimum-length supported catalog route. The multi-route arm distributed eight candidates round-robin over up to four routes. Total: **320 candidate trajectories and 32,000 denoiser forward calls**. Shared route/candidate slots were checked for exact trajectory equality. Equal candidate and forward counts do not imply equal wall time or sequence-token budget because route lengths change sampled durations.

All 20 requests also have reference support for the physical comparison. References were opened only in the separate evaluation action. They are reused G36 development realizations, one per OD, so this cannot establish route-choice probabilities or an optimal itinerary.

## Physical results

Class-macro distribution distances to the existing SUMO references; lower is better. Every cell covers the same five classes and 20 requests. Uniform is the inherited deterministic random-candidate selector; S/V/SV are spatial/speed/combined GAF selection from the same pool.

| Candidate pool | Selection | Native speed W1 (m/s) | Native acceleration W1 (m/s²) |
|---|---|---:|---:|
| Single route | Uniform | 1.2257 | 0.1872 |
| Single route | S | 1.3686 | 0.1699 |
| Single route | V | 1.1905 | 0.1950 |
| Single route | SV | 1.0789 | 0.1412 |
| Multiple routes | Uniform | 1.2700 | 0.1937 |
| Multiple routes | S | 1.6451 | 0.1529 |
| Multiple routes | V | 1.4349 | 0.2100 |
| Multiple routes | SV | 1.3936 | 0.1687 |

Within the multi-route pool, S reduces acceleration distance by **21.1%** but increases speed distance by **29.5%** versus Uniform. V increases both distances (+13.0%, +8.4%). SV reduces acceleration distance by 12.9% but increases speed distance by 9.7%. Moreover, the single-route SV arm has lower distances than multi-route SV for both endpoints. Thus neither extra routes nor GAF selection has established an overall quality advantage.

All class-level changes below are relative to multi-route Uniform, in percent (negative means lower error). Four requests per class are insufficient for confirmatory claims; these descriptive values are not a basis for selecting a different policy for each class.

| Class | S speed / acceleration | V speed / acceleration | SV speed / acceleration |
|---|---:|---:|---:|
| Bus | +73.9 / +22.2 | −75.4 / +4.2 | −74.1 / +3.9 |
| Motorcycle | +46.5 / +17.2 | +155.9 / +77.7 | +17.2 / +9.9 |
| Private car | +56.4 / −18.6 | +35.2 / +26.2 | +284.7 / +52.7 |
| Bicycle | −37.4 / −63.7 | +6.0 / −25.8 | −37.4 / −63.7 |
| Taxi / ride-hailing | −5.1 / −4.2 | −12.9 / −12.5 | −34.5 / −12.4 |

All ten physical-summary endpoints, including adverse results, are retained with per-class W1/MAE and denominators in [evaluation/complete.json](evaluation/complete.json). An alternative itinerary is not scored as wrong simply for differing from the reference. Aggregate motion discrepancies measure agreement with this reference cohort, not unique-trip correctness.

## What GAF actually changed

Compared with Uniform in the multi-route pool, S changed the selected route for **14/20** requests, V for **9/20**, and SV for **12/20**. These are effects on selection, not evidence that the new routes are preferable. Each pool already contained eight distinct spatial images in both arms; increasing route diversity was not needed merely to avoid identical images in this experiment.

The first supported OD per class was fixed as the illustration rule before rendering. All five examples and all selectors are retained in `figures/routes-0.png` through `routes-4.png` and `figures/gaf-0.png` through `gaf-4.png`. Every plotted spatial image matches the image hash used during scoring; speed images use the same transform with input normalization undone. Coincident curves or repeated images represent actual same-candidate selections. The bicycle example is illustrative, not representative of the other four classes.

## Interpretation and next experiment

The useful result is an operational extension from fixed-route motion generation to **OD-conditioned route proposals followed by diffusion motion synthesis**. Routes are currently graph-algorithm proposals, not a learned diffusion route policy. GAF can alter the route selected, but class confidence alone does not identify plausible route preferences; the speed/acceleration trade-off must remain explicit.

Before a stronger thesis claim, fit route preferences using training-side observations only, compare against a matched non-GAF route selector, and preregister evaluation with independent OD/episode realizations. Measure route-choice distributions, detour, diversity and motion quality together. Bus service-stop conditions must remain separate from unrestricted OD choice. A single realized SUMO trip per OD cannot validate the probability of alternative routes. No new training queue or thesis-result rewrite is claimed by this report.

## Verification and execution record

- Existing geometry constructor: 33 synthetic checks passed; scope is recorded in `geometry-checks.json`.
- Existing evaluation checks passed for exact polylines, fractional clocks, time weighting, short support and unequal candidate/request weights.
- Route-proposal checks covered directed alternatives, waypoints, redaction and forbidden edges before launch.
- Local route/generation/plot script hashes match the server result receipts; the evaluation input generation receipt hash matches exactly.
- Cloud completion is verified; no queue failure marker. Finite limits were two hours, 8 GiB output and at least 12 GiB free space. Approximately 50 GiB remained at the final progress snapshot.
- The first stage-two deployment attempt backgrounded its input stream and produced an empty archive. It did not launch the queue. Upload and detached launch were separated and then succeeded; no duplicate queue was launched.
- Local result extraction first encountered an older Python `tarfile` API lacking `filter=`. It was retried with explicit path/type validation; server artifacts were unaffected.
- Raw coordinates, candidate-level records and weights stay on the server. Only scripts, aggregate receipts and figures are published. Historical results and frozen tests were not modified.
