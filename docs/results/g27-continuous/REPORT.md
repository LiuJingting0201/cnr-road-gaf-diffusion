# G27 · Training a controllable complete-trip diffusion model

The default for within-class motion control is HM, the continuous route-progress generator conditioned on the physical mode. Six modes cross a training-defined mean-speed band with steady/variable four-second speed change. At inference use a specified supported mode or draw from a same-class, route-length-matched training prior. Route and class are given, while arrival time and the full sequence of along-road progress are generated. No future observation or classifier score is needed for drawing a trip.

Eight new SUMO episodes (278001–278008) on the same map provided 384 eligible requests, 48 per participant class. Each of the four new arms used two fits of 12,000 updates on the same 6,033 trips. Each fit supplied four paired draws per request. Metrics are class-macro, intervals resample eight episodes. Each arm's seed-42 first draw passed the road or pedestrian-sidewalk audit on all 384 requests. This check covers one draw per request.

| Arm | Physical mode hit | Longest cruise W1 (s) | Duration W1 (s) | Four-second speed change W1 (m/s) |
|---|---:|---:|---:|---:|
| H, plain | 37.7% | 9.21 | 12.54 | 0.259 |
| HM, requested mode | 71.1% | 8.79 | 13.38 | 0.256 |
| HV, mode + speed curve | 70.9% | 8.87 | 13.41 | 0.254 |
| HG, mode + speed GASF/GADF | 70.7% | 9.02 | 13.37 | 0.252 |

Mode hits use physical measurements on the generated trip. HM gained 33.3 percentage points over H (95% interval 31.0–35.3). Holding route, class, weights and noise fixed on 336 requests with both modes supported in a speed band, variable minus steady for HM was −24.30 s in longest cruise (−26.26 to −22.13), +0.671 m/s in four-second speed change (0.641 to 0.704), +6.11 s in duration and −0.327 m/s in mean speed. Among buses the cruise change was −6.39 s. The switch probes the generator on seven classes and has no paired observational counterfactual.

For HG, removing the hard blocks on the same requests reduced cruise W1 from 22.43 to 9.02 seconds. Against direct speed supervision, the GAF auxiliary changed cruise W1 by +0.154 s (−0.580 to +0.597), duration by −0.037 s (−0.074 to +0.012), and four-second speed-change W1 by −0.002 m/s (−0.004 to +0.001). HM vs H likewise had inconclusive marginal cruise and timing gains. The controlled motion response is the reproducible training benefit. The inherited spatial GAF remains a separate participant scorer. The new angular objective compares speed patterns during training and has yet to demonstrate added fidelity over direct speed supervision. Coverage is same-map, arrival-confirmed simulation, with limited pedestrian routes. Frozen tests were excluded.

Reproduction details are in docs/G26_PROTOCOL.md and docs/G27_PROTOCOL.md. The local continuous-mode add-on overlays the G22 generation kit and includes the four seed-42 arms. The delivery digest is recorded alongside the metrics. For the reported generator use the HM arm.
