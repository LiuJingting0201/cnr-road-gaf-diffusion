# Complete G39 route × duration mechanism diagnostic

All four cells and ten equal-pool policies are retained. Negative changes mean lower distribution distance; effective-route count is diversity, not an error.

| Policy | Weighted speed W1 | Δ% | Weighted acceleration W1 | Δ% | Lower speed / acceleration cells | Route JS Δ% |
|---|---:|---:|---:|---:|---|---:|
| R | 1.257966 | +0.00 | 0.173868 | +0.00 | 0/4 / 0/4 | +0.00 |
| Global-S | 1.341112 | +6.61 | 0.177295 | +1.97 | 0/4 / 1/4 | +1.80 |
| Global-V | 1.287631 | +2.36 | 0.172354 | -0.87 | 1/4 / 2/4 | +0.79 |
| Global-SV | 1.311715 | +4.27 | 0.175276 | +0.81 | 1/4 / 2/4 | -0.96 |
| FixedRoute-S | 1.319024 | +4.85 | 0.177661 | +2.18 | 0/4 / 1/4 | +0.00 |
| FixedRoute-V | 1.291042 | +2.63 | 0.172626 | -0.71 | 1/4 / 2/4 | +0.00 |
| FixedRoute-SV | 1.304682 | +3.71 | 0.175616 | +1.01 | 1/4 / 2/4 | +0.00 |
| FixedRouteT-S | 1.278069 | +1.60 | 0.176029 | +1.24 | 0/4 / 1/4 | +0.00 |
| FixedRouteT-V | 1.256438 | -0.12 | 0.173637 | -0.13 | 1/4 / 2/4 | +0.00 |
| FixedRouteT-SV | 1.269342 | +0.90 | 0.174921 | +0.61 | 0/4 / 2/4 | +0.00 |

FixedRouteT preserves the joint route and drawn-duration distribution to numerical precision; it does not force the same individual selected route/duration under every policy. Weighted native metrics isolate this distributional mechanism. Selected K1 speed/acceleration, journey/stopping endpoints and every class remain in the complete JSON/CSV; do not reinterpret selected-duration differences as a fixed-duration intervention.

These are two already-exposed OD scenes and two noise seeds, with repeated bus services. No population significance or fresh-replication claim. The larger factorial pool is not an equal-budget comparison with historical K8.
