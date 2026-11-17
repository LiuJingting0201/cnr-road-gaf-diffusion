# Using one generated path and another generated speed profile

For each of the 3,000 previously selected validation requests, take the coordinate model's sampled polyline and the original velocity model's sampled speed sequence. Both draws use the same supplied label, request and noise. Normalize cumulative distance along the velocity draw to [0,1] and interpolate the coordinate polyline at the resulting fractions of its arc length. This construction retains the coordinate draw's start and end exactly. All reported motion metrics are recomputed from the final 200 positions. Two trained generators are required.

| Measure (seeds 42 / 43) | Coordinates | Velocity | Retimed coordinates |
|---|---:|---:|---:|
| Requested destination hit | 71.7 / 70.5% | 38.2 / 37.7% | 71.7 / 70.5% |
| Speed W1 (m/s) | 1.213 / 1.080 | .212 / .179 | .793 / .627 |
| Acceleration W1 (m/s²) | .181 / .190 | .027 / .032 | .043 / .043 |
| Bus minus bicycle, low-speed share (pp) | 2.54 / 2.90 | 26.08 / 25.09 | 26.11 / 25.09 |
| Bus minus bicycle, low-speed entries / 10 min | .54 / .74 | 5.40 / 5.13 | 5.35 / 5.07 |

The final two rows use the same 53 matched pairs with similar requested mean speeds. SUMO differences were +39.62 percentage points and +7.87 entries. The construction preserves most of this learned label response. Absolute speed W1 is intermediate. In continuous arc length the transferred speeds are multiplied by route length divided by the integral of sampled speed. The final 200-position evaluation also measures straight chords across bends. The experiment does not separate these two effects. Destination accuracy is inherited by construction. Individual stop locations and compliance with the road network require separate checks. [Per-class metrics and scales](metrics.json)
