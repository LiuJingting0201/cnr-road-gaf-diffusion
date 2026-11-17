# Does velocity diffusion lose direction?

The denoiser predicts noise for both signed velocity components. The request contains origin and destination cells. Its training loss averages componentwise noise errors and has no extra term on the cumulative displacement. We audited existing true-label draws from the same 3,000 validation requests for each seed. All reference substitutions below are retrospective diagnostics.

| Median or pooled statistic | Coordinates 42 / 43 | Velocity 42 / 43 |
|---|---:|---:|
| Net bearing error (deg) | 3.29 / 3.33 | 4.08 / 4.18 |
| Absolute straight-line displacement length error (m) | 147 / 147 | 194 / 186 |
| Start error (m) | 193 / 195 | 198 / 197 |
| Net displacement vector error (m) | 268 / 269 | 333 / 334 |
| End error (m) | 195 / 201 | 342 / 350 |
| Angular fraction of squared displacement vector error | .472 / .456 | .432 / .465 |

For observed and generated net displacement lengths a and b and angular difference theta, squared vector error is (a-b)^2 + 2ab(1-cos(theta)). This identity separates magnitude and direction for net displacement. It does not partition cell misses. At the single-interval scale, the median heading difference for intervals where both trips moved above 1 m/s was 39/39 degrees for coordinates and 41/40 degrees for velocities. Observed and sampled trips can take different routes.

Keeping the generated start, replacing only the velocity model's net bearing with the observed bearing changed destination hits from 38.2/37.7% to 43.8/42.6%. Replacing only the straight-line displacement magnitude yielded 46.1/44.2%. With only the start replaced, hits were 40.8/38.2%. These changes are non-additive.

The endpoint error is the vector sum of start error and displacement error. The cross term reduced mean squared endpoint error relative to the sum of those two squared errors by 64/64% in coordinate diffusion and 22/20% in velocity diffusion. Part of this cancellation follows algebraically because displacement uses the generated start. It describes the output geometry and does not isolate a mechanism learned during training.

The validation draws suggest testing feedback on the full cumulative displacement vector or the integrated endpoint, measured alongside the existing speed and class-control outcomes. Only a matched retraining comparison can identify a benefit from that signal. [Aggregate results by class](direction-debug.json)
