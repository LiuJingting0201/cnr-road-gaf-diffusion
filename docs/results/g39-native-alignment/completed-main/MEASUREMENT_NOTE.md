# Coordinate precision: a bounded measurement audit

This note does not recompute, replace or reinterpret the fixed G39 primary
endpoints as a successful result. It records a measurement assumption to check
when locking subsequent experiments.

The source SUMO configuration has no explicit precision override. In the first1000
vehicle observations of completed episode390201, x, y and exported speed each have
two decimal places. SUMO documents default Cartesian FCD location precision as1cm.
[Official FCD documentation](https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html)

The primary evaluator derives native speed and acceleration from coordinate
differences. Generation stores interpolated floating-point coordinates without
matching decimal rounding. Consequently, the comparison can include different
observation precision as well as different underlying motion. This is not a proof
that quantization dominates the result or explains an arm's change.

A deterministic straight-line fixture makes the issue concrete: constant speed
13.37m/s, sampled every0.25s and rounded to1cm, produces coordinate-derived mean
absolute acceleration0.08033m/s² and maximum0.16m/s² despite true acceleration0.
This is an analytic synthetic fixture, not an empirical error correction estimated
from favorable generated samples.

G39 also retains physical-time one-second acceleration, four-second speed change
and stopping endpoints. Several of those worsen with SV, so the precision issue
cannot be used to dismiss the existing adverse evidence. All original metrics and
weighting remain intact. Future confirmation should explicitly declare measurement
precision and include matched observation-operator or simulator-reported-motion
checks alongside the unchanged trajectory metrics. Any new endpoint remains
separate from the already-exposed G39 primary comparison.

Evidence: `measurement-metadata.json`; code paths `g36_requests.run`,
`g33_evaluate.extract_record`, `g31_roundtrip.signals`, `g39_generate.generate`.
No new simulation, training, policy selection or historical frozen-test access
was performed for this metadata/analytic audit.
