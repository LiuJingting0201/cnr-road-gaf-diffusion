# G35b parallel launch

Execution code `307ec8d` was pushed before launch. Three SUMO branches started at Unix 1793982426.110541 in the new isolated `g35b-job`; data supervisor 219389, training waiter 219390. All 13 simulation, 6 extraction, 9 queue and 15 shared-helper checks passed on the server. Runtime script hashes match the committed snapshot. No diffusion fit had started at the recorded checkpoint.

At the saved 160-second checkpoint, all three simulations were advancing: seed 350101 at 20255.00 simulation seconds, 350102 at 20283.75, and 350201 at 20404.50. Each starts at 14400 and ends at 50400. The predeclared recording lists contain 16599, 16906 and 8086 identities respectively; these are potential recording identities, not completed legs or accepted training parents. New data occupied 176890694 bytes. No failure marker appeared.

A separate read-only timing snapshot at Unix 1793982656.26 compared the common interval [18000,20940), 11760 steps. SUMO `summary.duration` means milliseconds spent computing that simulation step. The old seed 350101 averaged 15.5028 ms; new seed 350101 averaged 15.4764 ms, effectively unchanged. New seeds 350102/350201 averaged 15.1474/15.5155 ms. On [19000,20940), old/new 350101 averaged 18.4680/18.4635 ms, both with p95 22 ms. This supports no material observed per-scene compute slowdown under three concurrent scenes. It does not establish a completed-run threefold wall-clock speedup.

SUMO 1.28 computes `mySimStepDuration` before `writeOutput()`, so this timer excludes FCD output I/O. Reduced recording may improve end-to-end time but that effect is not measured here. The early 14400–18000 interval has very few vehicles and must not be used for full-day ETA. Sources: [SUMO summary output](https://sumo.dlr.de/docs/Simulation/Output/Summary.html), [SUMO 1.28 MSNet source](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_28_0/src/microsim/MSNet.cpp).

For seed 350101, running/loaded/inserted/ended counts matched at every one of 26160 common steps in [14400,20940). Aggregate matching does not establish per-vehicle coordinate equality. Recording-only motion equivalence remains unverified; no inference is made about classification or generation quality.

The v1 attempt was gracefully canceled after 3776.787 seconds at simulation time 31803.25. Its partial FCD is 6627972237 bytes, SHA256 `9b4938dbac9390cf1458fe9ea26ea10eecfa81a21557d948b38325c242f14cb8`. Completion was not claimed. Failure/configuration/log files remain. A local archive transfer and bounded checksum finalizer have started; no server FCD removal is allowed before a complete verified local copy. Archive completion is not yet observed. Initial archive approval was rejected; read-only provenance established that the file is generated entirely from public MoST inputs, and the same transfer was subsequently allowed. No original user CSV is being copied or published.

The first deployment extraction ran before its upload had finished and reported archive EOF. No simulation had started. After upload completion, deployment into the still-unused new directory and all checks succeeded; this was a deployment failure, not a scientific result.

The unchanged finite training queue waits for verified merged data, then launches four diffusion fits and two GAF readers. New generation validation and any thesis update remain pending.
