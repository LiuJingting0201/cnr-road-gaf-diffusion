# G40 finite pilots and prior recovery

Four1000-update motion fits completed. The original supervisor failed while treating a diagnostic source-model hash as a local artifact path. All four model artifacts, identical initial motion states, effective tokens, statistics and final batch/CUDA RNG hashes were independently verified. The source verifier is fixed in a new snapshot; old failed.json and all outputs remain. No motion fitting was repeated.

## Code-prior prerequisite

The1000-update EMA prior samples extrapolated to absolute code maxima119–178. At the unchanged formal5000-update prior budget, the same25training-parent diagnostic produces ordinary-scale codes and no corresponding noise-MSE penalty. This supports proceeding; it does not isolate EMA lag as the cause or demonstrate held-out trajectory quality.

|Arm|Known training-code MSE|Sampled5000-step prior MSE|Maximum sampled absolute code|Same-class code swap prediction RMS|
|---|---:|---:|---:|---:|
|B|0.01719325|0.01719325|0.00000|0.00000000|
|F-SV|0.01717709|0.01718205|3.32176|0.00100442|
|J-SV|0.01718157|0.01718262|3.36416|0.00079584|
|J-RAW|0.01718476|0.01717802|2.72223|0.00082687|

Code influence is nonzero but small after1000motionupdates. True-code, wrong-trip and sampled-code comparisons are in-sample implementation diagnostics at a fixed noise level, not independent synthesis scores. The ten direct pilot draws per arm retain stochastic shape differences and finite monotone outputs. No fresh reference trajectory has been opened for this review.

Formal work retains the originally declared5000motion/5000prior budgets, eight arms and both seeds. Every formal motion fit restarts from its common G39B checkpoint; no pilot checkpoint is selected. No favorable quality result is presumed. First two request manifests passed nonbus train-parent exclusion and paired-window checks, with456previously exposed nonbus source parents excluded. Public bus service templates remain shared.
