# Formal G40 training closed; generation evidence pending

All sixteen fixed 5,000-update motion fits and sixteen prior receipts are complete. Twelve priors are learned; B and C use deterministic zero/class lookup. The supervisor verified common per-seed initialization, token counts, normalizers and final batch/CUDA random states. Every fit restarted from the corresponding G39 B final model; no pilot checkpoint was reused.

|Arm|Fit minutes, seed 42 / 43|Trainable encoder parameters|Peak allocated GiB, 42 / 43|
|---|---:|---:|---:|
|B|22.90 / 24.02|0|0.61 / 0.61|
|F-SV|26.39 / 25.70|139,712|11.79 / 11.79|
|J-S|24.66 / 24.55|8,466,149|6.32 / 6.32|
|J-V|26.26 / 25.91|8,466,149|6.32 / 6.32|
|J-SV|27.46 / 26.60|16,932,298|12.02 / 12.02|
|J-RAW|26.18 / 25.74|5,922,250|1.23 / 1.23|
|J-SH|27.42 / 25.99|16,932,298|12.02 / 12.02|
|C|24.52 / 23.12|139,712|0.61 / 0.61|

Every model registers 916,801 motion parameters, including the 24,832-parameter code adapter; B bypasses that adapter, leaving the original 891,969 parameters active in its forward path. Encoder counts describe the extra trainable tail/head/bottleneck parameters, not complete frozen readers. Peak memory includes on-device frozen-prefix caches. Times cover these continuation fits under concurrent L20 operation, not upstream pretraining, prior fitting, generation or a hardware-normalized benchmark.

These files are training records only. Final training losses, class-preservation losses and code variance are not held-out classification or trajectory-quality results. Prospective synthesis, utility, independent artifact verification, figures and scientific review remain pending. Raw observations and weights are not included.
