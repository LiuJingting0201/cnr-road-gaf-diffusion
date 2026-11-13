# SUMO conditional DiffTraj, global batch 32

Repeat run_sumo_original.sh in an isolated directory using the original observations.npz and DIFFTRAJ_GLOBAL_BATCH=32. Same 1,478 training vehicles, seeds 42 and 43, 200 epochs, constant-rate AdamW, final weights, original DiffTraj coordinate loss, sampling seed and 177 reserved requests. Each fit uses two GPUs. The final batch of six is divisible by two; 46 full batches plus one partial batch yield 9,400 optimizer updates. The earlier batch-1,024 comparison made 400. Reducing the batch changes gradient variability as well as update count.
