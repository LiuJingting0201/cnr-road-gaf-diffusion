# Turin update-budget comparison

Reuse the prepared data and full conditioning of the first bus-only experiment. Both settings see the same number of passes, optimizer, clipping, epoch-based LR schedule, seeds, generation requests and noise. Reducing global batch from 1,024 to 56 at 200 epochs changes updates from 600 to 9,200, matching the Porto long-trip step count. It also changes gradient variability and random draw grouping. It does not isolate update count from batch size.

First evaluate the old and new best validation-loss checkpoints on the same 256 development requests. Then report the prespecified 341 reserved requests at both seeds. The reserve has previously been inspected. Key diagnostics are generated median length versus observed, length W1 and out-of-range point fraction; retain the four distribution metrics and equal-sized Porto control. All trips belong to a single Turin service day.
