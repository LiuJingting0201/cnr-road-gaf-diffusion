# Matched 20,000-taxi optimizer comparison

The initial seed-43 trajectory lengths and denoising-loss series motivated a corrective experiment. Both training seeds (42 and 43) were specified before observing the corrected outputs. The two stabilizers are applied together.

| Item | Initial | Repeat |
|---|---|---|
| Optimizer | AdamW, constant 0.0002 | AdamW, cosine schedule from 0.0002 to 0.00002 over 200 epochs |
| Gradient handling | Regular update | Clip global gradient L2 norm to 1.0 before every update |
| Primary checkpoint | Epoch 200 ordinary weights | Epoch 200 ordinary weights |
| Additional checkpoint | — | Epoch with lowest MSE on 512 fixed development tracks, fixed noise and levels |

Each repeat starts from the original random seed, with the same 20,000 training records, network, batch of 440, 9,200 optimizer updates, XY preprocessing, released noise loss, sampler, and one GPU per seed. Reuse the same 1,000 development requests in the same order; the DDIM sampler starts from the same pseudorandom draws within each pair. Use the original 10k-taxi grid and training distance-bin edges. The archive contains the exact coordinate-reference digest to verify pairing.

Inspect training loss, validation loss and the count of clipped updates. Compare both seeds' density, joint OD and travelled length JS, hotspot F1, physical length W1 and paired endpoint errors. Plot the first, middle and last requests with identical axes in corresponding before-and-after panels. Use both seeds when interpreting improvement and report the validation-selected results as a supplementary check, since the selection subset overlaps this evaluation population. No test examples are used to select weights.

On the development server the repeat is isolated under `/workspace/cnr-experiments/taxi20k-stabilized-v1`. The run and the post-training comparison each hold a lock and write a completion marker. Both fits reached epoch 200 and all four scoring runs completed. The paired graphics and aggregate tables passed the request identity and reference-grid checks.
