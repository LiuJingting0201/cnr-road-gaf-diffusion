# Conditional DiffTraj on supplied and newly simulated trips

[Report source](report.tex) · [Author comment and screenshot](SOURCE.md) · [Map examples and selection rule](MAP_NOTES.md) · [Model settings and metrics](metrics.json) · [Training loss by epoch](training-loss.csv)

This report uses the published conditional coordinate network. It includes the released Chengdu checkpoint sampling check, two fits on 1,478 supplied vehicle records, and two fits each on nested sets of 10,000 and 20,000 SUMO taxi trips. The primary published comparisons are the Chengdu and Xi'an results in Table 1. The 100,000-trip simulated collection contains 20,000 training journeys in each of five classes.

For the 20,000-taxi optimizer comparison, the primary endpoint is epoch 200 for each of the original and repeat fits. The repeat uses gradient-norm clipping at 1.0 and a 200-epoch cosine schedule from 0.0002 to 0.00002. The minimum fixed development denoising loss provides a supplementary checkpoint. Both training seeds and all 1,000 development requests are retained for evaluation, with identical input files, sampling noise and reference bins. The two changes are evaluated together.

Both repeat fits and all four checkpoint evaluations completed. [Paired results](stability-comparison.json) · [Epoch loss series](stability-training.csv) · [Seed 43 map](figures/stability-maps-43.png) · [Seed 42 map](figures/stability-maps-42.png) · [All-request density](figures/stability-occupancy.png) · [Final report](../../../deliverables/sumo-difftraj-reproduction/DiffTraj_SUMO_Report.pdf).
