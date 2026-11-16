# Does speed information or the vehicle class improve sampled trajectories?

The target is an urban trip distribution conditional on a standard DiffTraj request. We test the same procedure on bicycle, bus, motorcycle, passenger car and taxi trips in MoST/SUMO. The two existing shared checkpoints were trained with correct labels on all five classes, standard coordinate denoising and the original eight attributes including mean speed. Two seeds score the same 3,000 validation requests (600 per class) with identical noise.

## Which contrasts identify which effects?

1. For speed supervision during training, the earlier C/D comparison varies the segment-speed auxiliary loss in otherwise matched labelled fits. The five-class mean speed W1 rises from 0.9170 to 0.9576 m/s at training seed 42 and from 0.8831 to 0.8901 at seed 43. Each seed favours supervision for bicycle and motorcycle and disfavors it for bus and taxi, with different directions for passenger car. These results apply to that particular objective. [Results](../difftraj-five-motion/validation-tables.md).
2. For speed information at generation, freeze the C weights. Compare original DDIM with the same training-derived speed guidance for every class. It matches five speed quantiles of 32 nearby same-class training trips, matched on trip duration and mean speed. Every class uses the same strength, sampler, and hyperparameters as the previous development study. No observed validation speed sequence enters generation. This is a development extension of a rule already examined for two classes.
3. For the value of knowing the class at generation, compare true supplied class to a label sampled using only the eight request attributes. For every request the latter draws the class of one of 32 nearest training headers, using a fixed Monte Carlo seed, standardized continuous attributes, cyclic departure time and both 2-D grid cells. The fallback can correctly infer the class. This compares knowing the class to this specific header-only fallback within the same trained generator and does not estimate the effect of omitting class during model training. In guided arms the supplied or inferred class is used both for the model and retrieval. Outcome groups always use the actual validation class.

The primary motion endpoint is equal-class mean speed W1 in m/s. Acceleration W1 uses first differences of scalar segment speed divided by the physical interval duration. The four original distribution measures are reported by class on the common training grid and bins. The fixed test set remains reserved. These data only evaluate generality across the five simulated classes on this network.

## Results

| Sampler | Class input | Seed | Macro speed W1 | Macro acceleration W1 | Density JS | OD JS | Length JS | Hotspot F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | provided | 42 | 0.9170 | 0.1514 | 0.0052 | 0.1997 | 0.0243 | 0.9500 |
| base | header_only | 42 | 0.9183 | 0.1511 | 0.0052 | 0.2009 | 0.0244 | 0.9500 |
| base | provided | 43 | 0.8831 | 0.1542 | 0.0044 | 0.1968 | 0.0228 | 0.9500 |
| base | header_only | 43 | 0.8866 | 0.1532 | 0.0044 | 0.1956 | 0.0227 | 0.9500 |
| speed | provided | 42 | 0.8714 | 0.1504 | 0.0052 | 0.2003 | 0.0237 | 0.9400 |
| speed | header_only | 42 | 0.8738 | 0.1502 | 0.0052 | 0.2012 | 0.0239 | 0.9400 |
| speed | provided | 43 | 0.8363 | 0.1542 | 0.0044 | 0.1966 | 0.0226 | 0.9500 |
| speed | header_only | 43 | 0.8408 | 0.1529 | 0.0044 | 0.1950 | 0.0225 | 0.9500 |

Speed W1, m/s. Each pair is seed 42 / seed 43. Rows are the true vehicle class.

| Vehicle | Original, inferred | Original, supplied | Guided, inferred | Guided, supplied | Fallback correct |
|---|---:|---:|---:|---:|---:|
| bicycle | 0.552 / 0.527 | 0.551 / 0.524 | 0.530 / 0.502 | 0.527 / 0.498 | 591/600 |
| bus | 1.029 / 0.957 | 1.029 / 0.957 | 1.006 / 0.921 | 1.006 / 0.921 | 600/600 |
| motorcycle | 0.907 / 0.885 | 0.954 / 0.928 | 0.845 / 0.826 | 0.896 / 0.873 | 223/600 |
| passenger | 0.966 / 0.952 | 0.935 / 0.915 | 0.910 / 0.897 | 0.875 / 0.857 | 206/600 |
| taxi | 1.136 / 1.112 | 1.116 / 1.091 | 1.078 / 1.059 | 1.053 / 1.033 | 206/600 |

With true labels, generation-time speed guidance reduced speed W1 for every class at both seeds (five-class mean reductions of 5.0% and 5.3%). The acceleration effect was small; original spatial metrics moved in both directions. Supplying true labels rather than sampling a class from the request-conditioned training neighbours gave a small mean speed-W1 change. The fallback already assigned the correct class to all 600 buses and 591 bicycles. True labels improved car and taxi speed W1 at both seeds, while motorcycle W1 was worse. This diagnostic estimates incremental information given the full header, not label-free training or cross-city transportability.
