# DiffTraj reproduction and vehicle-motion extension

## Problem and motivation

A synthetic trajectory is an ordered sequence of positions. A user who specifies an origin, destination and trip duration can reasonably expect several possible trajectories. A pointwise average across these alternatives may describe none of the observed journeys. A generative model instead estimates a distribution and samples individual sequences from it. This report asks how closely DiffTraj can reproduce observed trajectory distributions and how speed supervision and participant class affect the samples.

A perturbation baseline starts from a recorded trip and adds noise. It is simple and provides a useful reference for how much spatial structure survives modification. Variational autoencoders encode trips into a latent distribution and learn to decode samples. Trajectory GANs use a discriminator to train a generator to produce samples resembling observations. DiffTraj learns a sequence of denoising operations from known synthetic noise and allows trip attributes to condition each draw. Its released architecture, Chengdu weights and example request vectors make it a concrete starting point for a replication. The original study compares these families using the same distributional tasks. [Zhu et al., NeurIPS 2023, Sections 2 and 5](https://proceedings.neurips.cc/paper_files/paper/2023/file/cd9b4a28fb9eebe0430c3312a4898a41-Paper-Conference.pdf)

## Original method

Let X be an observed trip, represented by two standardized coordinate channels at 200 ordered positions. A training example consists of X and a vector h with departure time bin, distance, duration, original sample count, average displacement, average speed, start cell and end cell. At noise level t the training input is X_t = sqrt(a_t) X + sqrt(1-a_t) e, where e is independent standard Gaussian noise and a_t is the cumulative product of 1 minus the per-step noise variances through t. Traj-UNet uses temporal convolutions, downsampling and upsampling to predict e given X_t, t and h. The training loss is the mean squared difference between predicted and injected noise. The released guided model combines two predictions using conditional and placeholder embeddings. Generation starts with a random coordinate sequence and repeatedly uses the predicted noise to recover an estimate of X. The release uses 500 training noise levels and 100 reverse transitions. [Author implementation, pinned revision](https://github.com/Yasoz/DiffTraj/tree/27d25a9eb4cb94bcbe76068d5c8825f713a7ac2c)

## Reproduction experiment

The primary comparison is Table 1, with 3,000 generated trips per method. Chengdu and Xi’an use November 2016 taxi GPS records. Density error compares point occupancy over a 16 by 16 grid. Trip error compares joint start/end cell counts. Length error compares the distributions of summed segment lengths. For probability vectors P and Q the Jensen–Shannon divergence is half the sum of KL(P || M) and KL(Q || M), where M is their average. Smaller values indicate closer distributions. Pattern score compares frequent visited-cell patterns by the harmonic mean of precision and recall. Length discretization and the number of frequent cells must be documented alongside any reproduced figures. The procedure and targets are in the original paper, Sections 5.1–5.2 and Appendix B.

| Published DiffTraj results | Density | Trip | Length | Pattern |
|---|---:|---:|---:|---:|
| Chengdu | .0055 | .0154 | .0169 | .823 |
| Xi’an | .0126 | .0165 | .0203 | .764 |

The repository provides a Chengdu sampling example with request attributes and a checkpoint. The published numerical comparison also requires observed reference coordinates. The dataset provider currently lists a synthetic ride-hailing series on its public catalog. The original two-city reference and exact evaluation splits have yet to be located. The released checkpoint experiment and the local SUMO experiment have their own result records. [DiDi research data catalog](https://outreach.didichuxing.com/opendata/datasets)

## Extension experiment

The supplied midimost table includes time, vehicle identifier, planar coordinates in metres, simulator speed in metres per second, and class. A sample is one retained vehicle record. The development pilot requires at least 20 observations, positive travelled distance, a single class, finite positions and speeds, and inter-observation gaps at most two seconds. It evaluates five vehicle types. Physical time is sampled at 200 equally spaced instants per trip. The stored simulator speed is integrated on the original clock and averaged over each new interval. This retains a time-varying target in addition to DiffTraj's existing trip-average speed condition.

The coordinate baseline uses the released denoiser with local coordinate and attribute normalization fitted on training vehicles. The speed arm adds a weighted smooth-L1 loss between log(1 + speed) from estimated clean coordinates and log(1 + interval-average observed speed). Its pilot weight is 0.05 times a_t. The class arm adds a trainable class embedding to the conditional trip embedding. The combined arm applies both changes. All arms use the same vehicle split, training batches, sample requests and initial sampling noise. All receive observed trip descriptors as prescribed generation conditions. Evaluation compares conditional samples at those descriptors.

The initial development comparison uses 1,416 training vehicles and 375 validation vehicles, with 128 paired draws for evaluation. The local 2,000-update budget and batch size 16 are pilot settings. Local spatial diagnostics use a training-bounded grid with a separate overflow cell, 20 training-quantile length bins and natural logarithms. Speed Wasserstein distance measures how far the generated speed distribution lies from the reference in m/s, using velocities inferred from successive coordinates on both sides. Class counts accompany per-class results. These development vehicles have already influenced experimental decisions. Formal testing needs a separate untouched population and multiple seeds.

## Development results on the supplied SUMO data

A first 200-update run exposed a missing speed measurement in a private-car trajectory. Both speed-supervised fits produced nonfinite loss. The initial run also included pedestrians. The second run applied the same vehicle-class and finite-measurement filters to all four arms. The class counts after filtering are 136 buses, 173 bicycles, 351 motorcycles, 1,039 private cars and 92 taxis/ride-hailing vehicles.

| Development condition, 2,000 updates | Speed W1 (m/s) | Length W1 (km) | Density JS | Generated points outside training bounds |
|---|---:|---:|---:|---:|
| Original coordinate model | 709.73 | 142.82 | .6412 | 88.38% |
| Speed supervision | 1,281.65 | 295.04 | .6777 | 96.00% |
| Class embedding | 750.52 | 153.94 | .6501 | 88.06% |
| Both | 603.90 | 160.18 | .6719 | 96.63% |

The median reference speed across intervals was 11.36 m/s. The original coordinate model generated a median of 128.99 m/s. Across 100 random disjoint pairs of 128 observed validation trips, the median speed W1 was 0.41 m/s and median length W1 was 424 m. The combined model reduced the pilot speed W1 while increasing length and density error. All four generated populations fell far outside the reference motion range.

A second experiment evaluated the original coordinate model on 32 development trajectories with known added noise. At noise levels 50, 250 and 495, single-step clean-coordinate RMSE rose from 134 m to 774 m and then to 58,247 m. Noise-prediction MSE fell over the same levels, reflecting the amplification of noise prediction error when recovering clean coordinates. A 100-transition reverse-process control supplied with the exact noise recovered positions to 0.0000012 m RMSE. This control verifies the local transition algebra. The follow-up increased the baseline training budget to 8,000 updates with the same training seed, inputs, optimizer, batch size, sampler and evaluation requests.

## Decision sequence

1. Run the author checkpoint on its example headers, recording the sampler settings and whether every resulting coordinate is finite. Replicate both cities' reference-based scores once the observation data are available.
2. Establish the coordinate baseline on the labelled corpus. Check generated speeds, lengths, out-of-bounds point rate and origin–destination frequencies. Diagnose sampling failures before interpreting extension effects.
3. Compare baseline, speed supervision, class conditioning and their combination on shared requests. Measure the two main effects and their interaction, with per-class support. Check whether closer speed distributions accompany closer spatial distributions.
4. For formal evaluation use development data to fix training budget and preprocessing, then use independent simulator episodes or a separately untouched source population. Report each experiment with its question, data, baseline, generated examples, primary numbers and conclusion.

## Report structure

1. Problem definition and review of perturbation, latent-variable, adversarial and diffusion methods.
2. DiffTraj method, attribute definitions, forward noise process and reverse sampling.
3. Reproduction setup, available assets, observed reference data and result comparison.
4. Speed and class modifications with their training and inference inputs.
5. Matched comparisons, speed and spatial metrics, per-class results and generated examples.
6. Conclusions about reproducibility and the measured effects of the two features.

GAF and the prior road-conditioned study are archived as earlier project work.

The 8,000-update coordinate baseline reduced speed W1 from 709.73 to 51.64 m/s, median generated speed from 128.99 to 20.99 m/s, and out-of-bounds fraction from 88.38% to 8.23%. Length W1 fell from 142.82 to 11.38 km. This establishes substantial training-budget dependence on the same 128 development requests. Three further fits give each feature extension the same 8,000 updates.
