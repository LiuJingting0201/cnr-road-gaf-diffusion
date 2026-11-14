# Speed supervision in DiffTraj on five SUMO vehicle classes

Each journey is an ordered sequence of 200 two-dimensional positions, accompanied by the original eight DiffTraj trip descriptors including average speed. The verified simulation release contains 40,000 training journeys per class from disjoint training episodes. Its development and reserved sets contain 1,000 journeys per class each. Use training and development only for this queue.

## Experimental comparisons

| Population and request information | Coordinate denoising | Coordinate denoising and speed supervision | Seeds |
|---|---|---|---|
| One class per training run (taxi, bus, bicycle, motorcycle or passenger car) | A | B | 42, 43 for each class |
| All five classes, with the vehicle class supplied for every request | C | D | 42, 43 |

There are 20 single-class fits and four labelled multi-class fits. A vs B within the same vehicle class isolates the auxiliary speed loss. C vs D measures the same effect within a shared class-conditioned model. C and D can also be evaluated by changing just the vehicle label for a fixed request and sampling noise in supported bus/bicycle contexts. Cross-population differences between A and C also include training size, batch size and shared parameters. They cannot isolate the effect of adding a label.

Each corresponding pair keeps the same examples, initialization of common weights, shuffle order, training noise, optimizer, grid and original input attributes. The original coordinate-only objective predicts injected Gaussian noise by MSE. Speed supervision uses the model's estimated clean coordinates, metric coordinate scale and the request's interval duration to compute 199 segment speeds. Its auxiliary loss is smooth L1 between log(1 + speed) for the estimate and the time-aggregated native SUMO speed, weighted by 0.05 times cumulative signal power at the sampled noise level. The real interval speed is a training target. At inference, each model starts from random noise with the request attributes, and C/D additionally receive the requested class.

## Budget and scoring

Every fit uses 50 full training epochs and 196 optimizer updates per epoch. Single-class fits use batch 205 on 40,000 trips, and shared fits use batch 1,024 on 200,000, with microbatches up to 256. This keeps the paired budgets identical and the total 9,800 updates close to the earlier Porto training budget. All use AdamW, global gradient norm clipping at 1 and epoch-wise cosine decay from 0.0002 to 0.00002. The final epoch is primary. The lowest fixed-development noise-MSE checkpoint is retained separately. The original 500 training noise levels and 100 deterministic DDIM sampling transitions are preserved.

Score the same 600 development requests per vehicle class. Class-conditioned pooled runs score all 3,000. Report point density JS, joint origin–destination JS, length JS and top-20 hotspot F1 with the fixed training grid and length bins. Report speed W1, low-speed time and entries per ten minutes from the sampled coordinates with matched time conventions on the reference side. Interpret sparse within-class OD results alongside independent observed-subset comparisons. Label swaps use paired requests/noise and report supported sample counts. Frozen test observations remain reserved.

## Queue

An isolated server worker prepares the native speed targets. Four GPUs process ready fits, one fit per card. Each speed-supervised fit follows training and a finite, correctly sized development evaluation of its own coordinate baseline (A for B, C for D). The controls check operational validity without selecting arms by whether the scores are favourable. All remaining jobs, scoring and paired comparisons run on the server. Completion requires 24 scored fits and within-pair request/noise digest agreement. The local user session can disconnect after the detached worker and live updates are confirmed.
