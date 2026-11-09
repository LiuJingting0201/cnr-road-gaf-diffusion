# G39: learn motion with frozen GAF representations

Status: preregistered before new training or reference outcomes, 2026-11-09.
User authorizes four-GPU execution, independent analysis, bounded iteration and GitHub records.

## Question and scope

Given the same map-derived route, class, training-only duration prior and noise,
does auxiliary supervision from inherited spatial Hilbert–GAF and speed GAF
improve the trajectory produced directly by a native-clock diffusion model?
This repeats the G28 representation-alignment idea under restored native demand,
the G35 five-class teachers and G33 native clock; it is not a new alignment
algorithm. G28 did not establish a robust benefit. G38 terminal feedback improved
some acceleration metrics while worsening speed. Neither result predetermines G39.

Spatial teacher inputs remain the original XY/Hilbert construction; speed teacher
inputs remain speed-series angular images. Both CNNs are ours, not supplied by CNR.
No GAF image inversion and no future reference path, duration or feature at sampling.

## Fixed training design

G35 immutable direct-legs: 29,895 train legs / 24,073 parents; development 7,230 /
5,842. Five classes, unchanged source split. Frozen tests are not opened. Teacher
features are the already saved final-epoch training predictions, with exact ID,
parent, label, receipt and artifact SHA checks. Only train rows define targets.

Seven arms, seeds 42/43, **14 fits**, 10,000 updates each, effective batch64,
AdamW lr2e-4, weight decay1e-4, gradient clip1, EMA.001. Native .25s G33 denoiser,
width64, microbatch32/token budget32768. Equal steps and sampled examples, actual
GPU seconds reported (not a claim of identical FLOPs). Fixed final EMA, no best-val
checkpoint or auxiliary-weight search.

| Arm | Auxiliary target |
|---|---|
| B | None |
| S | Spatial GAF feature |
| V | Speed GAF feature |
| SV | Mean of spatial and speed losses |
| P | Two class-centroid features; label-information control |
| SH | Two features from another same-class, similar-length/duration train trip |
| RAW | Mean of raw XY-series and raw speed-series encoder feature losses |

Each view target is training-global-mean centered then unit normalized. Class
centroids are normalized averages of those unit features. Hidden features from the
last middle block are pooled with physical-time weights, then passed through
128→128→512 MLPs and unit normalization. Every arm instantiates the same two heads
with RNG preservation; baseline output state omits auxiliary heads at export.
Loss = unchanged physical-time-weighted denoising MSE + 0.1 × alpha_bar(t) ×
cosine loss, equal trajectories; dual-view losses are averaged, not summed.
No sampled clean GAF is required or differentiated through Hilbert quantization.

Fixed SH derangement: within class, sort by log route length, partition into
groups of at most128, sort each group by log observed train duration, then cycle
blocks of at most16 with a seeded permutation. Singleton blocks merge into their
predecessor. Persist pair IDs, relative length/duration differences and any same
parent matches. This controls approximate conditions, not all route information;
SH versus SV is not an isolated causal test of motion information.

RAW teachers: two 1D residual CNNs, 512-D pooled features, 50 epochs, batch64,
AdamW2e-4/1e-4 and inverse-sqrt class weights, final epoch only. Raw XY uses224
relative-time samples normalized by train coordinate mean/SD; raw speed uses the
same128 profiles as the GAF reader, divided by35m/s. No handcrafted class features.
Same labeled cohort/epoch budget, but architecture and pretraining differ: GAF
ResNet18 uses ImageNet initialization, RAW starts from scratch. Report parameters,
updates, seconds and dev classification, do not claim encoding-only causality.

All seven diffusion arms share initial backbone, train batch/timestep/noise stream
per seed; heads do not consume that stream. Teacher caching uses no augmentation.

## Independent reference and generation

Eight new SUMO episodes390201–390208: paired windows18000/32400s departure,
four request cohorts (selection seeds39001–39004). End = departure+5000s; native
background starts14400s; .25s output. Each has24 public bus service plans and19
each other class. Nonbus source parents exclude the prior G36/G38 cohorts and
are disjoint between the four cohorts; each cohort is repeated across its two
windows. Cohorts use original development identity metadata only, never outcomes.
Bus templates and map are shared. Independent simulator RNG does not make shared
OD/service templates independent populations; no unseen-city or real-world claim.

Freeze manifests before simulation. No failed-condition replacement. Record
request→catalog→simulation→admissible-reference support by class and all failures.
Route catalog is the inherited network-only proposal. Freeze G38 neural route
policy per seed; sample route using its catalog-normalized probabilities. Generate
four unscored independent route/motion draws per request, matched across arms:
K1 (first draw) is primary, all-four distribution and diversity secondary. No
GAF ranking in this main comparison. Identical route and sampled duration across
arms must be verified. Motion training alone cannot improve the fixed route or
duration marginals. The generator remains route-policy + duration-prior + diffusion.

Primary quality endpoints: class-macro native speed W1 and acceleration W1; retain
both. Secondary: stop fraction, stop/start frequency, longest stop/cruise, speed
quantiles, one-second acceleration, duration, road legality, decode clamping,
candidate diversity and nearest training-motion distance. The latter uses normalized
257-point progress, up to512 SHA-ranked train legs per class, with a 1e-4 shape
distance diagnostic threshold; it is not a full spatial memorization/privacy test. Reference is one SUMO
realization, so pointwise disagreement does not mean an alternative is invalid.
Report every class, scene, seed and endpoint, equal request/physical time within
trajectory. Report paired scene deltas and cohort-paired bootstrap intervals
(four cohort clusters; descriptive uncertainty, not 16 independent experiments).

Primary comparison SV–B; S/V identify which view contributes; P/SH/RAW test stronger
explanations. A broad positive claim requires both primary distances improve in
the aggregate, compatible across seeds, without collapsed diversity/legality or
large adverse stopping changes. Mixed results remain mixed. Do not search a
metric, class, sampler step or checkpoint after seeing outcomes. Teacher confidence
is diagnostic only and cannot establish physical benefit.

## Mechanism and useful-data follow-up

First reuse G38 candidates for route-marginal-preserving S/V/SV weighting and
compare with original global weighting; this is an exposed-data diagnostic.
Then use a matched expanded pool on the two G38 scenes: per route2 durations
drawn from the training prior ×4 motion noises; all policies see this same pool.
Compare global, within-route-normalized and within-route×duration-normalized GAF
weights. No reference T. This isolates marginal shifts from motion reweighting;
the expanded budget must never be compared as though it were the oldK8 budget.

After primary evaluation, run a separately recorded finite utility stage with
independent raw-series evaluation networks: real-only, empirical resampling,
B synthetic, S/V/SV synthetic; same example and update budgets, synthetic-only
and real+synthetic regimes, hold out reference cohorts by parent. Downstream
utility is not assumed from quality. Lock exact counts before utility fits and
do not choose a single best GAF arm on its downstream evaluation data.

## Execution, failure and publication

New isolated g39-job; preserve all earlier jobs and failed attempts. Four GPU
workers: first B42/B43/S42/V42; follow with remaining fits and two RAW teachers;
RAW fits wait for those targets. SUMO CPU queue may run concurrently, max2 scenes.
At least12GiB free, at most24GiB G39 new outputs, bounded24h initial queue. Resource
failures terminate only owned child groups and preserve checkpoints/logs. Atomic
complete/failed markers and source/config/model hashes. No duplicate active queue.
Smoke checks include exact initialization/RNG, padding-invariant auxiliary pooling,
EMA hook ownership, nonzero auxiliary gradient, ID alignment, train-only normalizer,
shuffle derangement, prospective input guards and equal route/T across arms.

Commit/push protocol before launch; code, aggregate tables, figures, logs of
decisions and failures to public code repo. Raw data/weights/individual trajectories
stay on authorized server or verified private artifact repository. Thesis claims
change only after checked independent generation results. Bounded next action is
diagnose a concrete failure or run the preregistered utility/mechanism follow-up;
no infinite weight/sampler search.
