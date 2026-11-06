# G28 representation and supervision audit

## Hypothesis being examined

The user proposes that time-ordered spatial Hilbert images and speed angular images can both carry participant-class motion information. G23 trained the first reader to predict eight participant classes and the second to predict six physical speed modes. G25 compared those frozen embeddings. G26–G27 compared direct speed-curve reconstruction with speed angular-image reconstruction. These experiments address different portions of the proposal. G26–G27 did not train with a CNN latent semantic objective.

The G28 tests fit both representations to the same eight participant labels on matched complete training journeys. The spatial branch continues to receive only its inherited position encoding. The speed branch receives speed derived from the same timed position observations. The two branches have their own inputs and losses. The new same-label study follows the user's latest clarification. Historical six-mode results retain their original interpretation.

## Concrete encoding differences

- `scripts/g22_native_gaf.py`, function `encode`, adds prescribed start and end positions to the native quarter-second positions before creating training images. `scripts/g25_extract.py`, functions `observed` and `inputs`, previously encoded native observations directly. The current audit compares both representations on the same monitor IDs.
- `scripts/g22_sim.py`, function `episode`, derives the diffusion progress target from cumulative chords between one-second observed positions and the prescribed endpoints. The full target is interpolated to 257 relative-time points. `scripts/g23_modes.py`, function `speed_series`, derives speed from quarter-second observation chords, then samples 128 relative-time positions. `scripts/g26_train.py`, function `normalized_speed`, computes differentiable speed using prescribed route arc length. The audit separately measures target reconstruction loss and the discrepancy between that differentiable proxy and the emitted coordinates.
- `src/cnr/original_gaf.py`, function `trajectory_sequence`, quantizes positions into Hilbert cells and applies the inherited run compression and mirror duplication. The `raw_channels` function applies per-sequence GAF range normalization. The speed angular image uses a shared 0–35 m/s scale. These representations retain different physical information.
- G25's adjusted CKA removes training-fitted contributions of class, duration, mean speed and route covariates. A small residual alignment cannot by itself rule out shared class-related motion information.

`scripts/g28_roundtrip.py` audits all training and monitor journeys. Its spatial sensitivity check selects the first four IDs per class per monitor episode before inspecting outputs. All output statistics are stored in `g28-data/roundtrip.json`. Training or monitor targets are used only to measure the representation's reconstruction floor. This script never reads evaluation cohorts.

## A deterministic temporal-information counterexample

Run the inherited compression with a 32-point output on the run sequence `[0] * 90 + [1] * 40 + [2] * 70`. Then duplicate every input observation and apply the same compression. Both outputs are exactly identical. At fixed input sampling cadence, the second traversal takes twice as long. This was executed against `vendor/original_cnr/src/cnr_trajectory/encoding/sequence.py` on 2026-11-06. It is a counterexample to automatic preservation of absolute speed by the spatial encoder, rather than an empirical classification result.

Reproduction uses the original implementation directly.

```python
from cnr_trajectory.encoding.sequence import duplicate, squeeze
original = [0] * 90 + [1] * 40 + [2] * 70
slower = [value for value in original for _ in range(2)]
a = duplicate(squeeze(original.copy(), 32), 32)
b = duplicate(squeeze(slower.copy(), 32), 32)
assert a == b
```

## Actual simulator class parameters

The running binary reported Eclipse SUMO 1.28.0. Through TraCI, the existing demand at `g22-data/train/272001/demand.xml` was loaded with its existing map, queried using `traci.vehicletype` getters, and closed before any simulation step. `scripts/g22_sim.py` creates each vehicle type with its `vClass`; the remaining parameters come from SUMO defaults. The table below records the queried values.

| Type | Length m | Acceleration m/s² | Deceleration m/s² | Maximum speed m/s | Speed deviation |
|---|---:|---:|---:|---:|---:|
| bus | 12.0 | 1.2 | 4.0 | 27.778 | 0.00 |
| bicycle | 1.6 | 1.2 | 3.0 | 13.889 | 0.10 |
| motorcycle | 2.2 | 6.0 | 10.0 | 55.556 | 0.10 |
| private car | 5.0 | 2.6 | 4.5 | 55.556 | 0.10 |
| taxi | 5.0 | 2.6 | 4.5 | 55.556 | 0.05 |
| truck | 7.1 | 1.3 | 4.0 | 36.111 | 0.05 |
| delivery | 6.5 | 2.6 | 4.5 | 55.556 | 0.05 |

All seven vehicle classes returned speed factor 1, tau 1 s and imperfection 0.5. Minimum gap was 0.5 m for bicycles and 2.5 m for the other six classes. The code's label `bycicle` maps to SUMO `bicycle`. Passenger cars and taxis share most queried physical parameters, with distinct speed variation and potentially different permitted roads. Delivery differs in vehicle length. These findings support checking class overlap and confusion. They do not establish an irreducible accuracy ceiling. No pickup, delivery stop, or scheduled bus stop behavior is explicitly added by this demand generator. Official descriptions of the defaults provide external context. [SUMO vehicle type parameter defaults](https://sumo.dlr.de/docs/Vehicle_Type_Parameter_Defaults.html)

## Fixed matched-label readout

`scripts/g28_probes.py` reads seed-42 teacher embeddings from `g28-data/teachers/{spatial,speed}-42/{train,monitor}.npz`. It verifies aligned IDs and participant labels. Each branch uses a separate training-fitted 64-component PCA, then training-fitted standardization and balanced logistic regression with fixed C=1. No monitor parameter search is performed. The concatenated representation is compared with both individual branches. Its larger dimension is reported.

Two low-dimensional comparison inputs use prescribed route geometry alone and the two available physical statistics, mean speed and four-second speed change. Their concatenation is also evaluated. The full eight-class task and a separately fitted six-motor-vehicle task are reported. All class-specific fits and transforms use training rows only. Paired monitor intervals resample complete episodes. With only three monitor episodes and one fitted teacher seed, these intervals are exploratory and conditional on the teacher fits. A classification improvement alone does not establish a gain in trajectory synthesis. Output is `g28-data/probes.json`.
