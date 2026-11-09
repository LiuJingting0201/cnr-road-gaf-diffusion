# G39 performance isolation and equivalence protocol

2026-11-09. Authorized by the user's instruction to finish the current batch,
then optimize efficiency. This is engineering validation, not a new scientific
arm or a change to any running G39 snapshot.

## Sequence and isolation

Prepare candidate code in a separate `g39perf-job` directory. Preserve the exact
original scientific scripts under `original/scripts`. The detached controller
waits for recovered main G39, utility and factorial completion; verifies their
counts, pairing and receipt chain; and then requires an agent-written
`performance-data/scientific-review.json` tied to those three receipt hashes.
That file is written only after the completed batch has actually been reviewed.
No request for user approval is needed. A tiny CPU-only synthetic input fixture
may run during preparation; it performs no training and reads no outcomes.

Do not edit g39r/g39ur/g39fr or their inputs. At benchmark entry require all GPUs
free. Maximum dependency/review wait24h, active work60min, output3GiB, filesystem
reserve12GiB. Failure records and originals stay intact; no automatic retry or
additional full scientific training. The finite controller owns only its new
child process groups. No desktop monitoring automation is created.

## Candidate changes

1. Cache invariant native clocks, normalized context, routes and target arrays
   in device memory. Preserve selected examples, length sorting, microbatch
   grouping, effective batch64, token budget32768, width64 and FP32.
2. Transfer detached loss summaries once per optimizer update instead of several
   scalar device reads per microbatch. Check loss and gradient finiteness before
   optimizer mutation. Preserve loss, auxiliary weight, optimizer and EMA.
3. Decompress generated `txy`/offset arrays once per evaluation rather than once
   per trajectory. Preserve every metric and diagnostic calculation.

The training flag defaults to `original`; cached mode is explicitly opt-in.
The evaluator change is not deployed into existing scientific jobs. Input caching
uses more device memory. These are plausible bottlenecks, not measured speedups.

## Fixed benchmark

Synthetic CPU/CUDA fixtures first compare all input tensors and strides exactly,
including repeated examples, padding and fractional clock endpoints, with no CPU
or CUDA RNG consumption. Four concurrent slots are B42, SV42, B43, SV43. Each
runs original/cached twice,300updates per fit; orders are ABBA or BAAB by GPU.
The original baseline is the frozen G39 source, not merely a candidate flag.
All fits use existing G35 training rows and teachers, without development or
test-based checkpoint selection. Total16short fits, no full-model retraining.

Compare model, EMA, optimizer state, CPU/CUDA/NumPy RNG states, training statistics
and valid-token counts after300updates. Original repeats reveal baseline numerical
variability. Record maximum absolute difference as well as exact equality.
Adopt caching only if these states are bitwise identical and measured throughput
improves. Failure does not authorize relaxed tolerances or scientific deployment.

Report whole-run time (including cache creation), warmed100→300update rate, peak
allocated device memory, approximately10s host-RSS/GPU activity observations.
GPU occupancy alone is not success; sparse RSS snapshots can miss short peaks.

Evaluate one predeclared already-exposed completed cell,390201/B42, twice per
implementation with reversed order. Require complete metric/diagnostic JSON
equality. This re-computation cannot select a model or change scientific metrics.
Speedups from one fixed case need not generalize to all sequence lengths.

## Evidence and boundaries

PyTorch's [performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
identifies scalar device reads and CPU/GPU transfers as synchronization sources.
The local code audit is documented in `results/g39-native-alignment/UTILIZATION_AUDIT.md`.
Neither establishes the achieved speedup; publish actual receipts when complete.
Mixed precision, changed microbatches, compilation and batched sampling are outside
this first benchmark because they need additional numerical and quality checks.

Public artifacts: code, fixed protocol, aggregate timings, numerical comparisons
and failure logs without individual trajectories. Raw data and checkpoints stay
on the server or in the already-authorized private artifact repository.
