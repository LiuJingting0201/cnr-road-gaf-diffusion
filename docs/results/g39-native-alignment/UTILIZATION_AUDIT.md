# G39 utilization check

Observed 2026-11-09 07:57:58–07:58:09 UTC. Read-only inspection of the
four running P/SH fits; no running code, configuration or process changed.

Eleven `nvidia-smi` observations per GPU at approximately one-second intervals:

| GPU | GPU utilization samples (%) | Mean (%) | Memory used (MiB) |
|---|---|---:|---:|
| 0 | 18,18,17,16,17,24,16,21,18,15,19 | 18.1 | 1108 |
| 1 | 18,22,18,16,20,22,20,21,21,23,18 | 19.9 | 1110 |
| 2 | 22,18,18,20,17,19,21,26,26,17,19 | 20.3 | 1110 |
| 3 | 17,17,23,24,20,24,19,18,22,17,23 | 20.4 | 1108 |

Each reports 46068 MiB total memory, approximately 115–130 W draw against a
350 W limit and 2520 MHz SM clock. GPU utilization is an activity measurement,
not achieved FLOPs, so its reciprocal is not a predicted acceleration factor.
The model has 1,057,089 parameters including auxiliary heads. Logged peak tensor
allocation is approximately 647 MB; driver-visible usage is about 1.08 GiB.

Training logs at inspection: P42 6600 updates / 3.448 updates/s; P43 6200 /
3.382; SH42 5600 / 3.360; SH43 5600 / 3.382. These cumulative rates include
initialization. The latest updates use 6–7 microbatches per effective batch64.
All four training processes are active. The pending RAW fits remain under the
existing supervisor; low GPU utilization is not an absence of training workers.

Host reports about242GiB available memory and load average5.43. Each training
process consumes roughly one CPU core, also confirmed by a separate three-second
per-thread CPU-time sample. CPU cgroup quota files/mounts are not exposed in this
session; the visible CPU count is not proof of unrestricted CPU entitlement.

## Implementation evidence and limits

- `g39_train.py` uses a32768-token microbatch budget and maximum32 examples;
  selected variable-length examples commonly require6–7 forward/backward calls.
- `g33_model.pack_motion` allocates NumPy arrays and constructs time, route and
  context features synchronously for every microbatch before device transfer.
- Each microbatch converts multiple device loss scalars to Python values and
  performs a scalar finite check, creating repeated host/device synchronization.
- The training loop is eager FP32, without AMP or compiled execution. This is
  an implementation observation, not proof that either alternative is faster
  or scientifically interchangeable for this workload.

Low GPU activity, small memory use and the code path suggest host dispatch,
packing and synchronization overhead. No operator-level profile was collected,
so their individual shares and any attainable speedup are not measured. The
current queue must retain its frozen execution snapshot and paired controls.

The next engineering benchmark should first compare identical sampled examples,
noise and update budgets while precomputing invariant input features and reducing
scalar synchronization. Larger microbatches alter noise allocation/rounding and
must be validated as a separate implementation; mixed precision likewise needs
numerical and quality checks. Do not silently change only the remaining arms,
multiply throughput by the inverse utilization, or claim an achieved speedup.
