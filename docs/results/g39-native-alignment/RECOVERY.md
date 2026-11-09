# G39 receipt verification failure and isolated recovery

Detected 2026-11-09 09:55 UTC. The original queue stopped after its first four
generation cells completed, before evaluation. All14diffusion fits, two RAW
teachers and eight reference scenes were already complete. Each completed pool
contains388candidates; all four will be reused only after artifact/source checks.

Root cause: `g39_run.verify` interpreted every top-level `model_sha256` as a hash
of a local `model.pt`. A generated-pool receipt uses that field for its upstream
model, while its `artifacts` manifest lists the local generated files. The queue
therefore tried to open `generated/B-42/model.pt`, which correctly does not exist.
The utility and factorial supervisors stopped on the dependency failure. This
was an orchestration bug, not a failed diffusion fit or a scientific outcome.

## Repair and continuation boundary

- Verify the explicit local artifact manifest. Legacy training receipts without
  a manifest still require their model/config/stats files; utility prediction
  hashes are checked too. Explicit source paths validate pool model/catalog
  provenance independently. Missing/corrupt files still fail.
- Preserve original `g39-job`, `g39u-job`, `g39f-job`, failure markers and logs.
  Create new isolated `g39r-job`, `g39ur-job`, `g39fr-job` snapshots. Completed
  models, inputs and pools are linked read-only by convention into the recovery
  output tree; downstream tools only write new output directories.
- `g39_resume.py` verifies original scientific source hashes, training receipts,
  references, catalogs and completed pools before reuse. It performs no training,
  simulation, parameter selection or change to sampling/metrics. The14×8cell
  design and follow-up protocols remain fixed. Generation/evaluation continue
  on four GPU slots; the original24hour/24GiB/12GiBreserve guards remain.
- Utility waits for the recovered main result; factorial waits for recovered
  utility. New dependency CLI paths change orchestration only. All original
  scientific scripts and protocols are copied from their deployed snapshots.
- This repair is necessary to finish the current batch. The separately authorized
  throughput optimization remains deferred until this batch completes.

Regression checks cover a generated pool with no local model file, correct and
incorrect upstream hashes, string/dictionary manifests, changed/missing local
artifacts, a failure marker, model configuration integrity and utility prediction
integrity. Local default Python3.9 cannot run the existing `hashlib.file_digest`
helper; checks passed using Python3.12, matching the server runtime. This local
environment mismatch did not affect any remote scientific run.

Status: implementation and local checks complete; deployment, real-data
verification and renewed progress are recorded separately after observation.
