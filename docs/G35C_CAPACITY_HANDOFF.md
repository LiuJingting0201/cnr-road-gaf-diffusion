# G35c: capacity amendment without restarting SUMO

2026-11-07, fixed before operational handoff. At 01:35 Beijing the three simulations had reached approximately 63%, 61% and 71%, with 9.98 GB written. The filesystem had 27.60 GB free after the package-cache cleanup, but the original 15 GiB attempt cap would plausibly stop the simulation before completion and cannot accommodate extraction/merged/GAF artifacts confidently. This is a resource-planning correction, not a new scientific experiment.

## Unchanged scientific inputs

Keep the exact existing three SUMO processes, executable, input demand, network, preselected FCD identities, random seeds, clocks and output paths. Do not restart, replace or resample simulations. Keep parent and episode separation, event boundaries, observation coverage, all accepted legs, Hilbert/GAF transforms, class labels, model architecture, loss, training steps and teacher epochs unchanged. No frozen evaluation data is opened.

## Explicit operational transfer

New code and monitoring live in a separate `g35c-job`; all old code and metadata remain. Before any signal, lock a manifest of PID, process start time, parent identity, executable, working directory and command hash. Freeze only the old Python queue supervisors and per-scene Python wrappers with individual SIGSTOP signals; SUMO children continue. This prevents obsolete capacity checks and duplicate downstream launches while keeping each original parent available to hold its child's true exit status.

The new monitor verifies process identity repeatedly, checks both data roots together, and records each SUMO's real wait status when the finished child is in zombie state under its frozen parent. Linux `/proc/pid/stat` field 52 is the waitpid-form exit status, while field 22 distinguishes process reuse. Check zero and nonzero synthetic child exits on the target server before production handoff. Never infer exit code zero from a disappeared PID. Documentation: [Linux proc_pid_stat manual](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html).

Accept a scene only after the real zero exit status, full final summary at 50399.75 seconds, exact original configuration/source/recording hashes and complete artifact hashes are verified. New adoption receipts are stored separately; no original-run completion receipt is forged. After capturing the finished child's exit evidence, terminate only its identity-verified frozen Python wrapper. The obsolete queue supervisors are removed only through the same explicit identity check. All transfers and failures are recorded. On a new-monitor failure, freeze still-running, identity-matched SUMO children to stop disk growth and preserve their state for recovery; never silently leave unmonitored writers running.

## Capacity and continuation

Count the original `g35b-job/g35-data` plus new `g35c-job/g35-data` together. The revised ceiling is 40 GiB, with an independent requirement to retain at least 8 GiB filesystem free space. The ceiling is not a disk allocation guarantee: the free-space check remains binding. No symlink tricks, data deletion or uncounted relocation are used to escape either limit. New manifests and code are fixed before launch; old hashed files are not edited.

Continue extraction as each scene passes adoption checks, one extraction process at a time to bound peak RAM. Merge the three unchanged numeric schemas, then automatically run the original four diffusion fits and two matched teachers. Record the combined resource policy in every new component's provenance. The original data queue's stale progress file after transfer is not a current status source; the new job is authoritative.

The extractor may store only each candidate parent's union of event intervals plus its first original observation, while still scanning every FCD record, keeping all original counters and accumulating the parent's full-observation type set. Preserve lane string width as well. This is permitted only after synthetic comparisons show identical per-candidate coverage decisions, diagnostics and returned trajectories against the old full-parent collector. Stops inside each journey remain unchanged; long activity parking outside all candidate journeys is no longer unnecessarily retained in RAM. This does not drop accepted trajectories or change their samples.

Current document records a pre-execution plan. Production handoff, receipt validation, data counts and training completion must be reported separately; no new classification or generation result is claimed.
