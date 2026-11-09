# G39 native-clock GAF alignment

Protocol and implementation pushed before launch: `4615ab3`, `3fbf0d9`.

The cloud supervisor has been detached in the new isolated `g39-job` directory.
Planned: 14 diffusion fits (seven arms × two seeds), two RAW-series control
teachers, eight fresh SUMO episodes and 112 direct-generation evaluation cells.
Training completion and generation benefit have **not** been established.

Real-data two-update checks passed on B/S/V/SV: identical initial backbone and
first denoising loss, finite auxiliary losses and gradient updates. Unit checks
confirm RNG preservation, right-padding invariance, EMA hook ownership, nonzero
backbone auxiliary gradient and strict export to the original sampling model.
One exposed G38 condition was used only for a sampling smoke test: B and S
produce four outputs with identical route/duration draws. It is not a quality result.

The primary comparison is the first direct sample, without GAF candidate
selection. Both native speed and acceleration W1, all classes, motion structure,
diversity and failure coverage must be reported. Utility and expanded-pool
mechanism follow-ups remain separate stages. No thesis success claim is updated.

Public repo stores code and aggregate receipts only; individual trajectories and
weights remain in authorized cloud/private storage.

Verified progress at supervisor +120s: B42/B43/S42/V42 all200updates, 2.56–2.77updates/s including startup; GPU utilization snapshot15–19%, about1.1GiB per GPU. Small model/short microbatches do not saturate L20. Both first SUMO references and map catalog are active. Free49.42GiB. This is actual progress, not training completion.

A later verified snapshot shows3,700–4,000/10,000updates in the first four fits;
no failures. Two new morning references have been extracted.
The detached utility and factorial follow-up supervisors are waiting on their
dependencies, not consuming the training GPUs. Utility protocol is fixed in
`docs/G39_UTILITY_PROTOCOL.md`; manufactured-data end-to-end checks passed.
Latest snapshot and launch receipts accompany this status.

Current handoff: `docs/G39_RUN_STATE.md`. Latest stored snapshot has the first
four fits at5,400–5,600updates and no failures, free49.21GiB. The existing-pool
route-marginal diagnostic has completed separately; its mixed physical evidence
is documented in `docs/results/g39-route-marginal/REPORT.md`.

Latest completion check:4/14diffusion fits (B42/B43/S42/V42) have completed
10,000updates in46.64–48.21minutes; model/config/stats hashes verified. Both
RAW-series control teachers have also completed50epochs and artifact checks.
Four GPU workers automatically advanced to S43/V43/SV42/SV43, about1,000–1,200
updates at inspection. Eight reference scenes:2complete, another near its end.
No failures;48.86GiB free. Generation/evaluation cells remain0: no new physical
benefit conclusion. Receipt: first-four-complete.json. Remaining training ETA
approximately2–2.5hours from this inspection, excluding generation/evaluation.


2026-11-09 07:52 UTC progress check: 8/14 diffusion fits complete (B/S/V/SV, both seeds), each 10,000 updates; all model/config/stats hashes verified. Four GPUs run P42/P43/SH42/SH43; RAW42/RAW43 remain pending. Seven of eight SUMO references have completion receipts. No failed markers; approximately48.4GiB free. Main generation/evaluation remains0/112, with utility/factorial queues still waiting. Based on current throughput, training alone needs roughly70–90more minutes; generation/evaluation are additional, not completed outcomes. Exact observation timestamp and steps: eight-fits-complete.json. No new GAF quality claim, no duplicate queue and no running-source changes.


2026-11-09T08:51:30.778140+00:00 progress check:12/14 diffusion fits complete and model/config/stats hashes verified; all8SUMO reference completion receipts present. Remaining RAW fits: {'RAW-42': 7700, 'RAW-43': 7200}, each targets10000updates. Two GPUs are now idle because only these two training tasks remain; generation starts on four GPUs after the training barrier. Generation/evaluation0/112; utility and factorial wait. No failure markers, free48.40GiB. About15minutes of training remain at observed throughput; this excludes generation/evaluation and follow-ups. Performance optimization remains deferred until the current batch is complete. Receipt:twelve-fits-complete.json.


2026-11-09T09:05:24.585546+00:00 training barrier passed:14/14 diffusion fits and2RAW teachers complete; model/config/stats hashes verified and supervisor paired initialization/data/token/statistics check passed. All8reference scenes are complete. Four generation workers have actually advanced to9–11requests/36–44candidates each in the first four cells, not just been launched. Complete generation/evaluation cells remain0/112 at this snapshot. No failure markers; free48.39GiB. Utility/factorial still await dependencies. Receipt:training-handoff.json. No quality conclusion or performance change; these remain pending the complete batch.
