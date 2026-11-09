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
