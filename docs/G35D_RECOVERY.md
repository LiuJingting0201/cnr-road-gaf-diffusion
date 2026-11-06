# G35d: recover interrupted training simulations

2026-11-07. This supersedes the planned G35c live handoff as the executable route. The capacity fix was not deployed in time. Original training scenes 350101/350102 were interrupted by their own 15 GiB guards at 44122.50/42455.75 simulation seconds. Their failure receipts explicitly record `new bytes 16111596xxx > 16106127360`; SUMO returned zero after handling the interrupt, which does not make a truncated scene successful. Both failures and partial raw outputs remain untouched.

Validation scene 350201 completed all 14400–50400 seconds with a genuine original completion receipt. It is reused without rerunning or replacing its trajectories. First G35c adoption launch rejected the changed scenario state before sending any signals. By the time obsolete Python monitors were frozen, the two training SUMOs had already received the capacity interrupt. Their original parents were resumed only to publish failure evidence. No running process can be resumed from these completed partial outputs; there is no saved SUMO state checkpoint. The earlier no-rerun handoff plan was therefore not achieved.

## Fixed recovery

Use a new isolated `g35d-job`, rerun only the two original training seeds, and reuse the completed validation output from `g35b-job`. Preserve identical network, public source, demand, vehicle recording identities, class definitions, step length, start/end time, seeds and SUMO command options. New simulation schema `g35-native-full-day-v3-capacity` distinguishes the operational resource policy; it does not change the data-generating physics.

The new monitor and all model/cache stages count the old G35b data plus new G35d artifacts together. The combined limit is 40 GiB, with at least 8 GiB free filesystem space independently required. Reused source outputs and failed attempts are therefore not hidden from the budget. No live hashed code is edited. Old stopped supervisors and the abandoned validation extractor are terminated by exact recorded identity; their files remain. The existing validation extractor's partial output is not reused as a completed dataset.

Two train simulations run in parallel; validation can be extracted immediately. Run one extractor at a time using the verified interval-storage implementation, then merge the three scene datasets and automatically launch the four original diffusion fits and two original GAF teachers. Scientific eligibility, all numeric target arrays, teacher/model settings and analysis protocols remain unchanged. Do not substitute partial training scenes for the declared full-day corpus.

## Space preservation

The prior pip/conda download-cache cleanup remains valid. To preserve the failed raw outputs while allowing full recovery, two inactive G21 training image caches may be gzip-compressed on the server. They are distinct arrays and must not be deduplicated. Record original SHA256 and verify the full decompressed archive against it before removing only the uncompressed copy. Original data contents, historical weights and results remain recoverable. Current G33/G35 cache inputs are untouched. This operation does not require the user's local computer to remain online.

Launch and progress are recorded separately after verification. The original 03:00–04:00 training-start estimate is withdrawn; it cannot describe the recovery delay. This is an infrastructure failure and correction, not evidence for or against GAF or diffusion quality.
