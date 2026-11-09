# G39 completed scientific batch

All112main evaluation cells,26utility fits and4factorial cells completed by
2026-11-09 13:39:55UTC (21:39:55Asia/Shanghai). The collector independently verified
artifact/source links, route/T pairing, utility predictions and factorial joint
marginals. Aggregate export SHA256:
`0b1492caf82ce4a158b7ed52a2f6568f559f857377c120fef0f9d3e4bad58ba2`.

Main: SV vs B speedW1+0.743%, acceleration+0.466%. Several small stopping gains
coexist with adverse acceleration/cruise endpoints and classes. No overall
physical improvement from this hidden-alignment integration is established.

Utility: real-only91.25% macroF1, real+B93.17%, real+V94.10%, real+RAW94.06%.
Synthesis is useful on this fixed task; GAF-specific superiority is not supported.
Synthetic-only remains below real-only. Two held-out OD cohorts and shared bus
service templates limit generalization.

Factorial:10,784candidates, same pool for all10rules, two exposed scenes×two seeds.
Global-S/V/SV weighted speed changes+6.61/+2.36/+4.27%; acceleration+1.97/−0.87/+0.81%.
FixedRouteT-S/V/SV changes speed+1.60/−0.12/+0.90%, acceleration+1.24/−0.13/+0.61%.
Joint route/T mass maximum error1.6653e−16. Speed-V improvement is1/4cells,
acceleration-V2/4. Bus-specific effects coexist with adverse classes; all remain
in `../completed-factorial`. No stable overall classifier-score feedback benefit
or proof of a useful finer-motion objective follows. These are distributional
controls; individual K1 selected route/T may differ under shared Gumbel sampling.

The evidence motivates testing generation-trained representations used as actual
conditions, with a train-only code prior, rather than stronger class-score selection.
This does not establish that full GAF features lack movement information, nor that
a new injection mechanism will succeed. Proposed design: `../../../G40_GAF_GENERATIVE_CONDITIONING_DESIGN.md`.

Operational note: the first local summary attempt started before its SCP transfer
finished and encountered a truncated-JSON error. After transfer completion, parsing
and full200class-row summarization succeeded. This was an export-timing failure,
not a scientific evaluation failure. The original scientific snapshots are intact.

Performance candidates are prepared but their scientific-review gate has not been
released. The user's latest direction is to focus on the thesis/GAF mechanism.
No performance speedup or new G40 training is claimed; no paper success claim has
been updated.
