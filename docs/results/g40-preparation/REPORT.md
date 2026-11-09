# G40 preparation and training-side audit

Completed in302.204s, before any G40 fitting or new reference outcomes. Four views each contain29,895training legs. All final classification features exactly match their archived teacher features (maximum absolute error0); all FP32 cache shards round-trip losslessly. Original spatial image bytes and speed profiles were also reproduced on every audited source trajectory.

The frozen-prefix archives occupy4.43GiB total. They avoid repeating the frozen CNN prefix; this is different from the rejected native trajectory-input caching optimization. Original trajectory input packing remains in use.

## Controlled sensitivity,500distinct training parents

100per class; fixed SHA-ranked ordering. Values below are equally averaged over the five classes. Changes are distances from the original, not improvement in realism. Native speed and acceleration distances retain their original units.

|Variant|Spatial feature cosine distance|Speed feature cosine distance|Speed W1 change (m/s)|Acceleration W1 change (m/s²)|
|---|---:|---:|---:|---:|
|copy|0.00000003|0.00000003|0.00000000|0.00000000|
|replay|0.00000003|0.00000002|0.00000026|0.00000135|
|warp-0.1|0.09373368|0.07607814|0.34997990|0.19087292|
|jitter-0.1|0.00506661|0.00010651|0.47486566|4.97033228|
|warp-0.3|0.18389620|0.16038870|1.09124959|0.27858318|
|jitter-0.3|0.01261599|0.00061033|1.57607990|16.54469588|
|smooth-4|0.01676775|0.01116035|0.05445520|0.49296646|
|smooth-16|0.03094661|0.06571970|0.19876376|0.68351992|

Replay changes are negligible. Monotone time warps affect both views. Within-second mass redistribution changes native acceleration strongly while changing the speed feature very little: the unchanged one-second/128-value speed encoding has a temporal blind spot. Spatial encoding also changes less under this perturbation than under whole-trip warps. Small nonzero copy cosine distances are floating-point normalization noise, not changed inputs.

This audit supports testing representation use and retaining physical endpoints; it does not establish that either feature improves generation or that the perturbations resemble real traffic. Do not change the locked encoding or primary endpoints after this audit. Proceed only with the four finite1000-update pilots and implementation checks. Formal fits require a separate pilot review.

Detailed per-class metrics and receipts are preserved under g40a-data; raw caches, original trajectories and weights remain server-side.
