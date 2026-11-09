# Completed performance check — 2026-11-10 Beijing

The entire G39 scientific batch was completed and reviewed before this bounded benchmark. No scientific full fit was repeated.

## Adopted: evaluation and utility input IO

One fixed main B42 evaluation:104.168s mean →10.017s mean, **10.399×**. All metric and diagnostic JSON values exactly equal. This is a single-cell measurement, not an overall experiment or training speedup. The separately rebuilt utility cache took19.944s; all21arrays, metadata and normalizers exactly matched. Its old preparation time was not instrumented, so no utility speedup ratio is claimed.

## Not adopted: native training input cache

Four concurrent matched slots,300updates each, two ABBA/BAAB orders.

|Slot|Steady speedup|Whole-run speedup|Original repeat model exact?|Cached model exact?|
|---|---:|---:|---|---|
|B-42|0.9833|0.9520|False|False|
|SV-42|1.0141|0.9794|False|False|
|B-43|1.0031|0.9784|False|False|
|SV-43|1.0089|0.9685|False|False|

The training state equality gate failed. Original-to-original repetitions also differed, so this benchmark does **not** isolate the cache as the cause of numerical drift. Sample/noise RNG states, training statistics and valid token counts matched. Cached peak allocated memory rose from roughly0.645GB to1.979GB; whole-run throughput did not improve. Keep original native input packing for G40. This result does not justify another optimization sweep before scientific work.

Full small receipts:report.json,complete.json,scientific-review.json,utility-cache-equivalence.json. Existing scientific and performance source snapshots stay immutable.
