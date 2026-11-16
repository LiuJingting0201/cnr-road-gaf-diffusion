# Vehicle-label control on SUMO validation trajectories

Each fit uses 200,000 training trips and 9,760 updates. Each evaluation uses 3,000 requests, 600 per vehicle class, and 7,016 distinct request–label draws. The trip request includes average speed. Both representations receive the same eight attributes and use the same label embedding architecture.

## Fixed-request vehicle swaps

The observed contrast is the mean difference between matched SUMO trips. The generated contrast switches the vehicle label at each header using identical noise, subtracts in the same direction, and averages over both headers of each matched pair.

5 of 10 vehicle pairs have at least 30 matches. The prespecified skill per feature is 1 − Σ|generated − observed| / Σ|observed| over these pairs. Every pair contributes one absolute error. A score of 1 reproduces the SUMO contrasts; 0 gives the total error of always predicting zero class effect. Negative scores have greater error. The denominator is the absolute SUMO contrast summed in the stated physical units. A zero denominator is shown as n/a.

| Feature | Σ∣observed∣ | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 |
| --- | --- | --- | --- | --- | --- |
| Mean speed (m/s) | 2.9394 | 0.031 | 0.059 | 0.051 | 0.041 |
| 90th-percentile speed (m/s) | 13.3431 | 0.116 | 0.113 | 0.633 | 0.585 |
| Time below 1 m/s (pp) | 48.3998 | 0.052 | 0.056 | 0.659 | 0.612 |
| Low-speed entries per 10 min | 10.2624 | 0.065 | 0.074 | 0.653 | 0.631 |
| Mean absolute acceleration (m/s²) | 1.0447 | 0.078 | 0.062 | 0.624 | 0.601 |

### Direction of the class effect

The descriptive direction screen uses at least 20 matches and a SUMO gap of at least 0.2 pooled standard deviations in magnitude, with positive pooled SD. Each entry counts eligible pairs with the correct sign. The last two columns require agreement in both seeds on the same pairs. A zero generated difference counts as a miss.

| Feature | Eligible | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 | Coordinate both | Velocity both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mean speed (m/s) | 5 | 3 | 3 | 5 | 5 | 3 | 5 |
| 90th-percentile speed (m/s) | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Time below 1 m/s (pp) | 5 | 5 | 2 | 5 | 5 | 2 | 5 |
| Low-speed entries per 10 min | 5 | 4 | 1 | 5 | 5 | 1 | 5 |
| Mean absolute acceleration (m/s²) | 7 | 3 | 3 | 5 | 5 | 3 | 5 |

### Coverage and signed differences

Pair labels give the order of subtraction. Requests can contribute to more than one comparison.

| Second − first | Matched pairs | Included in skill |
| --- | --- | --- |
| bus − bicycle | 113 | Yes |
| motorcycle − bicycle | 41 | Yes |
| passenger − bicycle | 18 | No |
| taxi − bicycle | 21 | No |
| motorcycle − bus | 20 | No |
| passenger − bus | 24 | No |
| taxi − bus | 14 | No |
| passenger − motorcycle | 577 | Yes |
| taxi − motorcycle | 579 | Yes |
| taxi − passenger | 580 | Yes |

<details>
<summary>Signed effects in physical units for all observed pairs</summary>

| Second − first | Feature | SUMO | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 |
| --- | --- | --- | --- | --- | --- | --- |
| bus − bicycle | Mean speed (m/s) | -0.5947 | +0.1661 | +0.1029 | -0.0682 | -0.0594 |
| bus − bicycle | 90th-percentile speed (m/s) | +6.9426 | +0.9881 | +0.9176 | +5.3175 | +5.2097 |
| bus − bicycle | Time below 1 m/s (pp) | +41.9353 | +2.3480 | +2.7571 | +26.1151 | +25.2079 |
| bus − bicycle | Low-speed entries per 10 min | +8.3670 | +0.5303 | +0.8175 | +5.3144 | +5.1897 |
| bus − bicycle | Mean absolute acceleration (m/s²) | +0.3521 | +0.0692 | +0.0499 | +0.2595 | +0.2510 |
| motorcycle − bicycle | Mean speed (m/s) | +1.7822 | +0.0862 | +0.0995 | +0.1000 | +0.0666 |
| motorcycle − bicycle | 90th-percentile speed (m/s) | +5.4471 | +0.2626 | +0.2843 | +2.6777 | +2.1685 |
| motorcycle − bicycle | Time below 1 m/s (pp) | +5.8095 | +0.1716 | -0.0000 | +6.1895 | +3.8546 |
| motorcycle − bicycle | Low-speed entries per 10 min | +1.6687 | +0.1542 | -0.0258 | +1.8691 | +1.2254 |
| motorcycle − bicycle | Mean absolute acceleration (m/s²) | +0.5125 | +0.0370 | +0.0406 | +0.2532 | +0.2326 |
| passenger − bicycle | Mean speed (m/s) | +1.4929 | +0.1392 | +0.1640 | +0.0743 | -0.0016 |
| passenger − bicycle | 90th-percentile speed (m/s) | +6.1415 | +0.3993 | +0.3522 | +4.2133 | +3.9403 |
| passenger − bicycle | Time below 1 m/s (pp) | +9.4361 | -0.0279 | +0.0558 | +8.2915 | +7.4400 |
| passenger − bicycle | Low-speed entries per 10 min | +2.5996 | -0.0470 | -0.0774 | +2.1975 | +2.2128 |
| passenger − bicycle | Mean absolute acceleration (m/s²) | +0.4652 | +0.0042 | +0.0175 | +0.2493 | +0.2348 |
| taxi − bicycle | Mean speed (m/s) | +1.5103 | +0.2977 | +0.3067 | +0.0560 | +0.0224 |
| taxi − bicycle | 90th-percentile speed (m/s) | +6.6669 | +0.6604 | +0.6740 | +4.5313 | +4.3238 |
| taxi − bicycle | Time below 1 m/s (pp) | +11.5578 | +0.0838 | -0.1675 | +11.0074 | +10.4092 |
| taxi − bicycle | Low-speed entries per 10 min | +2.7266 | -0.1692 | -0.1794 | +3.0712 | +2.9291 |
| taxi − bicycle | Mean absolute acceleration (m/s²) | +0.4874 | +0.1170 | +0.1056 | +0.3231 | +0.2979 |
| motorcycle − bus | Mean speed (m/s) | +0.4534 | -0.0335 | -0.0717 | +0.0422 | +0.0575 |
| motorcycle − bus | 90th-percentile speed (m/s) | -2.1494 | -0.3231 | -0.3678 | -0.7222 | -0.9784 |
| motorcycle − bus | Time below 1 m/s (pp) | -24.8492 | -0.4397 | +0.1508 | -3.4799 | -3.2161 |
| motorcycle − bus | Low-speed entries per 10 min | -4.9652 | -0.1724 | +0.2090 | -0.5581 | -0.5033 |
| motorcycle − bus | Mean absolute acceleration (m/s²) | +0.0862 | -0.0380 | -0.0215 | -0.0195 | -0.0177 |
| passenger − bus | Mean speed (m/s) | +1.1051 | +0.0359 | +0.0484 | +0.0405 | +0.0423 |
| passenger − bus | 90th-percentile speed (m/s) | -0.7385 | -0.0230 | -0.1373 | -0.3765 | -0.4385 |
| passenger − bus | Time below 1 m/s (pp) | -22.2362 | -0.7014 | -0.0523 | -2.0205 | -1.6018 |
| passenger − bus | Low-speed entries per 10 min | -4.2538 | -0.3792 | +0.0952 | -0.2726 | -0.1445 |
| passenger − bus | Mean absolute acceleration (m/s²) | +0.1536 | -0.0267 | -0.0304 | -0.0175 | -0.0138 |
| taxi − bus | Mean speed (m/s) | +1.1538 | +0.0305 | +0.0745 | +0.0595 | +0.0489 |
| taxi − bus | 90th-percentile speed (m/s) | -0.5252 | -0.1717 | -0.0661 | -0.0887 | -0.1855 |
| taxi − bus | Time below 1 m/s (pp) | -21.9670 | -0.6281 | -0.1436 | -1.8126 | -1.6691 |
| taxi − bus | Low-speed entries per 10 min | -4.3370 | -0.3080 | +0.0418 | -0.2856 | -0.3024 |
| taxi − bus | Mean absolute acceleration (m/s²) | +0.2540 | -0.0335 | -0.0151 | -0.0060 | -0.0054 |
| passenger − motorcycle | Mean speed (m/s) | +0.2165 | +0.0903 | +0.1076 | +0.0021 | +0.0014 |
| passenger − motorcycle | 90th-percentile speed (m/s) | +0.4283 | +0.1621 | +0.1956 | +0.2203 | +0.1880 |
| passenger − motorcycle | Time below 1 m/s (pp) | +0.2160 | +0.0048 | -0.0170 | +0.0793 | +0.1872 |
| passenger − motorcycle | Low-speed entries per 10 min | +0.1140 | -0.0072 | -0.0133 | -0.0635 | +0.0216 |
| passenger − motorcycle | Mean absolute acceleration (m/s²) | -0.0735 | +0.0112 | +0.0163 | -0.0669 | -0.0710 |
| taxi − motorcycle | Mean speed (m/s) | +0.2731 | +0.0843 | +0.0866 | -0.0084 | -0.0011 |
| taxi − motorcycle | 90th-percentile speed (m/s) | +0.4519 | +0.1504 | +0.1535 | +0.2153 | +0.2161 |
| taxi − motorcycle | Time below 1 m/s (pp) | +0.3220 | -0.0217 | -0.0143 | +0.1970 | +0.2942 |
| taxi − motorcycle | Low-speed entries per 10 min | +0.1090 | -0.0091 | -0.0110 | -0.0026 | +0.0435 |
| taxi − motorcycle | Mean absolute acceleration (m/s²) | -0.0911 | +0.0114 | +0.0120 | -0.0691 | -0.0722 |
| taxi − passenger | Mean speed (m/s) | +0.0729 | -0.0046 | -0.0187 | -0.0131 | -0.0062 |
| taxi − passenger | 90th-percentile speed (m/s) | +0.0734 | -0.0089 | -0.0377 | +0.0115 | +0.0263 |
| taxi − passenger | Time below 1 m/s (pp) | +0.1170 | -0.0043 | +0.0078 | +0.0671 | +0.0849 |
| taxi − passenger | Low-speed entries per 10 min | -0.0037 | -0.0089 | +0.0031 | +0.0173 | -0.0073 |
| taxi − passenger | Mean absolute acceleration (m/s²) | -0.0156 | +0.0021 | -0.0023 | -0.0032 | -0.0012 |

</details>

## Generated trajectories with the supplied vehicle class

Results are arithmetic means of five within-class scores. The four DiffTraj metrics use the native observed trajectories and generated paths interpolated to the comparison clock. Speed and acceleration use 199 equal-duration intervals on each 200-position path. Acceleration uses differences of consecutive scalar speeds. Destination hits require the generated endpoint to be in bounds and in the requested cell.

| Metric | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 |
| --- | --- | --- | --- | --- |
| Density JS ↓ | 0.0077 | 0.0078 | 0.0371 | 0.0408 |
| OD JS ↓ | 0.2357 | 0.2448 | 0.3398 | 0.3470 |
| Length JS ↓ | 0.0310 | 0.0253 | 0.0213 | 0.0112 |
| Hotspot F1 ↑ | 0.9500 | 0.9300 | 0.8000 | 0.8200 |
| Speed W1 (m/s) ↓ | 1.2134 | 1.0799 | 0.2123 | 0.1788 |
| Acceleration W1 (m/s²) ↓ | 0.1805 | 0.1897 | 0.0271 | 0.0323 |
| Destination cell hit (%) ↑ | 71.6667 | 70.5000 | 38.2000 | 37.7333 |

### Value of supplying the correct vehicle label

A fixed draw from 32 training requests with similar headers changes the label for 1,174 of 3,000 requests. The table subtracts the header-only label result from the supplied-label result. Negative deltas favour the supplied label for ↓ metrics and positive deltas favour it for ↑ metrics. The hit-rate difference is in percentage points.

| Metric | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 |
| --- | --- | --- | --- | --- |
| Density JS ↓ | +0.0000 | +0.0000 | -0.0001 | +0.0004 |
| OD JS ↓ | +0.0012 | -0.0009 | +0.0023 | -0.0009 |
| Length JS ↓ | -0.0002 | -0.0001 | +0.0003 | -0.0001 |
| Hotspot F1 ↑ | +0.0100 | -0.0100 | +0.0000 | +0.0100 |
| Speed W1 (m/s) ↓ | +0.0007 | -0.0009 | -0.0240 | -0.0377 |
| Acceleration W1 (m/s²) ↓ | +0.0004 | +0.0002 | -0.0159 | -0.0142 |
| Destination cell hit (pp) ↑ | -0.0333 | +0.1000 | -0.4667 | +0.2000 |

The next table gives the change in error for a class’s average motion statistic (supplied minus header-only). Each score averages the five absolute within-class mean errors in the listed units. Negative values favour the supplied label.

| Absolute motion error change | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 |
| --- | --- | --- | --- | --- |
| Mean speed (m/s) | +0.0005 | -0.0004 | +0.0004 | +0.0001 |
| 90th-percentile speed (m/s) | +0.0001 | +0.0013 | +0.0008 | -0.0004 |
| Time below 1 m/s (pp) | +0.0047 | -0.0018 | -0.0414 | -0.0090 |
| Low-speed entries per 10 min | +0.0019 | +0.0053 | +0.0103 | -0.0134 |

### Individual vehicle classes

Each cell shows seed 42 / seed 43 with the supplied label. Destination hit is a percentage.

| Class | Representation | Speed W1 (m/s) | Acceleration W1 (m/s²) | Median destination error (m) | Destination cell hit (%) |
| --- | --- | --- | --- | --- | --- |
| bicycle | coordinate | 0.652 / 0.656 | 0.159 / 0.181 | 217.202 / 223.422 | 76.167 / 71.167 |
| bicycle | velocity | 0.113 / 0.071 | 0.004 / 0.009 | 365.593 / 387.114 | 33.667 / 33.833 |
| bus | coordinate | 1.504 / 1.309 | 0.143 / 0.146 | 61.776 / 62.323 | 85.000 / 79.333 |
| bus | velocity | 0.181 / 0.133 | 0.022 / 0.019 | 114.557 / 105.785 | 67.667 / 68.333 |
| motorcycle | coordinate | 1.240 / 1.071 | 0.128 / 0.141 | 249.884 / 252.634 | 69.167 / 68.333 |
| motorcycle | velocity | 0.235 / 0.216 | 0.038 / 0.049 | 420.029 / 420.592 | 30.167 / 28.333 |
| passenger | coordinate | 1.246 / 1.067 | 0.224 / 0.223 | 246.297 / 242.437 | 65.167 / 67.833 |
| passenger | velocity | 0.224 / 0.188 | 0.039 / 0.046 | 423.669 / 414.349 | 26.167 / 29.333 |
| taxi | coordinate | 1.426 / 1.298 | 0.249 / 0.257 | 244.348 / 247.711 | 62.833 / 65.833 |
| taxi | velocity | 0.308 / 0.286 | 0.033 / 0.037 | 404.326 / 429.065 | 33.333 / 28.833 |

## Pairing and provenance

All four evaluations agree on these SHA-256 fingerprints. Matched support hashes each pair’s count and ordered request-ID fingerprint. The observed-summary digest covers the per-class observed features, median lengths, and the observed contrasts and pooled SDs.

| Field | SHA-256 |
| --- | --- |
| request_sha256 | `d6177a41afe6248427df030a31ee2cccc4eecfbaae1ca5d8add44f231b8172e3` |
| head_sha256 | `b35ad5b198d2ce7fa7243b7dca55307f871734b5aad837948267218e9b6559df` |
| noise_sha256 | `076d10cfb759ce4ec5f39f2e1ccadc374a682178a267bbb40a87653a9e0e524f` |
| source_manifest_sha256 | `2166be4362d59423b0df0b05ab27ca2e18df3f149270267bdf7bb8bc21318137` |
| prepared_manifest_sha256 | `f145be0e6560d5fa433dce99569bab343f8b4c423f2db06dcbf16a9af98ff236` |
| observed_summary_sha256 | `6109f997daba2be387f4dece2aa8b57e7dd2a6b9c2c1f136fa8b09a010bb5edc` |
| matched_support_sha256 | `3b27dcf4640466546362a3b0abc87a3c322f1688019d2b120b5bb313cd4829f7` |
