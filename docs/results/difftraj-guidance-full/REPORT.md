# Full bus and bicycle validation with fixed speed guidance

Both frozen class-conditioned models generated all 1,000 bus and 1,000 bicycle requests, once per method. The original eight trip descriptors, specified vehicle class and initial noise were identical within every comparison. The first 600 requests per class retained the noise used in the previous study. Guidance code was copied unchanged from the 113-pair pilot. Both samplers used 100 DDIM transitions. The remaining 887 requests of each class are also reported separately to show how closely the full-set findings extend beyond the pilot. This is development validation on SUMO journeys.

Motion statistics use the same 199 intervals per trip on both sides. A slow interval has speed below 1 m/s. An entry requires a preceding faster interval followed by two consecutive slow intervals. Each entry rate is scaled to ten minutes. Signed errors below are generated minus observed. The four distribution scores use the original local replication conventions (16 × 16 training grid, 20 training-quantile length bins, top-20 hotspot F1).

## full

| Class | Seed | Speed W1, m/s | Slow time error, pp | Slow entry error, per 10 min | Density JS | OD JS | Length JS | Hotspot F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bus | 42 | 1.0253 → 1.0022 | -17.3005 → -15.6905 | -0.4892 → 0.1360 | 0.0052 → 0.0052 | 0.0443 → 0.0443 | 0.0878 → 0.0885 | 0.9000 → 0.9000 |
| bus | 43 | 0.9516 → 0.9162 | -17.3236 → -14.8533 | -0.9636 → 0.2101 | 0.0033 → 0.0033 | 0.0376 → 0.0376 | 0.0680 → 0.0673 | 0.9500 → 0.9500 |
| bicycle | 42 | 0.5451 → 0.5226 | 0.0020 → 0.3628 | -0.0986 → 0.0322 | 0.0042 → 0.0042 | 0.1427 → 0.1430 | 0.0109 → 0.0104 | 0.9500 → 0.9500 |
| bicycle | 43 | 0.5249 → 0.4995 | 0.0447 → 0.3724 | -0.0888 → 0.0031 | 0.0035 → 0.0035 | 0.1466 → 0.1466 | 0.0101 → 0.0098 | 1.0000 → 1.0000 |
| combined | 42 | 0.6538 → 0.6277 | -8.6492 → -7.6638 | -0.2939 → 0.0841 | 0.0041 → 0.0041 | 0.0913 → 0.0914 | 0.0284 → 0.0292 | 0.9500 → 0.9500 |
| combined | 43 | 0.6143 → 0.5798 | -8.6394 → -7.2405 | -0.5262 → 0.1066 | 0.0027 → 0.0027 | 0.0894 → 0.0894 | 0.0251 → 0.0253 | 0.9500 → 0.9500 |

Cells give original sampling → fixed speed guidance.

## outside_pilot

| Class | Seed | Speed W1, m/s | Slow time error, pp | Slow entry error, per 10 min | Density JS | OD JS | Length JS | Hotspot F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bus | 42 | 1.0296 → 1.0075 | -17.0356 → -15.4436 | -0.3648 → 0.2675 | 0.0059 → 0.0059 | 0.0427 → 0.0427 | 0.0978 → 0.0979 | 0.9000 → 0.9000 |
| bus | 43 | 0.9626 → 0.9273 | -17.2775 → -14.6890 | -0.8976 → 0.3589 | 0.0036 → 0.0036 | 0.0354 → 0.0354 | 0.0768 → 0.0765 | 0.9500 → 0.9500 |
| bicycle | 42 | 0.5459 → 0.5230 | -0.0006 → 0.3756 | -0.1012 → 0.0346 | 0.0046 → 0.0046 | 0.1583 → 0.1586 | 0.0134 → 0.0127 | 0.9000 → 0.9000 |
| bicycle | 43 | 0.5269 → 0.5008 | 0.0544 → 0.3898 | -0.0821 → 0.0094 | 0.0040 → 0.0040 | 0.1608 → 0.1608 | 0.0122 → 0.0121 | 1.0000 → 1.0000 |
| combined | 42 | 0.6537 → 0.6274 | -8.5181 → -7.5340 | -0.2330 → 0.1511 | 0.0046 → 0.0046 | 0.0990 → 0.0991 | 0.0331 → 0.0337 | 0.9000 → 0.9000 |
| combined | 43 | 0.6194 → 0.5842 | -8.6115 → -7.1496 | -0.4898 → 0.1841 | 0.0031 → 0.0031 | 0.0964 → 0.0964 | 0.0290 → 0.0295 | 0.9500 → 0.9500 |

Cells give original sampling → fixed speed guidance.
