# Validation results for the five-class DiffTraj experiments

Each vehicle class contributes 600 validation requests. A and B fit separate models per class. C and D use one shared model with vehicle labels. B and D add speed supervision. Request and initial-noise fingerprints match for every comparison within each vehicle class. The two slow-motion errors are absolute differences of trip means.

## Equal-weight mean over the five classes

| Arm | Seed | Density JS | OD JS | Length JS | Hotspot F1 | Speed W1 (m/s) | Slow time error (pp) | Slow entries error (/10 min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 42 | 0.0070 | 0.2239 | 0.0218 | 0.9300 | 0.9629 | 3.6536 | 0.5913 |
| A | 43 | 0.0075 | 0.2202 | 0.0210 | 0.9200 | 0.9755 | 3.7243 | 0.5643 |
| B | 42 | 0.0073 | 0.2359 | 0.0265 | 0.9300 | 0.9773 | 4.4028 | 0.8572 |
| B | 43 | 0.0080 | 0.2226 | 0.0214 | 0.9300 | 0.9791 | 4.1209 | 0.6073 |
| C | 42 | 0.0052 | 0.1997 | 0.0243 | 0.9500 | 0.9170 | 4.4183 | 0.5972 |
| C | 43 | 0.0044 | 0.1968 | 0.0228 | 0.9500 | 0.8831 | 4.4496 | 0.6881 |
| D | 42 | 0.0051 | 0.1959 | 0.0239 | 0.9600 | 0.9576 | 5.2682 | 0.9077 |
| D | 43 | 0.0044 | 0.2021 | 0.0215 | 0.9500 | 0.8901 | 5.0400 | 0.9432 |

## Shared models evaluated on all 3,000 requests

| Arm | Seed | Density JS | OD JS | Length JS | Hotspot F1 | Speed W1 (m/s) | Slow time error (pp) | Slow entries error (/10 min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 42 | 0.0032 | 0.1450 | 0.0072 | 1.0000 | 0.7179 | 4.4183 | 0.5972 |
| C | 43 | 0.0024 | 0.1446 | 0.0081 | 0.9500 | 0.6883 | 4.4372 | 0.6881 |
| D | 42 | 0.0030 | 0.1404 | 0.0080 | 1.0000 | 0.7555 | 5.2682 | 0.9077 |
| D | 43 | 0.0025 | 0.1457 | 0.0071 | 0.9500 | 0.7047 | 5.0400 | 0.9432 |

## Results by vehicle class

| Class and arm | Seed | Density JS | OD JS | Length JS | Hotspot F1 | Speed W1 (m/s) | Slow time error (pp) | Slow entries error (/10 min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle, A | 42 | 0.0084 | 0.2044 | 0.0119 | 1.0000 | 0.5869 | 0.1382 | 0.2691 |
| bicycle, A | 43 | 0.0075 | 0.1965 | 0.0114 | 0.9500 | 0.5766 | 0.2705 | 0.3096 |
| bicycle, B | 42 | 0.0092 | 0.2177 | 0.0164 | 1.0000 | 0.5861 | 0.2278 | 0.2443 |
| bicycle, B | 43 | 0.0081 | 0.1987 | 0.0151 | 0.9500 | 0.5737 | 0.2822 | 0.2958 |
| bicycle, C | 42 | 0.0056 | 0.1594 | 0.0126 | 1.0000 | 0.5506 | 0.0285 | 0.1267 |
| bicycle, C | 43 | 0.0042 | 0.1718 | 0.0091 | 0.9500 | 0.5237 | 0.0310 | 0.1114 |
| bicycle, D | 42 | 0.0057 | 0.1651 | 0.0080 | 1.0000 | 0.5028 | 0.3283 | 0.2404 |
| bicycle, D | 43 | 0.0046 | 0.1824 | 0.0058 | 0.9500 | 0.4805 | 0.3827 | 0.2515 |
| bus, A | 42 | 0.0039 | 0.0582 | 0.0622 | 0.9500 | 0.7543 | 13.0829 | 0.2415 |
| bus, A | 43 | 0.0022 | 0.0290 | 0.0494 | 0.9500 | 0.7935 | 13.4648 | 0.1217 |
| bus, B | 42 | 0.0052 | 0.0489 | 0.0779 | 0.9000 | 0.8970 | 16.6013 | 1.5265 |
| bus, B | 43 | 0.0028 | 0.0227 | 0.0610 | 0.9500 | 0.8694 | 15.1658 | 0.2724 |
| bus, C | 42 | 0.0056 | 0.0485 | 0.0844 | 0.9500 | 1.0288 | 16.9556 | 0.4819 |
| bus, C | 43 | 0.0036 | 0.0380 | 0.0727 | 0.9500 | 0.9574 | 17.1650 | 0.9898 |
| bus, D | 42 | 0.0036 | 0.0492 | 0.0906 | 0.9500 | 1.1798 | 20.6918 | 1.8019 |
| bus, D | 43 | 0.0041 | 0.0465 | 0.0799 | 0.9500 | 1.0544 | 19.4648 | 1.9716 |
| motorcycle, A | 42 | 0.0094 | 0.2893 | 0.0107 | 0.9000 | 1.1087 | 1.5276 | 0.7519 |
| motorcycle, A | 43 | 0.0109 | 0.2932 | 0.0149 | 0.9000 | 1.2056 | 1.4975 | 0.7199 |
| motorcycle, B | 42 | 0.0097 | 0.3130 | 0.0126 | 0.9500 | 1.1252 | 1.5946 | 0.7978 |
| motorcycle, B | 43 | 0.0102 | 0.2992 | 0.0093 | 0.9000 | 1.1539 | 1.5779 | 0.7403 |
| motorcycle, C | 42 | 0.0046 | 0.2702 | 0.0088 | 0.9500 | 0.9544 | 1.5075 | 0.7111 |
| motorcycle, C | 43 | 0.0054 | 0.2426 | 0.0161 | 0.9500 | 0.9284 | 1.5352 | 0.7234 |
| motorcycle, D | 42 | 0.0055 | 0.2426 | 0.0090 | 0.9500 | 0.9525 | 1.5670 | 0.7306 |
| motorcycle, D | 43 | 0.0049 | 0.2582 | 0.0074 | 0.9500 | 0.8955 | 1.6173 | 0.7700 |
| passenger, A | 42 | 0.0069 | 0.2930 | 0.0144 | 0.9000 | 1.1865 | 1.6533 | 0.8142 |
| passenger, A | 43 | 0.0104 | 0.3015 | 0.0168 | 0.9000 | 1.1281 | 1.6374 | 0.8403 |
| passenger, B | 42 | 0.0060 | 0.2954 | 0.0155 | 0.9500 | 1.0947 | 1.7429 | 0.8421 |
| passenger, B | 43 | 0.0117 | 0.2968 | 0.0143 | 0.9000 | 1.1262 | 1.7035 | 0.8527 |
| passenger, C | 42 | 0.0052 | 0.2573 | 0.0081 | 0.9500 | 0.9353 | 1.7446 | 0.8511 |
| passenger, C | 43 | 0.0049 | 0.2608 | 0.0091 | 0.9500 | 0.9147 | 1.7295 | 0.8115 |
| passenger, D | 42 | 0.0059 | 0.2606 | 0.0055 | 0.9500 | 0.9753 | 1.8107 | 0.8796 |
| passenger, D | 43 | 0.0048 | 0.2578 | 0.0065 | 0.9500 | 0.9035 | 1.7965 | 0.8654 |
| taxi, A | 42 | 0.0063 | 0.2747 | 0.0096 | 0.9000 | 1.1782 | 1.8660 | 0.8799 |
| taxi, A | 43 | 0.0063 | 0.2808 | 0.0124 | 0.9000 | 1.1738 | 1.7513 | 0.8301 |
| taxi, B | 42 | 0.0063 | 0.3044 | 0.0102 | 0.8500 | 1.1835 | 1.8476 | 0.8752 |
| taxi, B | 43 | 0.0074 | 0.2955 | 0.0071 | 0.9500 | 1.1723 | 1.8752 | 0.8750 |
| taxi, C | 42 | 0.0052 | 0.2629 | 0.0078 | 0.9000 | 1.1160 | 1.8551 | 0.8152 |
| taxi, C | 43 | 0.0041 | 0.2710 | 0.0073 | 0.9500 | 1.0914 | 1.7873 | 0.8042 |
| taxi, D | 42 | 0.0050 | 0.2618 | 0.0065 | 0.9500 | 1.1775 | 1.9430 | 0.8859 |
| taxi, D | 43 | 0.0037 | 0.2655 | 0.0080 | 0.9500 | 1.1164 | 1.9389 | 0.8577 |

## Matched bus–bicycle label contrasts

Each observed pair contains a bus and a bicycle with similar requested trip attributes. Each paired request is generated twice in the shared model with identical initial noise. Signed contrasts are bus minus bicycle.

| Arm and seed | Matched pairs | Observed slow-time contrast (pp) | Generated (pp) | Observed slow-entry contrast (/10 min) | Generated (/10 min) |
|---|---:|---:|---:|---:|---:|
| all-C-42 | 113 | 41.935 | 7.747 | 8.367 | 2.157 |
| all-D-42 | 113 | 41.935 | 5.074 | 8.367 | 1.386 |
| all-C-43 | 113 | 41.935 | 8.627 | 8.367 | 2.395 |
| all-D-43 | 113 | 41.935 | 5.452 | 8.367 | 1.606 |

## Independent groups of observed trips

For each mode the 1,000 available validation trips are shuffled deterministically and split into two disjoint groups of 500. This describes empirical variation at that sample size. Evaluations above compare 600 observations against 600 samples generated for the same requests.

| Class | Density JS | OD JS | Length JS | Hotspot F1 |
|---|---:|---:|---:|---:|
| bicycle | 0.0073 | 0.4271 | 0.0110 | 0.9500 |
| bus | 0.0031 | 0.0111 | 0.0057 | 0.9500 |
| motorcycle | 0.0046 | 0.5471 | 0.0075 | 0.9500 |
| passenger | 0.0075 | 0.5769 | 0.0137 | 0.9500 |
| taxi | 0.0044 | 0.5660 | 0.0083 | 0.9000 |
