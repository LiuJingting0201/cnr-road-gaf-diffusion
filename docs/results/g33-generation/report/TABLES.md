# G33固定结果矩阵

状态：14项固定评价全部完成。

所有W1越低越好；不合成为单一总分，不计算单场伪独立置信区间。表中NA表示缺失支持。

| 方法/种子 | 原请求 | 可配对参考 | 生成条数 | 原生速度W1 m/s | 原生加速度W1 m/s² |
|---|---:|---:|---:|---:|---:|
| relative_257-42 | 100 | 94 | 776 | 0.6794 | 0.8397 |
| relative_257-43 | 100 | 94 | 776 | 0.8238 | 0.8319 |
| native_0_25s-42 | 100 | 94 | 776 | 0.6093 | 0.1492 |
| native_0_25s-43 | 100 | 94 | 776 | 0.7437 | 0.1575 |
| empirical-42 | 100 | 94 | 776 | 0.8373 | 0.3245 |
| empirical-43 | 100 | 94 | 776 | 0.9691 | 0.3588 |
| native_0_25s-42/Uniform | 100 | 94 | 97 | 0.7329 | 0.1245 |
| native_0_25s-42/S | 100 | 94 | 97 | 0.5782 | 0.1362 |
| native_0_25s-42/V | 100 | 94 | 97 | 1.0599 | 0.1233 |
| native_0_25s-42/SV | 100 | 94 | 97 | 0.7214 | 0.1280 |
| native_0_25s-43/Uniform | 100 | 94 | 97 | 0.5828 | 0.1597 |
| native_0_25s-43/S | 100 | 94 | 97 | 0.7619 | 0.1500 |
| native_0_25s-43/V | 100 | 94 | 97 | 1.0078 | 0.1314 |
| native_0_25s-43/SV | 100 | 94 | 97 | 0.8619 | 0.1373 |

## 十项物理指标：表1

| 方法/种子 | duration_s (s) | arc_length_m (m) | longest_steady_s (s) | mean_absolute_lag4_speed_change_mps (m/s) | mean_speed_mps (m/s) |
|---|---:|---:|---:|---:|---:|
| relative_257-42 | 48.2841 | 4.6610 | 20.4234 | 0.2330 | 0.4601 |
| relative_257-43 | 46.3904 | 4.6817 | 19.2598 | 0.2199 | 0.6682 |
| native_0_25s-42 | 48.2841 | 4.6980 | 21.5398 | 0.2379 | 0.4602 |
| native_0_25s-43 | 46.3904 | 4.7070 | 20.4322 | 0.1935 | 0.6682 |
| empirical-42 | 48.2841 | 4.6858 | 22.5400 | 0.5810 | 0.4599 |
| empirical-43 | 46.3904 | 4.8039 | 20.7270 | 0.5953 | 0.6685 |
| native_0_25s-42/Uniform | 57.7782 | 4.5722 | 20.5561 | 0.2155 | 0.6082 |
| native_0_25s-42/S | 58.4039 | 4.5503 | 25.4848 | 0.3205 | 0.3985 |
| native_0_25s-42/V | 74.4163 | 4.5190 | 25.3082 | 0.2722 | 0.9320 |
| native_0_25s-42/SV | 59.0127 | 4.6159 | 24.1906 | 0.2950 | 0.6988 |
| native_0_25s-43/Uniform | 51.5908 | 4.5515 | 21.6164 | 0.2416 | 0.3590 |
| native_0_25s-43/S | 51.4898 | 4.6036 | 19.1246 | 0.1354 | 0.7402 |
| native_0_25s-43/V | 77.6837 | 4.5177 | 30.7719 | 0.2088 | 0.9148 |
| native_0_25s-43/SV | 68.0578 | 4.4888 | 23.0170 | 0.2141 | 0.8416 |

## 十项物理指标：表2

| 方法/种子 | mean_absolute_acceleration_mps2 (m/s2) | stopped_fraction (fraction) | speed_p95_mps (m/s) | longest_stop_s (s) | stop_episodes_per_minute (episodes/min) |
|---|---:|---:|---:|---:|---:|
| relative_257-42 | 0.1129 | 0.0278 | 1.8650 | 4.6934 | 0.1111 |
| relative_257-43 | 0.1131 | 0.0268 | 1.8252 | 4.8932 | 0.1184 |
| native_0_25s-42 | 0.0661 | 0.0179 | 1.9802 | 7.7539 | 0.1000 |
| native_0_25s-43 | 0.0560 | 0.0171 | 1.9495 | 7.6598 | 0.0939 |
| empirical-42 | 0.1970 | 0.0156 | 2.7377 | 4.7270 | 0.1265 |
| empirical-43 | 0.2059 | 0.0128 | 2.4870 | 4.1470 | 0.1236 |
| native_0_25s-42/Uniform | 0.0547 | 0.0221 | 1.7914 | 9.2801 | 0.1185 |
| native_0_25s-42/S | 0.0870 | 0.0167 | 1.7201 | 7.6515 | 0.1248 |
| native_0_25s-42/V | 0.0700 | 0.0203 | 1.8115 | 7.7146 | 0.1312 |
| native_0_25s-42/SV | 0.0759 | 0.0169 | 1.6402 | 8.9170 | 0.1100 |
| native_0_25s-43/Uniform | 0.0693 | 0.0163 | 1.5543 | 8.1813 | 0.0880 |
| native_0_25s-43/S | 0.0432 | 0.0159 | 1.4687 | 7.9047 | 0.1039 |
| native_0_25s-43/V | 0.0563 | 0.0234 | 1.9131 | 9.6322 | 0.1208 |
| native_0_25s-43/SV | 0.0561 | 0.0198 | 1.7139 | 10.7994 | 0.1169 |

逐类全部指标及支持数见per-class.csv；方法差值、条件多样性与训练形状近邻见summary.json。

共享时长先验检查：
- seed42: True；三种全池时长W1=[48.28406609761403, 48.28406609761403, 48.28406609761403]。
- seed43: True；三种全池时长W1=[46.39041051449065, 46.39041051449065, 46.39041051449065]。

- All physical errors are descriptive on one new SUMO episode, shared map and public demand templates; no confidence intervals.
- Every row preserves the 100-request denominator; paired quality covers the stated intersection only.
- Five-class macro is NA when any class lacks support; unsupported metrics are never zero-filled.
- Full-pool clocks and empirical transfer share a duration prior; duration W1 is not evidence of motion-diffusion gain.
- Conditional route geometry and endpoint constraints are shared infrastructure, not learned road legality.
- GAF differences compare selected data with Uniform from the same candidate pool, not classifier scores.
- Equal support counts do not prove identical per-metric request IDs when partial support differs; no paired significance claim.
- Nearest normalized-progress similarity is descriptive; it does not establish privacy or absence of memorization.
