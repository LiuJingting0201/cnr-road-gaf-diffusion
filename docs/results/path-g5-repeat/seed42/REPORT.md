# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; repeat count recorded in config; pairwise distance measured only between draws sharing the same source condition

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| gaf_0.3 | 1.000 | 0.562 | 0.706 | 0.952 | 0.000 |
| gaf_1 | 1.000 | 0.603 | 0.716 | 1.027 | 0.000 |
| gaf_3 | 1.000 | 0.700 | 0.716 | 1.358 | 0.000 |
| none | 1.000 | 0.528 | 0.688 | 0.958 | 0.000 |
| shuffled_gaf | 1.000 | 0.466 | 0.575 | 1.040 | 0.000 |
| tcn_0.3 | 1.000 | 0.550 | 0.787 | 0.976 | 0.000 |
| tcn_1 | 1.000 | 0.575 | 0.844 | 1.097 | 0.000 |
| tcn_3 | 1.000 | 0.597 | 0.775 | 1.486 | 0.000 |

路径规划使用训练转移统计，扩散学习条件运动；不能把规划器保证的连接归功于GAF。引导教师未参与独立审计训练；同架构偏好仍可能存在。测试集未使用。