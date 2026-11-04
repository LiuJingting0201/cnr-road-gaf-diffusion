# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; repeat count recorded in config; pairwise distance measured only between draws sharing the same source condition

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| gaf_0.3 | 1.000 | 0.566 | 0.703 | 0.993 | 0.0 |
| gaf_1 | 1.000 | 0.588 | 0.709 | 1.037 | 0.0 |
| gaf_3 | 1.000 | 0.631 | 0.703 | 1.179 | 0.0 |
| none | 1.000 | 0.531 | 0.697 | 0.987 | 0.0 |
| shuffled_gaf | 1.000 | 0.481 | 0.603 | 1.032 | 0.0 |
| tcn_0.3 | 1.000 | 0.544 | 0.797 | 1.018 | 0.0 |
| tcn_1 | 1.000 | 0.566 | 0.834 | 1.119 | 0.0 |
| tcn_3 | 1.000 | 0.575 | 0.778 | 1.370 | 0.0 |

路径模型使用训练转移统计规划路线，坐标模型使用局部地图栅格；不能把规划器保证的连接归功于GAF。独立审计器未参与引导；同架构偏好仍可能存在。测试集未使用。