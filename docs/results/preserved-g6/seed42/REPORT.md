# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; repeat count recorded in config; pairwise distance measured only between draws sharing the same source condition

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| gaf_0.3 | 1.000 | 0.556 | 0.703 | 0.958 | 0.0 |
| gaf_1 | 1.000 | 0.584 | 0.697 | 0.998 | 0.0 |
| gaf_3 | 1.000 | 0.634 | 0.703 | 1.154 | 0.0 |
| none | 1.000 | 0.528 | 0.688 | 0.958 | 0.0 |
| shuffled_gaf | 1.000 | 0.475 | 0.606 | 1.005 | 0.0 |
| tcn_0.3 | 1.000 | 0.553 | 0.791 | 0.991 | 0.0 |
| tcn_1 | 1.000 | 0.562 | 0.819 | 1.095 | 0.0 |
| tcn_3 | 1.000 | 0.578 | 0.759 | 1.352 | 0.0 |

路径模型使用训练转移统计规划路线，坐标模型使用局部地图栅格；不能把规划器保证的连接归功于GAF。独立审计器未参与引导；同架构偏好仍可能存在。测试集未使用。