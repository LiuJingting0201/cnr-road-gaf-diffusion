# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; repeat count recorded in config; pairwise distance measured only between draws sharing the same source condition

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| gaf_0.3 | 1.000 | 0.572 | 0.706 | 0.992 | 0.000 |
| gaf_1 | 1.000 | 0.600 | 0.716 | 1.063 | 0.000 |
| gaf_3 | 1.000 | 0.684 | 0.706 | 1.380 | 0.000 |
| none | 1.000 | 0.531 | 0.697 | 0.987 | 0.000 |
| shuffled_gaf | 1.000 | 0.472 | 0.584 | 1.064 | 0.000 |
| tcn_0.3 | 1.000 | 0.553 | 0.794 | 0.996 | 0.000 |
| tcn_1 | 1.000 | 0.581 | 0.844 | 1.114 | 0.000 |
| tcn_3 | 1.000 | 0.594 | 0.772 | 1.474 | 0.000 |

路径规划使用训练转移统计，扩散学习条件运动；不能把规划器保证的连接归功于GAF。引导教师未参与独立审计训练；同架构偏好仍可能存在。测试集未使用。