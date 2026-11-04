# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; repeat count recorded in config; pairwise distance measured only between draws sharing the same source condition

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| gaf_0.3 | 1.000 | 0.572 | 0.713 | 0.991 | 0.000 |
| gaf_1 | 1.000 | 0.594 | 0.716 | 1.062 | 0.000 |
| gaf_3 | 1.000 | 0.684 | 0.706 | 1.375 | 0.000 |
| none | 1.000 | 0.528 | 0.691 | 1.004 | 0.000 |
| shuffled_gaf | 1.000 | 0.466 | 0.578 | 1.052 | 0.000 |
| tcn_0.3 | 1.000 | 0.553 | 0.794 | 0.996 | 0.000 |
| tcn_1 | 1.000 | 0.566 | 0.828 | 1.102 | 0.000 |
| tcn_3 | 1.000 | 0.581 | 0.772 | 1.494 | 0.000 |

路径规划使用训练转移统计，扩散学习条件运动；不能把规划器保证的连接归功于GAF。引导教师未参与独立审计训练；同架构偏好仍可能存在。测试集未使用。