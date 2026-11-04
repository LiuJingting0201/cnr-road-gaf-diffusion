# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; repeat count recorded in config; pairwise distance measured only between draws sharing the same source condition

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| validation_samples | 0.006 | 0.453 | 0.534 | 1.412 | None |

路径模型使用训练转移统计规划路线，坐标模型使用局部地图栅格；不能把规划器保证的连接归功于GAF。独立审计器未参与引导；同架构偏好仍可能存在。测试集未使用。