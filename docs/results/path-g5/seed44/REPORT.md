# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; one draw per supplied condition, within-condition diversity not yet measured

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| gaf_01 | 1.000 | 0.494 | 0.725 | 0.752 | 0.000 |
| gaf_03 | 1.000 | 0.512 | 0.713 | 0.727 | 0.000 |
| none | 1.000 | 0.481 | 0.713 | 0.775 | 0.000 |
| shuffled_gaf_01 | 1.000 | 0.469 | 0.700 | 0.802 | 0.000 |
| tcn_01 | 1.000 | 0.487 | 0.762 | 0.748 | 0.000 |
| tcn_03 | 1.000 | 0.506 | 0.781 | 0.701 | 0.000 |

路径规划使用训练转移统计，扩散学习条件运动；不能把规划器保证的连接归功于GAF。引导教师未参与独立审计训练；同架构偏好仍可能存在。测试集未使用。