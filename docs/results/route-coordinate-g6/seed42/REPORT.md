# 地图路径条件扩散与采样引导

validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; repeat count recorded in config; pairwise distance measured only between draws sharing the same source condition

| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |
|---|---:|---:|---:|---:|---:|
| validation_samples | 0.050 | 0.478 | 0.541 | 1.415 | None |

本组为path_xy：与进度模型使用相同规划路线和路线编码器，输出自由XY坐标；不使用局部地图栅格。训练与验证窗口、采样起点和数量一致，输出噪声维度不同。测试集未使用。