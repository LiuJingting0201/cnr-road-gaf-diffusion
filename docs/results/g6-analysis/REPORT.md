# G6：公平坐标基线与保持行程的GAF引导

所有模型使用相同16,779训练窗口；验证为同一批80个起点，每起点4次采样。两组坐标基线共六模型新训各20,000步；路径模型复用G5三模型。报告三种子均值。测试集未用。

coordinate-g6为栅格地图坐标扩散；route-coordinate-g6为同一路线条件的坐标扩散（追加对照）；path-g5-repeat为沿路线进度扩散。追加对照与进度模型共享路线编码器和噪声MSE目标、无运动辅助损失，区别主要为自由XY与非负进度表示。不同输出维度的噪声不能逐元素配对。栅格基线另有运动辅助损失，仅作为系统比较。

| 系列 | 引导 | 图一致% | GAF审计% | TCN审计% | 速度W1↓ | 同条件样本间距m |
|---|---|---:|---:|---:|---:|---:|
| coordinate-g6 | validation_samples | 0.62 | 45.42 | 53.96 | 1.46 | 82.35 |
| route-coordinate-g6 | validation_samples | 4.58 | 47.40 | 54.48 | 1.39 | 36.20 |
| path-g5-repeat | gaf_0.3 | 100.00 | 56.88 | 70.83 | 0.98 | 25.23 |
| path-g5-repeat | gaf_1 | 100.00 | 59.90 | 71.56 | 1.05 | 24.40 |
| path-g5-repeat | gaf_3 | 100.00 | 68.96 | 70.94 | 1.37 | 22.95 |
| path-g5-repeat | none | 100.00 | 52.92 | 69.17 | 0.98 | 25.82 |
| path-g5-repeat | shuffled_gaf | 100.00 | 46.77 | 57.92 | 1.05 | 26.53 |
| path-g5-repeat | tcn_0.3 | 100.00 | 55.21 | 79.17 | 0.99 | 25.27 |
| path-g5-repeat | tcn_1 | 100.00 | 57.40 | 83.85 | 1.10 | 24.36 |
| path-g5-repeat | tcn_3 | 100.00 | 59.06 | 77.29 | 1.48 | 24.35 |
| preserved-g6 | gaf_0.3 | 100.00 | 55.73 | 70.31 | 0.99 | 25.74 |
| preserved-g6 | gaf_1 | 100.00 | 58.44 | 70.31 | 1.02 | 25.63 |
| preserved-g6 | gaf_3 | 100.00 | 62.81 | 69.58 | 1.17 | 25.39 |
| preserved-g6 | none | 100.00 | 52.92 | 69.17 | 0.98 | 25.82 |
| preserved-g6 | shuffled_gaf | 100.00 | 47.92 | 60.00 | 1.03 | 25.86 |
| preserved-g6 | tcn_0.3 | 100.00 | 54.79 | 79.27 | 1.01 | 25.67 |
| preserved-g6 | tcn_1 | 100.00 | 56.15 | 82.60 | 1.11 | 25.42 |
| preserved-g6 | tcn_3 | 100.00 | 57.71 | 76.46 | 1.36 | 25.88 |

保持行程引导将每个低噪声干净预测的非负弧长增量投影到对应无引导样本总行程，不能使用真实未来长度。约束保持弧长及终点，不保证速度分布或加速度自动合理。其是否有价值必须由独立指标判断。

所有结果仅为可可靠匹配的五类移动非行人子集，未覆盖停留、换道和行人；5m道路图审计不是完整交通合法性。样本间距越大不必然越好，坐标模型离路也可造成大间距。

## 车辆分组的不确定性检查

vehicle, class-stratified; average repetitions/seeds/windows first; macro vehicle-weighted estimates differ from window-weighted headline

posthoc validation exploration; no multiple-comparison correction; teacher seed fixed; uncertainty does not include teacher variation

| 系列 | 引导 | 独立TCN配对变化pp | 95%车辆bootstrap区间pp |
|---|---|---:|---:|
| path-g5-repeat | gaf_0.3 | 1.47 | [0.54, 2.59] |
| path-g5-repeat | gaf_1 | 2.18 | [0.14, 4.57] |
| path-g5-repeat | gaf_3 | 0.80 | [-2.10, 3.77] |
| path-g5-repeat | tcn_0.3 | 9.54 | [5.81, 13.66] |
| path-g5-repeat | tcn_1 | 13.67 | [8.47, 18.92] |
| path-g5-repeat | tcn_3 | 7.26 | [2.18, 12.19] |
| path-g5-repeat | shuffled_gaf | -11.33 | [-15.08, -7.53] |
| preserved-g6 | gaf_0.3 | 0.91 | [0.16, 1.81] |
| preserved-g6 | gaf_1 | 1.01 | [-0.83, 3.00] |
| preserved-g6 | gaf_3 | 0.38 | [-2.64, 3.28] |
| preserved-g6 | tcn_0.3 | 9.68 | [5.91, 13.91] |
| preserved-g6 | tcn_1 | 12.20 | [6.97, 17.74] |
| preserved-g6 | tcn_3 | 5.72 | [-0.73, 11.69] |
| preserved-g6 | shuffled_gaf | -8.81 | [-11.92, -5.69] |


## 本轮判断

同一路线条件的自由XY对照仍有大量整轨迹图不一致，支持沿路进度参数化的约束价值；不涉及交通交互安全。弱GAF有跨架构的小幅正向证据，但尚未证明独特优势。保持行程方案降低强引导的速度失真，也削弱类别收益，暂作为机制消融；默认保留普通弱GAF0.3。

车辆bootstrap覆盖69个不同验证车辆（bus13、motorcycle16、private_car16、bycicle16、taxi8），不是960个独立样本。探索性区间未校正多重比较，也未覆盖教师随机性，不能当作最终测试结论。

[引导收益与运动误差图](guidance_tradeoff.png)。G7分类增广尚未执行。
