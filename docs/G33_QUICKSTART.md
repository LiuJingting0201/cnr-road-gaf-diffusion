# 当前完整轨迹模型：采样、输出与复核

2026-11-06。对应已完成的 G33，以及 G34 固定时长机制对照。当前可执行环境在隔离目录 `/workspace/cnr-experiments/gaf-inloop-v1/g33-job`；依赖相邻历史工作的工具模块，尚不是一个脱离实验环境的独立安装包。模型及原始/合成轨迹留在授权服务器，不进入 Git。

## 已有交付文件

相对上述工作目录：

| 内容 | 路径 |
|---|---|
| 原生0.25秒扩散模型 | `g33-data/models/native_0_25s-{42,43}/model.pt` |
| 对应训练侧时长先验、配置和完成回执 | 同目录 `stats.json`、`config.json`、`complete.json` |
| 257点时间表示对照 | `g33-data/models/relative_257-{42,43}/` |
| 自训空间/速度图像读出器 | `g33-data/teachers/{spatial,speed}-42/` |
| 事前公开请求和地图条件 | `g33-data/requests/`、`g33-data/maps/` |
| 未筛选完整轨迹 | `g33-data/pools/native_0_25s-{42,43}/generated.csv.gz` |
| 同池筛选后的完整轨迹 | `g33-data/selected-csv-export/native_0_25s-{42,43}-{Uniform,S,V,SV}.csv.gz` |
| G34固定时长筛选CSV | `g34-data-attempt2/selected-csv-export/native_0_25s-{42,43}-{Uniform,S,V,SV}.csv.gz` |
| 独立物理质量与来源核验 | `g33-data/report/summary.json`、`g34-data-attempt2/report/summary.json` |

CSV 列为 `time,id,x,y,type`。时间是每条合成行程从零开始的秒数，常规间隔0.25秒，必要时包含最后不足0.25秒的区间。XY 是 MoST 局部米制坐标，不是经纬度。候选 ID 带有请求与候选编号；`type` 为五类字符串。`samples.npz` 保存未做CSV十进制格式化的时刻与坐标，科学评价使用它。结果覆盖97个可构造地图条件，质量比较与97条合格参考取交集94；三个地图失败不会被偷偷补成轨迹。

筛选后的16份CSV已由独立导出器逐点回读，时间/XY的float64位模式、ID和车型与原NPZ完全相同。G33八份合计1,879,084点，G34八份1,826,500点；每份97条轨迹。不同策略会重复选择同一候选，新旧Uniform也完全相同，不能把这些文件当1,552条独立新增观测。小型核验回执见 `docs/results/g33-generation/selected-csv-export.json` 及 `docs/results/g34-fixed-duration/attempt2/selected-csv-export.json`。导出器 `scripts/g33_export_selected.py` 不载入模型、不重选候选、不读取参考。

## 在现有环境重复采样

下面示例使用冻结 seed42 模型与原100个公开请求。输出目录必须全新；程序主动拒绝覆盖既有实验。生成阶段只读取地图、模型及其训练侧统计，不读取参考未来轨迹。相同条件与随机键用于复现，不能把复现算作新的统计重复。

```bash
cd /workspace/cnr-experiments/gaf-inloop-v1/g33-job
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=src:g10-env:scripts:../g32-job/scripts:../g31-job/scripts:../g30-job/scripts:../g29-job/scripts:../g28-job/scripts:../g23-job/scripts:../g22-job/scripts:../scripts:../vendor/original_cnr/src

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/g33_evaluate.py generate \
  --maps g33-data/maps \
  --model g33-data/models/native_0_25s-42 \
  --family diffusion --seed 42 \
  --output g33-data/reproduction-native-42
```

固定配置为每请求8候选、100步DDIM。`complete.json` 记录实际网络调用、生成覆盖、耗时和产物哈希；出现 `failed.json` 或缺少完成标记时不能称复现成功。每条候选的路线起终点、有限性、单调进度都在导出前检查；这不等同于信号灯、避碰或所有车辆动力学限制均已满足。

生成完成后，才可单独使用参考做评价：

```bash
.venv/bin/python scripts/g33_evaluate.py evaluate \
  --generated g33-data/reproduction-native-42 \
  --maps g33-data/maps \
  --reference g33-data/reference-extracted \
  --training g33-data/direct-legs \
  --output g33-data/reproduction-evaluation-native-42
```

以上 CLI 与已完成正式运行共用函数。新的任意路线需要先由合法的地图条件准备接口构造，不能直接把真实未来坐标塞进 `maps.npz` 冒充生成请求。

## 如何理解 GAF 和完整模型的关系

最终生成器由训练侧时长先验、条件运动扩散、道路解码组成。扩散学习时长内的运动变化，时长本身不是扩散输出。GAF 是生成后可选择的类别反馈模块：S读取原Hilbert空间图，V读取位置推导出的速度图，SV等权组合分数；没有组内现成CNN，也没有在此阶段实现图像反解或去噪内梯度。

G33 比较所有候选可有不同训练先验时长的筛选；G34 限制每请求八条候选共享一个预定时长，识别运动形状的筛选作用。后者沿用已看过的参考，属于开发机制诊断。两者的全部物理指标及代价见各自报告，不用教师高分替代合成质量。
