# CNR Road / GAF Trajectory Diffusion

毕设目标：给定路网、起点和交通类别，扩散合成满足路网约束、运动合理性和类别语义的新轨迹；通过 GAF 分类反馈承接组内研究。

## 当前实现边界

本仓库独立于历史仓库。初始实现包括车辆级数据划分、连续片段提取、类别和起点条件的时序残差网络扩散基线、浮点 GASF/GADF 分类器，以及准确的点到车道中心线距离评价。

**路网编码条件、采样期地图引导、GAF 反馈和连接感知后处理仍待实现。** 当前扩散基线不保证道路合法性；几何评价不等于方向与连接合法性。GAF 分类器不是组内 Hilbert/MTF 完整复现。历史 GASF 图像扩散将保留为统一协议下的对照组。完整实验排期见 [docs/plan.md](docs/plan.md)。

## 数据与来源

- 主数据：用户提供的 `midimost_with_class2.csv`，六类、2,685 辆/参与者 ID、4,840,789 行。原文件不会进入 Git。
- 补充数据：用户提供的 `vehicle_positions_TS_New.csv`，单类；目前没有确认对应地图，不直接套用 MoST 地图。
- 地图：[MoSTScenario](https://github.com/lcodeca/MoSTScenario/blob/master/scenario/in/most.net.xml)，GPL-3.0 来源；仅下载缓存，不纳入本仓库重新授权。hash 和配准证据在 `docs/map_alignment.json`。
- 历史工作：[原项目](https://github.com/LiuJingting0201/Deep-Learning-Based-Generative-Models-for-Traffic-Trajectory-Synthesis)，审计基础为 week07。新代码为独立实现，旧方法及研究关系保留在计划和审计材料中。

CSV 的 XY 是局部坐标，不是经纬度。MoST 候选路网与六类数据几何高度一致，但仿真版本和类别的 SUMO vClass 映射尚未完全确认。`taxi_or_uber` 暂用 taxi/passenger 许可并集评价，并明确标为假设。

## 数据协议

先按类别分层、车辆 ID 分组进行 80/10/10 划分，再按时间排序切段。非 0.25 间隔或位置变化率大于 100 的边界切断，不作镜像填充。64 点跨度为 15.75 原始时间单位；每辆最多 16 窗口，训练窗步长 32，验证/测试步长 64。只使用训练窗估计坐标均值和标准差。窗口 manifest、排除车辆及原文件 SHA256 均保留。

所有参数与阈值在本轮固定；真实仿真时间单位、异常阈值及类别区分上限需在正式实验中说明。仅用验证集选型；当前测试集不用于调参。

## 开发机使用

工作目录 `/workspace/cnr-road-gaf-diffusion`；独立环境 `.venv`；四张 L20。旧开发内容无需立即删除，避免删除已有运行环境和访问配置。

环境固定为 Python 3.12、PyTorch 2.5.1+cu121，其他直接依赖见 `requirements.txt`；安装完成后保存 `docs/environment-lock.txt`。系统 Python 无 ensurepip，首次环境通过现有 pip 的 `--python` 功能引导，后续均用 `.venv/bin/python`。

```bash
cd /workspace/cnr-road-gaf-diffusion
export PYTHONPATH=src
.venv/bin/python scripts/verify_raw.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/launch_pilots.py --name pilot-001 --steps 1000
```

预处理拒绝覆盖已有目录。若已同步同一协议的数据，跳过 prepare。启动器在 GPU0/1 分别运行分类器与坐标扩散，记录进程、命令、日志、参数和 checkpoint，拒绝覆盖同名实验。训练完成须以各任务的 `complete.json` 为准，jobs.json 仅表示已启动。当前入口不支持断点续训，不应把中断后重跑当作延续同一任务。

```bash
.venv/bin/python scripts/evaluate_geometry.py --samples data/processed/most64/val.npz --output runs/real-val-geometry.json
.venv/bin/python scripts/evaluate_geometry.py --samples runs/pilot-001/raw_diffusion/validation_samples.npz --output runs/pilot-generated-geometry.json
```

几何报告对所有样本计数，分别报告全部车道和类别许可假设下的距离，并给出速度统计。需要配合地图方向、连接、修正量、拒绝率、分类一致性、多样性和记忆检查，才能用于正式论文结论。

## 一周执行原则

前两天完成真实数据地图校验和三条训练路线（GASF、直接坐标扩散、地图条件扩散）；第三至四天完成地图约束和 GAF 反馈消融；第五天核心配置多种子；第六天冻结测试评价；第七天整理论文图表和失败案例。四卡优先独立实验，是否扩容依据实测吞吐。先完成可复现闭环，再以指标决定模块保留，不预设结果胜出。
