# CNR Road / GAF Trajectory Diffusion

毕设目标：给定路网、起点和交通类别，扩散合成满足路网约束、运动合理性和类别语义的新轨迹；承接组内 Hilbert/GAF 表示与分类研究，验证其在生成与下游增广中的作用。

## 当前主线：承接原GAF工作开展扩散轨迹合成

用户最新明确目标为地图条件扩散、道路拓扑约束和GAF分类引导，不要求图像域生成再反解。当前执行计划见[实验路线](docs/NEXT_STAGE.md)；[GAF承接关系](docs/GAF_CONTINUITY.md)保留前一轮审计历史，不能覆盖最新路线。

### 已有扩展实验：扩散训练与GAF反馈

核心目标是训练能够合成轨迹的路网条件扩散模型，并检验GAF语义反馈是否带来实际生成或下游分类收益。G4/G4b共16次20,000步训练已完成，含三种子反馈对照与单种子软一致性消融；结果未证明GAF反馈有稳定的总体收益。

停车与地图核验属于数据协议和评价边界，不作为独立论文主线。论文核心工作安排见[主线与实验路线](docs/NEXT_STAGE.md)。

## 当前实现边界

本仓库独立于历史仓库。目前包括车辆级数据划分、连续片段提取、Hilbert/GAF/序列分类对照、类别与起点条件扩散、相对坐标扩散、类别许可道路距离场条件及训练损失，以及独立的向量车道几何评价。

**道路条件、候选语义筛选及训练期GAF语义反馈已实现；连接感知的生成约束尚未完成。** 当前道路条件是近似几何条件，不保证方向与拓扑合法性。已新增有向车道一致性审计及停车区域来源核验；整轨迹约束仍是主要瓶颈。Hilbert/GAF 已核对原递归编码及 pyts 输出，但采用新的窗口协议与分类器，不宣称复现原论文精度。历史 GASF 图像扩散仍待在统一协议下重训。完整实验排期见 [docs/plan.md](docs/plan.md)。

实验设计、执行过程和待验证假设见 [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)，第一轮实测分析见 [docs/results/ROUND1_ANALYSIS.md](docs/results/ROUND1_ANALYSIS.md)。R1已完成18项训练，公平位移对照R1b完成3项，G2完成三种子共9项20,000步生成训练；R2长窗口研究21项训练也已完成，报告见 [representation-r2-long](docs/results/representation-r2-long/REPORT.md)。全部目前为验证集探索，不是最终测试结论。

代码与记录已发布到私有仓库 [LiuJingting0201/cnr-road-gaf-diffusion](https://github.com/LiuJingting0201/cnr-road-gaf-diffusion)，同时保留开发机 Git 远端 `server`（服务器端名为 `origin`）。长窗口研究完成后，一次性回调只将完整报告推到 `results/representation-r2-long` 分支，不包含原始数据或权重。

## 最新结果入口

- [G6同路线公平对照与保持行程引导](docs/results/g6-analysis/REPORT.md)

- [G5地图路径扩散与GAF引导：三种子重复采样](docs/results/path-g5-repeat/REPORT.md)
- [固定起点全部采样地图图表](docs/results/path-g5-repeat/seed42/fixed_examples.png)

- [GAF继承关系与当前缺口](docs/GAF_CONTINUITY.md)
- [G4/G4b完整反馈对照：16次训练](docs/results/semantic-followup/REPORT.md)
- [道路连接与停车状态审计：数据与评价支撑](docs/results/TOPOLOGY_AND_PARKING.md)

- [224点表示对照：21项训练](docs/results/representation-r2-long/REPORT.md)
- [道路扩散三种子汇总：9项训练](docs/results/road-g2-multiseed/REPORT.md)
- [G3公平筛选：道路与类别一致性的权衡](docs/results/candidate-selection-g3/REPORT.md)
- [固定地图案例](docs/results/candidate-selection-g3/map_examples.png)与[追加运动诊断](docs/results/candidate-selection-g3/MOTION_AUDIT.md)
- [下一阶段路线与验收门槛](docs/NEXT_STAGE.md)

GAF评分已显示相对随机选择的语义筛选作用，但未显示相对同输入TCN的独特优势。地图贴合仍不等于道路连接与方向合法，当前所有结果均为验证探索。

## 首次启动验证（2026-11-04）

独立环境通过 `pip check`，7 项测试通过。GPU0/1 分别完成 100 步 GAF 分类与坐标扩散，保存 checkpoint；扩散采样得到六类共 96 条有限数值轨迹。此为每类 64 个训练窗、16 个验证窗的小样本链路测试，不能作为正式精度或吞吐结论。

这 96 条真实验证窗的 6,144 个点在全部车道和类别许可假设下均处于中心线 5 米内；方向、拓扑连接和精确 vClass 尚未验收。完整64点数据（26,642 / 2,791 / 3,030窗口）已上传校验，文件清单见 `docs/processed_hashes.json`。`baseline-001` 两条10,000步基线均已完成；另有完整 canonical 数据包支持服务器直接重建其他窗口长度。

队列只负责这次基线启动，不会自动实现一周计划内尚未完成的模型模块。原始 CSV 另行上传，未通过 `verify_raw.py` 前不得视为完整到达。

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
