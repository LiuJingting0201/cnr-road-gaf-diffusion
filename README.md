# CNR Road / GAF Trajectory Diffusion

毕设目标：给定路网、起点和交通类别，扩散合成满足路网约束、运动合理性和类别语义的新轨迹；承接组内 Hilbert/GAF 表示与分类研究，验证其在生成与下游增广中的作用。

## 当前主线：承接原GAF工作开展扩散轨迹合成

**唯一当前执行任务书：[路网条件扩散与 Hilbert GAF 类别反馈任务书](docs/THESIS_TASKBOOK.md)**。恢复工作先读任务书和最新实验日志；`AGENTS.md` 固定该入口。主线是路径条件扩散直接生成坐标轨迹，自训原编码 CNN 做类别反馈筛选；可微教师 guidance 是后续有条件模块。旧计划仅作历史，冲突处以任务书为准。

原仓库没有提供 CNN。组内编码、我们训练的辅助网络、历史位移 GAF 和正式论文配置分别标识，不能混称复现。

### 最新开发诊断（2026-11-06）

- [G29 原CNR与随机需求审计](docs/results/g29-source-audit/REPORT.md)：空间低分伴随类别—路线结构变化，不能仅归因于GAF编码或CNN。
- [G30 原生MoST需求恢复](docs/results/g30-native-demand/REPORT.md)：同类数/编码/训练规则下，原GAF宏F1为CNR59.88%、随机需求30.15%、原生需求58.69%；十项宏运动分布距离点估计更接近CNR。共享需求模板与线路限制泛化，尚非guidance有效证据。
- [G31 完整行程接口](docs/results/g31-leg-interface/REPORT.md)：单场提取1644段、1592个parent；原生输出无扰动。预定路线仍需换道/步行适配，257点时间进度会平滑长行程运动。未启动新生成训练，论文暂不修改；下一步先修地图解码和物理时间表示，再锁定生成对照。
- [G32 物理时间表示](docs/results/g32-physical-clock/REPORT.md)：固定三臂检查全部1644段，旧臂完整复现；原生0.25秒速度/加速度MAE由257点的0.2614/1.0514降至0.0093/0.0314。1秒表示仍丢秒内变化；下一模型目标选0.25秒变长序列。此为oracle重构，尚非生成收益。
- [论文收敛计划](docs/THESIS_FINISH_PLAN.md)：先完成原生完整行程的生成闭环，再做空间/速度/双GAF同预算反馈；正文重组为八章，历史探索归附录。计划、已完成检查和未启动训练严格分列。

### 历史扩展实验：扩散训练与GAF反馈

核心目标是训练能够合成轨迹的路网条件扩散模型，并检验GAF语义反馈是否带来实际生成或下游分类收益。G4/G4b共16次20,000步训练已完成，含三种子反馈对照与单种子软一致性消融；结果未证明GAF反馈有稳定的总体收益。

停车与地图核验属于数据协议和评价边界，不作为独立论文主线。论文核心工作安排见[主线与实验路线](docs/NEXT_STAGE.md)。

## 当前实现边界

本仓库独立于历史仓库。目前包括车辆级数据划分、连续片段提取、Hilbert/GAF/序列分类对照、类别与起点条件扩散、相对坐标扩散、类别许可道路距离场条件及训练损失，以及独立的向量车道几何评价。

**G5/G6 已实现选定五类移动子集的路径规划与路径条件运动扩散，以及有向连接评价。** 这不代表六类、全地图或任意轨迹均已完成。早期道路距离场实验只具备几何条件，保留为历史基线；E1–E5 已完成匹配时长的原编码反馈与下游验证，当前尚未证明 GAF 的稳定独立收益；长轨迹生成仍限于严格筛选的五类子集。

实验设计、执行过程和待验证假设见 [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)，第一轮实测分析见 [docs/results/ROUND1_ANALYSIS.md](docs/results/ROUND1_ANALYSIS.md)。R1已完成18项训练，公平位移对照R1b完成3项，G2完成三种子共9项20,000步生成训练；R2长窗口研究21项训练也已完成，报告见 [representation-r2-long](docs/results/representation-r2-long/REPORT.md)。上述历史结果为验证集探索；锁定后的正式测试另见下方最终报告。

代码与记录已发布到私有仓库 [LiuJingting0201/cnr-road-gaf-diffusion](https://github.com/LiuJingting0201/cnr-road-gaf-diffusion)，同时保留开发机 Git 远端 `server`（服务器端名为 `origin`）。长窗口研究完成后，一次性回调只将完整报告推到 `results/representation-r2-long` 分支，不包含原始数据或权重。

## 最新结果入口

- **[本轮毕设总结与论文写作依据](docs/THESIS_RESULTS_SUMMARY.md)**：E1–E5/E7及一次性正式测试已完成；路网条件扩散合成闭环跑通，GAF增广收益未建立。
- **[锁定正式测试报告与图表](docs/results/final-test-e7/REPORT.md)**：53个严格测试条件、1,272条扩散候选；行程分布有条件改善，速度/加速度区间跨零；GAF增广在TCN上较仅真实数据下降5.23个百分点。

- [E7 配对区间与运动记忆诊断](docs/results/motion-audit-e7/REPORT.md)：加速度单项区间未跨零，速度/行程仍跨零；未发现预定义的完整对齐近复制，不能宣称无记忆。[最终测试协议](docs/FINAL_TEST_PROTOCOL.md)与[217项锁定清单](docs/results/final-lock-e7/manifest.json)已固定，最终运行状态见实验日志。
- [E3b 相同路径的经验运动基线](docs/results/empirical-motion-e3b/REPORT.md)：三次重采样完成；扩散运动分布误差较小，但经验基线的候选差异更大，尚无显著性结论。后续配对审计与正式测试均已完成，见上述报告。
- [E5 下游增广结果与不确定性图](docs/results/augmentation-e5/REPORT.md)：36次正式训练完成，尚未证明稳定增益。
- [E4r 路线多样性对照](docs/results/route-diverse-e4r/REPORT.md)：三种子已完成，平均每池6.70条路径；GAF同家族分数改善仍未转化为跨家族收益。

- [E2 原 Hilbert–GAF 时长匹配诊断](docs/results/gaf-duration-e2/REPORT.md)：已完成，长轨迹需使用匹配时长分类器。
- [E1 原编码 ResNet18 对照](docs/results/gaf-resnet-e1/REPORT.md)：已完成，非组内未知 CNN 复现。
- [E3/E4 长轨迹生成与原 GAF 同池筛选](docs/results/long-selection-e4-cap64/REPORT.md)：三种子评价及道路搜索容量复核完成；GAF 尚未显示稳定跨模型收益；后续 E5 与 E4r 结果见上。原保守评价也保留在 [strict E4](docs/results/long-selection-e4/REPORT.md)。

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

GAF评分已显示相对随机选择的语义筛选作用，但未显示相对同输入TCN的独特优势。地图贴合仍不等于道路连接与方向合法，历史结果为验证探索，最终结果以锁定测试报告为准。

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

以 [当前任务书](docs/THESIS_TASKBOOK.md) 第 8 节为准。分类对照限定为辅助任务，随后推进长轨迹生成、同候选预算的类别筛选与下游增广；不再启动 GASF 图像生成反解主线。
