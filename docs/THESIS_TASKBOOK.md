# 路网条件扩散与 Hilbert GAF 类别反馈轨迹合成任务书

版本：1.0。制定日期：2026年11月4日。执行资源：独占服务器四张 L20；必要时可申请八张 H20，但先依据吞吐和显存实测决定。本任务书用于固定研究主线、安排实验、核验结论和积累硕士论文材料。最新用户要求优先；早期计划与本任务书冲突时，以本任务书为准。

## 1 最终目标

给定道路网络、起点、交通参与者类别和生成时长，合成具有道路连通性、合理运动规律和多样性的带时间戳轨迹，输出可供下游使用的合成数据集。主要生成模块必须是经过训练的扩散模型。

采用“路网候选路径 + 路径条件运动扩散 + Hilbert–GAF 类别反馈”的框架。继承组内轨迹编码研究，自行训练类别网络，检验其能否改善生成数据的类别一致性及下游用途。GAF 的有效性是待验证假设，不是预先保证的结论。

拟用题目：基于路网条件扩散与 Hilbert–GAF 类别反馈的交通轨迹合成。正式题目可遵从学校已备案题目。

## 2 不可漂移的边界

1. 扩散直接生成轨迹的运动表示，不把生成 GAF/GASF 图像后数学反解作为主线。
2. 用户没有提供 CNN。所有新 CNN 均称为自行训练的辅助网络，不声称是组内权重或未知分类器的复现。
3. 继承的是原仓库 Hilbert、序列处理和 GAF 构造。自定义位移 GAF 只作历史扩展对照，不替代继承关系。
4. 分类训练、地图核验、异常和停车处理均服务于生成；不扩展成独立的分类或停车课题。
5. 路径解码的拓扑合法性属于结构约束贡献，不能算作 GAF 的贡献。无证据时不称为可泛化到任意城市或完全物理可行。
6. 数据增广须在真实留出数据上验证；筛选教师自评分提高不等于合成数据有用。
7. 对已经失败的实验保留配置与结论，禁止为了讲故事删类、挑样本或改口径后不披露。

## 3 数据与已有资产

| 资产 | 已知情况 | 使用边界 |
|---|---|---|
| midimost_with_class2.csv | 用户提供，4,840,789 行、2,685 个 ID，六类参与者，时间间隔主要为 0.25 | 主数据；保留原文件哈希与标签映射 |
| vehicle_positions_TS_New.csv | 用户提供，2,189,591 行、12,061 个 ID，单类 | 对应地图未确认，暂不混入主实验，不套用 Monaco 路网 |
| MoST Monaco 路网 | 从公开 MoSTScenario 获取，和主数据几何配准证据已记录 | 仿真版本及部分 vClass 映射未完全确认；报告假设与误差 |
| 原仓库编码 | 固定源码提交 4773ad65bce88d38b8b108ef5e4773fd99883c94，已保留源码和哈希 | 逐点 Hilbert，保留时间顺序；非按 Hilbert 值重排轨迹 |
| 已有 G5/G6 | 路径条件运动扩散、多种子及道路连接评价 | 可复用主模型；原实验位移 GAF 反馈不等于原 Hilbert–GAF |
| 原编码小 CNN | 全轨迹验证准确率均值 68.77%，macro-F1 49.67% | 自训小网络结果，不是组内 CNN 或正式论文复现 |

六类为 pedestrian、bus、motorcycle、private_car、bycicle、taxi_or_uber；保留源数据 bycicle 拼写并在展示时解释为 bicycle。坐标是局部 SUMO XY，不能称为经纬度。G5/G6 现有生成队列仅覆盖筛选后的五类移动非行人轨迹；六类分类指标不能冒充六类生成完成。

## 4 数据协议

- 沿用按类别分层、按参与者 ID 的 80/10/10 划分，先划分再切窗口。所有窗口、完整轨迹、图像共享同一 ID 归属。
- 只用训练集估计坐标范围、图像通道范围、运动归一化、道路转移先验。验证用于开发；测试保持冻结，方法与阈值锁定后一次统一评价。
- 完整轨迹分类与短窗口生成是不同分布。分别记录原始长度、时间跨度、网格覆盖、常量 Hilbert 比例与排除原因，不直接混比精度。
- 轨迹有间断时不得伪装成连续观测；原编码全轨迹分类的处理和生成连续段处理分开记录。长度压缩失败必须明示，不能默默替换算法。
- 归一化范围冻结，验证与生成越界需计数。不能每批重新缩放图像导致同一轨迹的表示依赖同批样本。
- 实验比较尽量使用共同 ID/候选集合；不同时报告覆盖率和差异来源。生成样本不使用验证或测试真实未来路径。

## 5 技术路线与模块

### 5.1 路径条件运动扩散

从类别许可的有向道路连接图采样路径，起点可来自预先固定的评价条件，未来路径由训练数据统计及路网生成。给定路径几何、类别和起点，扩散模型学习沿路径前进距离的时间序列，再解码为 XY。现有实现采用 log1p 非负增量扩散，可作为起点。

路径必须足够长；报告路径耗尽、截断和强制停止比例，不把到达路径末端造成的停留当成学得行为。检查车道连接处跳变、方向、速度与加速度。结构上合法不代表动力学或多参与者碰撞安全；当前任务是单参与者轨迹合成。

下一步扩展时间尺度：64 点仅作已有基线，224/512 点作为候选；先核验样本覆盖与编码信息，再固定一个主生成长度。更长轨迹可采用分段生成与连续性约束，但不能只拼接后忽略接缝速度突变。条件应包含生成时长；固定时长实验必须明确固定值。

### 5.2 原始 Hilbert GAF 类别网络

继承全局坐标归一化、逐点 Hilbert、原 squeeze/mirror、GASF/GADF/MTF 三通道。默认 eps=.0037 对应 32×32 网格，即 1024 个单元；64×64 是 4096 个总单元，不能混成每轴 4096。

自行训练 CNN，首轮比较 ResNet18 随机初始化与 ImageNet 预训练，在两档网格下采用相同划分、轮次、损失和输入处理。图像不作自然图像随机翻转、裁剪、颜色扰动，以免改变时间或编码含义。以验证 macro-F1、各类召回和校准检查为主，准确率为辅。

完整轨迹网络不能直接移植到短片段而声称可用。必须在与生成相匹配的真实轨迹时长上检验；必要时训练相匹配的窗口网络。GAF 模块的独立分类研究限定在首轮约一天内，随后回到生成对照。低分允许形成负结果，但不能无限换架构刷分。

### 5.3 第一版 GAF 反馈筛选

对每个固定生成条件产生 K 条候选轨迹，先检查道路与运动基本有效性，再由原 Hilbert–GAF 网络对目标类别评分。保存全部候选、得分和被选索引；筛选后输出仍为坐标轨迹。正式称为类别反馈筛选，不称为已实现梯度 guidance。

主预算 K=8，预先固定种子与条件。对照包括随机选择、普通轨迹分类器选择、GAF 选择、打乱目标类别的 GAF 选择。各方法使用完全相同候选池；可附 K=1/4 的预算曲线，但不以更多候选冒充算法提升。报告重复率和多样性损失；不只展示成功例子。

若大部分候选编码相同或匹配时长上的类别信息不足，应先调整生成时间尺度/编码分辨率，并将改变标为改进版；无收益时保留筛选负结果，主模型照常推进。

### 5.4 可选的 GAF 教师引导

原 Hilbert、序列压缩与 uint8 编码不可直接稳定求导。若筛选有收益，可训练读取轨迹的可微网络拟合冻结 GAF 教师的概率/特征，再在扩散低噪声阶段使用其梯度。必须比较独立类别标签网络，并验证代理与教师一致性及分布外失真。

此模块须通过前置证据再启动，不以它阻塞第一周成果。若采用分布匹配或对比特征约束，单独列为新增方法，不包装为组内已有贡献。

## 6 实验任务和验收

| 编号 | 任务 | 产物与验收 |
|---|---|---|
| E0 | 固定任务书、源码与数据版本 | 哈希、划分、代码提交、当前状态可追溯 |
| E1 | 原编码 CNN 能力与网格对照 | 2 网格 × 2 初始化 × 3 种子；学习曲线、混淆矩阵、每类 F1、覆盖率；不是正式论文复现 |
| E2 | 时间尺度诊断与较长轨迹数据 | 64/224/512/完整轨迹编码信息统计；匹配生成长度的类别验证；选定主长度并解释覆盖代价 |
| E3 | 路径条件扩散主实验 | 复用并扩展现有模型，3 种子；拓扑、动力学、分布、多样性及生成耗时 |
| E4 | 公平 GAF 筛选 | 同候选池、同 K，与随机/普通轨迹分类器/打乱类别对照；独立评价网络 |
| E5 | 合成数据实用性 | 真实数据单独训练 vs 重采样 vs 简单扰动 vs 无筛选合成 vs GAF 合成；相同下游训练预算 |
| E6 | 可选教师 guidance | 仅在 E4 有可靠信号后实施；与相同采样预算及直接类别引导比较 |
| E7 | 锁定方案并测试、整理图表 | 一次冻结测试、失败案例、可复现实验清单、论文图表与局限 |

E1 预注册：每项 50 epoch、batch 64、AdamW 学习率 2e-4、权重衰减 1e-4、类别权重为训练频数平方根的倒数。主报告固定第 50 epoch，验证最优 checkpoint 为次要诊断；种子 42/43/44。预训练和随机初始化使用相同 ImageNet 通道标准化，不缩放裁剪已有 224×224 图像。完整轨迹为诊断任务，不能据此选择短轨迹并自称有效。初次四卡各跑一个组合，随后每卡串行补齐种子；无自动超参数搜索。

E4 推进门槛不是一个随意的准确率数字：匹配时长的真实验证需显示可区分的类别信息、候选 GAF 有区分度，并且独立评价中的收益不只是筛选老师自评分。若首轮失败，优先检查时间尺度，不自动启动教师蒸馏。

## 7 评价方法

1. 道路合法性：到类别许可车道距离、有向连接一致性、逆行/跳路、路径耗尽；分解结构约束和模型学习的作用。
2. 运动真实性：速度、加速度、停留比例、行程长度及其分布距离；明确时间单位和计算方式，按类报告。
3. 类别一致性：与筛选教师独立训练的轨迹模型、混淆矩阵和每类指标；不同种子但同结构并不等于完全无偏，须辅以分布和下游评价。
4. 多样性与记忆：同条件轨迹距离、路径覆盖、重复率、与训练样本最近邻相似度；反馈不得靠只选少数模式提高分数。
5. 下游价值：真实留出集上的 macro-F1、稀有类召回；合成类别与数量固定，其他增强对照匹配训练步数。低标注实验若使用更多标签训练生成器，必须披露，不称严格低标注。
6. 不确定性：至少三训练种子，车辆级聚类重采样置信区间；不把相关窗口当成独立样本。开发过程的多重比较和选型需披露。

目前任何验证探索结果都不称最终测试结果。GAF 有效的结论应综合收益、代价和多样性，不能仅报告单一最好数字。

## 8 一周执行安排

本轮启动起七天为目标窗口，按依赖推进，不承诺未验证的精度或所有可选模块完成。

| 时间 | 主任务 | 必须留下的结果 |
|---|---|---|
| 第 1 天 | E0/E1 启动与 E2 诊断 | 正式任务书、四卡运行证据、编码与类别问题定位 |
| 第 2 天 | 固定时长口径，E3 长轨迹训练 | 数据 manifest、训练曲线、轨迹样本 |
| 第 3 天 | E3 评价、E4 候选池及筛选 | 公平筛选表、地图轨迹图、失败案例 |
| 第 4 天 | 核心三种子、E5 下游启动 | 类别与多样性收益区间、增广训练记录 |
| 第 5 天 | 补关键消融，视证据做 E6 | 有边界的正/负结论；不无限堆新路线 |
| 第 6 天 | 冻结方案与测试 | 最终配置和一次性测试表 |
| 第 7 天 | 论文材料整理 | 方法图、实验表、消融图、案例图、章节提纲与局限 |

硕士工作量来自数据与道路协议、扩散生成框架、GAF 接入、对照与消融、下游实用性和可复现材料，不靠无依据增加训练次数。优先完成核心 E0–E5/E7，E6 可选。

## 9 工程与文字记录

每次运行保留代码提交、参数、依赖版本、数据哈希、随机种子、训练日志、checkpoint 和完成/失败标记。每轮写清问题、控制变量、实际结果、结论边界和下一步；保存负结果。

仓库只推代码、配置、小型指标与文字图表，不推原 CSV、模型权重和凭据。服务器实验放隔离工作树，避免覆盖历史未提交文件。推送后核验远端提交；长任务后台运行并保留状态，不能声称聊天结束后会自动开展未实现的新实验。

每次恢复工作先读本任务书与最新日志。若计划新增任务，先回答：它改善了哪个生成环节？能用什么对照证明？一周内的优先级是什么？答不清的任务先不执行。更改主线或核心数据协议须记版本、原因、影响和对应用户要求。

## 10 第一轮执行状态

- 任务书已制定；E1 代码与启动证据将在实验日志单独记录。
- E1 沿用已有完整轨迹数据划分；增加 64×64 网格和 ResNet18 初始化对照。
- 既有 64 点路径扩散成果继续保留；E2/E3 的长度匹配尚待执行，不因 E1 启动而宣称生成主线已完成。
- 当前无组内 CNN 权重，不以获取其权重作为阻塞条件。

## 11 文献与版本边界

- 原代码：[历史仓库](https://github.com/LiuJingting0201/Deep-Learning-Based-Generative-Models-for-Traffic-Trajectory-Synthesis)，本地 vendor 保存指定提交。
- 正式论文：[IEEE 11258590](https://ieeexplore.ieee.org/document/11258590)。当前读取了摘要；正式全文仍未获取，不使用相关学位论文替代其全部方法和结果。
- 相关学位论文：[Elisa Salvadori 2024](https://webthesis.biblio.polito.it/30378/)，提供方法参考；不宣称是正式期刊版本。
- [Torchvision ResNet18 文档](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html)：本轮使用明确版本的 ImageNet 权重；仅采用通道标准化，保留完整 GAF 图像，不沿用自然图像裁剪。

版本记录：v1.0 根据用户确认的“先写详细任务书，然后开始实验”制定。后续变更在此追加，不能静默重写研究目标。


## 12 迭代记录

2026年11月4日，E1–E5验证开发完成一轮，E5尚未证明稳定的下游增益。按原框架“候选路径+运动扩散”追加E4r：只改变候选路径多样性，冻结生成模型、GAF构造和分类器、K与评价条件。它是E4生成侧的有限对照，不是E6可微教师模块；不扩展为分类调参，不覆盖原负结果。完成后依据跨家族评价、运动质量和覆盖决定是否值得进一步验证，否则进入局限与可复现材料整理，保持测试冻结直到最终方案明确。

E4r已完成，路径多样性增加但GAF仍无可靠跨家族收益，不满足E6启动门槛。追加一个必要的E3b经验运动迁移基线（完整训练速度剖面映射到相同合法路径），隔离扩散学习相对于路径结构与经验采样的贡献；协议见最新实验日志。其后进入证据汇总与最终评价准备，不继续分类架构搜索。

E3b已完成：扩散在全部候选运动分布误差上点估计较低，经验基线候选差异更大；尚未建立显著性和无记忆结论。进入E7准备，先补已保存候选的配对区间与运动记忆诊断，再锁定正式测试协议。GAF收益仍作为待验证/负结果报告，E6不启动。

2026年11月5日，E7验证审计已完成：加速度W1的车辆配对区间未跨零，速度/行程跨零；全部区间条件于既有模型和候选，未作多重比较校正。未发现完整对齐近复制，非无记忆证明。最终测试协议候选见 docs/FINAL_TEST_PROTOCOL.md，尚未锁定或解封；先完成执行入口检查及哈希清单，再一次性正式评价。

最终执行代码02ba1dd的19项解封前检查及217个文件核验通过；FINAL_TEST_PROTOCOL v1.0现锁定。实际测试访问还需locked机器清单验证，运行状态另查最终队列。锁定后不调整模型、候选预算、编码或阈值，失败修复必须留档；最终评价和材料汇总完成后暂停本轮自动迭代。

2026年11月5日，已核验锁定正式测试全部完成，并生成固定地图案例、方法图、配对效应图及论文结果总述。核心E1–E5/E7完成，E6因证据门槛未满足未启动。正式结果见 docs/THESIS_RESULTS_SUMMARY.md 与 docs/results/final-test-e7/REPORT.md：路网条件扩散合成跑通，行程分布有条件改善；GAF稳定下游增益未建立，TCN增广出现负效应。测试已经暴露，禁止继续用它选择模型或阈值。按原约定结束本轮自动模型迭代；后续写作或另开研究轮次需保留现有正/负结果与覆盖边界，不改写为GAF有效。

2026-11-05 新授权开发轮 G8：在固定已规划道路上鉴别原 Hilbert 图对运动时序的敏感性，并比较一次低噪声重采样与等预算末端选择。只使用训练和开发数据，旧测试不作新方法评价；不改第一版结论。协议见 docs/G8_PROTOCOL.md。

G9: 仅开发集，以同权重同预算的终端筛选作主对照，比较三个引导时点与两种道路转向程度；以行程/速度/加速度、多样性及道路检查解释适用范围。结果见 docs/results/g9-timing。

G10 在同一 MoST 路网上用八个独立新 SUMO 需求种子复核 G9 的弯路速度信号，仅固定第76步与相同权重的末端选择比较；不使用此前参考轨迹、不调模型。设计与主终点见 G10_PROTOCOL.md。结果回收以前不更新论文。

G11 改问原编码是否能在候选运动已可区分时改善输出。事先固定基于运动变化与候选间差异的触发规则；全新八批仿真，末端匹配及类别内打乱触发时机的对照，按仿真批次评价。

G12 在原训练 ID 内分离拟合、checkpoint 选择与温度校准；比较原编码 CNN 的分类、概率质量及同路线候选排序效果。独立评估用另外八批新 SUMO 场景，不打开旧正式测试。规范见 G12_PROTOCOL.md。

G13 使用另外八批独立 SUMO 请求复核生成器的正面主张：相同路径与噪声下切换公交/私家车类别。汇总全部共有许可路径，按固定规则展示逐时运动实例；将组内表示对分类、局部筛选、整体合成的证据分开呈现。结论按实际复核结果填写。

2026-11-05 用户授权使用 SUMO 扩充数据。新增 G14，按完整 episode 划分训练、验证与留出；区分原始 CNR 数据、新 SUMO 观测和扩散输出。本阶段仅制作与核验数据，不更新已解封的测试结果，不推断扩容后的模型效果。见 docs/G14_DATA.md。

2026-11-05 G15 按用户要求扩充参与者类别与独立样本规模。目标八类各一万个独立 ID，整场种子划分，平衡抽样。行人使用 SUMO person，配送车与货车使用独立 vClass；新标签仅属于新数据与未来模型。资格为 SUMO 原生道路标识与连续窗口检查，与原五类严格重建协议分开。保持公布的论文评价不变。

2026-11-05 G16：按用户授权训练匹配预算五类原始/扩容扩散模型，再独立训练八类。固定局部路径表示、评估指标和两个训练种子；行人先限定单条可通行人行道。只用开发分区，原测试与 G15 留出均不用于选型。见 docs/G16_PROTOCOL.md。

2026-11-05 G17：G16 短预算结果未支持普遍增益，追加一次预先定长 20,000 步复核。主终点改用独立 G14 验证场次；复核原 GAF 在同候选池内的速度与行程收益。结果无论方向都保留，停止本轮扫描。

2026-11-05 G19 完成。围绕路网下不同类别的运动模式，用 G18 指标做等预算扩散训练；将原 Hilbert/GASF/GADF/MTF 适配到自训八类 CNN 并以相同候选预算检验反馈。主终点与代价、分类范围和编码继承见 docs/G19_PROTOCOL.md 与 docs/results/g19-motion-diffusion。该有限比较结束，不按观察到的开发结果自适应刷配置。

按用户最新定义：主产品为完整 OD 行程，区分轨迹 GAF 的参与者类别识别和速度 GAF 的运动模式识别。速度特征不得进入轨迹类别评分器，速度 CNN 不以车型为目标。历史运动输入→车型结果仅作信息诊断，不作为这两种引导的优劣证据。使用训练侧类别条件模式分布连接速度反馈与车型请求，并用同候选、独立完整仿真行程的物理质量比较。见 G22_FULL_TRIPS.md、G23_PROTOCOL.md。

G24 已完成整程采样中速度 GAF 引导与独立轨迹 GAF 评分实验。冻结空间读图器只参与最终评价，配对比较含置换目标和末端筛选。结果与限制见 docs/G24_PROTOCOL.md 和 docs/results/g24-speed-guidance；不因主要区间跨零扫描时间步或强度。

G25 对既有 G24 做表征审计，比较真实车流上的跨图对齐及固定条件下的生成分布变化。速度分支仍只负责运动模式，不以速度辅助空间车型预测。这是事后评估，不据此扫描引导强度。见 docs/G25_PROTOCOL.md。

G26：授权的后续训练对照聚焦完整行程。固定 G22 主干、步数和双种子，检验分层进度、训练侧运动模式条件、直接速度监督与速度 GAF 成对关系监督的增量。完整路线约束保留，空间 GAF 不以速度为车型输入。先锁协议，再看新独立开发车流。见 docs/G26_PROTOCOL.md。

G26–G27 已完成训练侧对照。整程轨迹主模型采用连续进度、可指定或按车型和路长抽样的类内运动模式（HM）。组内空间编码继续用于独立的参与者分类评价；速度角场单列为去噪训练辅助损失，与直接监督速度曲线进行公平对照。类内运动可控性已复核，同地图的新 SUMO 车流尚不能支持角场优于直接监督或总体保真度全面提高。结果及覆盖范围见 docs/results/g27-continuous/REPORT.md。

2026-11-06 G28 用户明确提出空间与速度 GAF 均含车型信息，授权对齐两支同一车型任务。新一轮允许纯速度 GAF 预测参与者类别，以检验双表征潜空间监督；其结果与旧六模式识别分述。先修复空间训练/评价的端点处理差异，审计轨迹重构误差，再比较无对齐、类别原型、空间、速度、双支和类内路长匹配打乱。阶段目标与停止规则见 docs/G28_PROTOCOL.md。旧测试不参与本轮。

G28 已完成，同标签潜空间直接监督实验及独立新车流复核没有建立稳定生成增益。当前证据支持速度表示中的车型信息；空间增量和双支互补仍未建立。新主张须同时经过独立表征读出、正确配对的监督对照、全候选物理质量与多样性评价。G28按协议结束，保持原始数据研究、固定窗口、完整轨迹和新仿真证据的范围。详细数值与失败见 docs/results/g28-paired-latents/REPORT.md。

2026-11-06 用户要求先对照原始CNR诊断空间GAF低分。暂停生成实验迭代和论文新增负结果。新增G29数据来源×编码阶段诊断，统一共同六类、样本数、冻结空间编码和训练预算，并核对原E1全观察正控制及固定512点窗口。结果先形成独立诊断报告，明确原始CNR空间车型结构与新随机路线仿真差异后再决策。见 docs/G29_PROTOCOL.md。

G29已完成30个预定分类拟合及原E1像素/权重回放。共同六类1983训练/257验证下，原CNR与新SUMO的纯几何宏F1为79.12%/34.85%，GAF为59.88%/30.15%，运动特征为88.08%/80.57%。新SUMO保留部分车型运动信号，但未复现原CNR的类别—路线关联及公交停留分布。原GASF数值变换可逆到归一化输入；单独网格量化影响小，Hilbert标量读出和逐轨归一化需分述，不能把CNN差值当信息论损失。见 docs/results/g29-source-audit/REPORT.md。继续暂停生成调参和论文扩写；下一步先修正并验证数据需求分布，保留共用道路与类内多样性，不能靠人为专属路线提高分类分数。原观察段首尾只是OD代理，缺少原SUMO需求文件不得声称完整复现。空间反馈的作用层级须对应其实际可读出的路线/地理结构，再与固定路线的速度模式反馈比较。

G30用户授权继续：找到与现用路网逐字节一致的官方MoST原生需求，固定版本接入公交线路、站点和活动链，先pilot兼容性再有限6训练/3开发场次。仅改仿真种子不称新OD；按原实体与场次双重隔离，原测试ID对应重放实体排除，原train/val实体遵循既有分区，未知才hash。先比较CNR、随机需求、原生需求的几何/运动分布及最多6个固定GAF拟合，不预设优势，不修改论文；依据数据结构决定后续生成反馈层级。协议见 docs/G30_PROTOCOL.md。

G30完成：原生需求GAF宏F1恢复到58.69%（CNR59.88%、random30.15%），纯几何76.95%（79.12%、34.85%）；10项宏运动分布距离均降低，行人部分指标仍恶化。新结果支持需求域不匹配解释，不能将空间低分笼统归因于原编码，也不能把分类恢复称作生成收益。共同需求模板和训练见过的公交线路限制泛化；完整结果见docs/results/g30-native-demand/REPORT.md。接下来G31只做单场完整行程接口检查：事件切leg、预定路线权限、真实目标重构；通过后才锁新生成对照。路线空间反馈与路线内速度反馈是待检验的分工，不以分类上升直接启用强guidance。保持论文暂不改动、旧测试不用于新选型。

G31接口检查完成、未达生成ready：1644完整事件/采样段来自1592parent，公开源计划与实现路径分列；587段需换道适配、593步行适配、49只有OD需事前规划，415静态lane链仍未做端点/站点/投影联合解码。现有257时间进度对长行程有重构损失：公交速度MAE0.424m/s、停止比例均值降低3.19pp；≥900秒段巡航状态不一致24.0%。应先接通真实完整leg的地图条件接口，比较固定物理时钟变长表示的重构保真度，保留连续进度而不回到硬分块边界；之后再锁定空间/速度GAF对生成的有限对照。不得删除长行程以回避误差，不把oracle重构称生成效果，不将此诊断倒推为全部历史负结果的原因。报告与范围见docs/results/g31-leg-interface/REPORT.md，论文仍未改。

2026-11-06 用户继续授权，并要求明确论文下一步。新增docs/THESIS_FINISH_PLAN.md，将主线收敛为可信需求下完整物理时间轨迹扩散，以及对应生成自由度的双GAF反馈。空间/速度两支沿用G28更新，均可预测参与者类别，输入严格分开；新五类车辆生成与教师统一标签，不能套用G30六类最长段准确率。先做两个生成版本各两种子，再做同池四种末端反馈及一个固定噪声区间采样反馈，不扩成无限分类/引导扫描。新结果完成后按八章重组论文；旧正式测试和失败保留附录，不反向调参。

G32固定协议f957af3先推送，随后隔离g32-job完成三臂全部1644段比较；旧G31坐标/汇总严格复现，未运行新仿真或训练。1秒表示使逐秒速度/加速度MAE 0.1647/0.2433→0.0101/0.0125，但原生0.25秒加速度MAE 1.0514→1.1542。原生时钟臂速度/加速度MAE为0.0093/0.0314，全部段这两项优于1秒。故下一模型以0.25秒变长序列为目标，1秒保留摘要评价，需长度桶/mask和连续全程建模；平均1714点、最长9419，算力开销尚未实测。共用2049点真实空间路径仍有尾部残差，不是地图接口或生成通过。见docs/results/g32-physical-clock/REPORT.md。地图适配构造与公共源请求覆盖另记，未达新训练启动条件。

G32地图接口完成有限构造验证：五车辆class实现显式事前路线/端点/站点到精确地图折线，行人尚未支持。v1公共24公交请求11条通过；首失败诊断定位整段换道混合偏出许可走廊，而非源路网无连接。v2规则先以e0592b7发布，保留全部请求SHA f38fee3fc0190799a340c29ea81d54ea5c1d82dfbf125aa0b32fddddff7a570e，改为最早可行10m几何换道窗口，禁止放宽许可/走廊容差。33项构造检查及单次24请求复核通过，21构造成功、3未支持；原11全部保留，成功计划含235/298站点。旧代码/回执、逆向patch、两次开发调用/fixture失败保留。静态几何不是动态仿真或训练目标投影成功，不称五类真实native覆盖完成。下一项G33需预先写固定请求/数据预算/分区，做少量新SUMO参考与同一事前路径上的目标对齐；通过后开始两个完整轨迹扩散版本各两种子的有限训练，不能继续无限地图审计，也不能用真实未来路径补齐正式生成条件。


2026-11-06 G33按用户最新纠正收拢为“扩散轨迹生成、GAF对生成的作用”。取消自制车道解码通过后才能训练的前置门槛：SUMO原生输出直接作为训练路线—运动配对；训练观测路径是监督，开发真实路径仅能标oracle诊断，正式生成只用独立事前地图请求。新协议docs/G33_PROTOCOL.md分别以29771d7、95310bb、af70ba4、7085591先发布数据/仿真/训练/生成评价规则后执行。完整五类车辆1051段(925train/126val)，父ID划分不变、不按几何成功筛选，当前是单pilot启动集而非规模扩容成果。

四路891969参数masked U-Net实际开训：257匹配表示与原生0.25秒各seed42/43、10000步，训练侧共享因子化时长先验，不能称时长由扩散生成或复刻原G22。两条257已到10000步，两个五类GAF教师已出现完成标记；原生模型仍运行，完整质量结果未完成。固定训练/反馈组件完成不等于GAF有生成收益。下一步固定K8/DDIM100，全部候选独立物理评价；只有原生模型作Uniform/S/V/SV同池主比较，257和经验迁移作生成基线。采样中反馈尚未执行，不冒称完成in-loop。

独立SUMO参考100请求全部出发到达，实际边序列等于冻结计划；97通过预定参考检查(公交22、摩托18、其余各19)，3失败不补样。新背景随机数/出发时刻/明确端点不等于新城市或未见线路。目标teleport1条已在拒绝项，另有接受的on-demand紧急停止。新生成参考不得反向进入训练/时长先验/选型。源计划解码失败和参考失败分别留在100分母，逐项物理W1按请求等权、请求内候选等权；缺失类别不得静默缩小五类宏平均范围。


2026-11-06 G33完成状态更新（替代上文“原生仍训练/评价未完成”的当时状态，保留历史记录）：四个10000步扩散模型、两个固定50轮五类GAF教师、六个各776候选的全池生成及原生两种子×四策略选择全部完成，14项评价回执齐全。独立100请求中地图支持97、合格参考97，质量交集94/100；每项选择交付97条，失败没有补样。训练耗时257为5.01/5.31分钟、原生为25.33/25.49分钟；质量收益和计算代价分别报告。

当前可支持的生成结论：原生0.25秒模型相对匹配257对照的原生加速度分布W1下降82.23%/81.07%，并优于经验运动迁移；但最长停车及p95速度等指标未同步改善，不称所有运动模式更真。空间/速度GAF开发宏F1为69.80%/84.99%，193/194个池有8张不同空间图，已能改变候选排序；最高类别分选择仍未形成普遍独立物理增益，不等于GAF应当弃用，也不声称互补已经成立。完整结果见docs/results/g33-generation/REPORT.md。结论限于单训练pilot、单新SUMO开发参考、公共道路模板与固定路线；旧正式测试不回流选型，本轮没有采样中guidance结果。

下一有限工作为已锁定的G34（ab2c501，docs/G34_PROTOCOL.md），当前实现准备，未报告启动或完成：固定原Uniform训练先验T，同路线/车型/总时长下比较空间/速度/双支末端反馈对运动模式的作用。保留G33结果，既不使用参考未来时长，也不为追正结果重选T、教师、权重或种子。主要可变评价为停走/巡航、速度分布形状、加减速及四秒变速；时长不变不当作GAF收益。主线仍仅完整道路条件扩散轨迹合成与GAF的生成反馈，后续依据完成证据更新。

2026-11-06 G34完成状态更新（替代上段“实现准备”的当时状态）：两个固定原生模型各完成K8/DDIM100生成及Uniform/S/V/SV四策略，10项评价齐全。每种子97地图请求生成776候选，质量配对仍94/100；四策略各输出97条。新旧Uniform逐点相同、最大时间/坐标差0，全部候选与选择共享从训练先验重演的T，没有使用真实参考时长。失败首试因G34装饰器源码定位错误在生成前被hash检查拦下，修复后使用独立attempt2目录，G33代码、权重和首试失败均保留。

当前可支持的GAF结论收紧为：固定路线、类别及采样总时长后，速度GAF末端选择在本次两种子宏平均中降低原生加速度W1 9.73%/17.22%、最长停车W1 22.49%/6.53%，停止比例与停走频率亦降低；但最长巡航和p95速度两种子均恶化，原生速度及四秒变速方向不一致。原生加速度并非每车型均改善，空间/双支也没有稳定增量。此结果说明可检验的局部运动反馈作用，不证明全面合成质量、GAF优于同信息其他表征、双支互补或采样中引导。G33与G34结果并列保留，恒定时长和微小均速差不作GAF成果。

本轮沿用已查看的单场G33开发参考，属于由先前结果启发、候选执行前锁定的机制诊断；不是新盲测，没有新训练、仿真或增广，没有访问旧测试。两次生成和八次选择评价已按协议结束，不据此追加时长/阈值/权重/种子扫描。完整物理结果、逐类异质性、覆盖、代价和失败见docs/results/g34-fixed-duration/REPORT.md；论文继续围绕完整道路条件扩散轨迹合成与GAF的实际运动反馈组织证据。


2026-11-06 用户确认：若新SUMO数据支持原轨迹GAF参与者分类作用，即扩大数据并开训。G30同预算CNR/native准确率74.12%/73.35%、宏F1 59.88%/58.69%为扩容依据，不能替代生成收益。新增G35完整10小时原生需求，两训练场次与一开发场次、parent事先固定实现并按ID分区，目标至少一万个独立训练parent；保留公交线路/站点/活动链，不回到随机OD，不把点或重复seed计为新增车辆。自动数据队列通过检查后进入4个10000步扩散拟合（两时钟×两种子）及2个固定50轮匹配GAF教师；全部完成后停止。本轮尚未有新规模或质量结果，论文暂不更新。协议docs/G35_PROTOCOL.md，原始CNR继续作为来源对照，旧正式测试不回流。

G35现已实际启动数据扩容队列，并登记数据完成后自动执行的训练队列（执行版本ecfb7a7）。首次进度确认第一场仿真在推进；模型尚等待新数据，不冒称已训练。当前阶段是全日扩容，不再停留在925条pilot。

2026-11-06 用户继续授权G35训练后自主分析与迭代，优先同新验证集上的GAF识别和独立生成质量，结果复核后更新论文；按ETA节点查看，不持续刷新。分类分析先锁docs/G35_ANALYSIS_PROTOCOL.md，G35正在运行的四扩散/两教师训练配置不改。后续G36另锁新请求/参考/同预算生成协议，旧测试不回流，不以分类高分代替生成收益。


2026-11-07 G35b: execution amendment fixed before launch in docs/G35B_EXECUTION.md. Three CPU SUMO scenes run concurrently with predeclared FCD identity lists; native background demand, split, time and model budgets remain. The v1 partial attempt is preserved; no completed data or speedup claim yet. New isolated g35b-job only.


G35b launch confirmed after commit 307ec8d: three CPU SUMO branches are progressing concurrently; all 43 focused checks passed on server and runtime hashes match. The saved 160-second checkpoint is docs/results/g35-parallel/launch-and-progress.json. Shared-interval per-scene compute time is essentially unchanged under concurrency; summary timing excludes output I/O and no exact full-day speedup/ETA is claimed. Old partial v1 attempt was gracefully canceled and preserved, local archive is pending checksum completion. Four diffusion fits and two GAF teachers remain automatically queued after data readiness. No new model result or thesis change. Details and deployment/approval limitations: docs/results/g35-parallel/STATUS.md.


2026-11-07 storage cleanup completed under user authorization: removed only unused pip HTTP cache and conda/mamba compressed downloads, 19.21 GB logical bytes total. Free disk rose to 32.69 GB after cleanup; all three SUMO processes remained active and no queue failure appeared. Environments, ImageNet weights, original data, all experiment outputs and old worktrees are preserved. Local v1 archive was found interrupted at 2.06 GB; server FCD was not deleted. Existing cloud data/training queue no longer relies on local archive transfer for disk headroom. The separate 15 GiB attempt-output cap remains unchanged and must be assessed against final artifact sizes. Receipts and limits: docs/results/g35-storage/REPORT.md.


2026-11-07 G35c capacity handoff preparation: at 01:35 Beijing existing scenes are 63/61/71 percent complete, with 9.98 GB output and 27.60 GB free. The original independent 15 GiB attempt limit risks stopping valid ongoing simulations; new execution policy docs/G35C_CAPACITY_HANDOFF.md preserves exact SUMO processes and scientific settings, uses identity-checked wrapper handoff with true kernel exit-status capture, and counts old/new roots jointly under 40 GiB plus at least 8 GiB filesystem reserve. New code stays in isolated g35c-job; old hashed source is not edited. Single-process extraction and mathematically equivalent interval storage are being validated. This is preparation, not an executed handoff or new model result.


2026-11-07 recovery correction: the proposed G35c handoff did not complete before the old 15 GiB threshold. Two training simulations were interrupted at 44122.50/42455.75 and original failure receipts confirm the capacity error; SUMO zero exit after graceful interrupt does not mean complete. Validation350201 is fully complete and preserved. First adoption launch rejected the already-completed validation scene before signals; frozen original train parents were subsequently resumed only to record the real failure. New G35d protocol reruns only the two train seeds, preserves validation/failures, and applies the combined40GiB/free8GiB policy throughout simulations and training. Old03:00-04:00 estimate withdrawn. See docs/G35D_RECOVERY.md. Launch status follows separately.


2026-11-07 02:19 Beijing G35d recovery verified active: commit8a0a496 launched detached cloud queues02:16:15; two training seeds rerun in parallel and both observed clocks advance, completed validation350201 preserved/extracting. Combined old/new roots16.32GB under40GiB; free39.995GB with8GiB reserve. Lossless full-SHA-verified compression of two inactive historical G21 image caches released18.743GB; original contents retained as gzip, raw trajectories/weights untouched. No new failure marker; four diffusion fits/two GAF teachers remain queued (zero started). Prior interruption/failure and withdrawn ETA remain documented. Read-only guard review confirms no obsolete15GiB threshold on the new execution path. Snapshot/docs: docs/results/g35-recovery/STATUS.md. No new scientific result or paper update.


2026-11-07 11:58 Beijing: G35d data and all six training fits complete, artifact hashes verified. Train29895legs/24073parents; val7230legs/5842parents. Spatial GAF val accuracy86.17%,macroF1 76.47%; speed89.13%,90.34%. Spatial taxi/uber F1 remains45.85%; imbalance and train–val gap retained. Four diffusion fits each10000steps complete, but expanded generation quality not yet evaluated. Preregistered same-val old/new teacher audit launched; no new fitting/test access. Free disk21.23GB. Results docs/results/g35-training/STATUS.md; no thesis update until generation evidence. Initial read-only status query quoting failure corrected without affecting jobs.


Same-validation audit completed: both old/new models evaluated on identical7,230 legs/5,842 parents. Spatial accuracy60.65→86.17%, macroF1 50.10→76.47%; speed accuracy63.61→89.13%, macroF1 61.82→90.34%. Paired parent-bootstrap95% intervals for macroF1 gain are spatial[24.54,28.16]pp and speed[27.27,29.73]pp. This supports improved classifier generalization on this public-demand development cohort. Scale, full-day coverage and number of optimization updates changed together; it is not a pure data-size causal effect, unseen-city test or demonstrated generation benefit. Full fixed analysis: paired-teacher-audit.json.


2026-11-07 G36 fixed before launch: two new SUMO episodes,100public conditions each, paired old/new native generators×seeds42/43. New pools scored by both old/new GAF pairs to isolate teacher improvement; all physical endpoints and coverage retained. Pending execution, no benefit claim. Scope narrows provisional250×2/activity-chain design to existing verified100-request interface across two time windows; limitations explicit in docs/G36_PROTOCOL.md. Local bytecode check initially failed only because Python cache path was sandbox-protected; in-memory syntax compilation passes.


G36 cloud queue launched after all four preflight groups passed and source SHA checks; commits39f30fd/5345f60. First SCP disconnected before deployment; safe transfer retry succeeded. Reference requests freeze before outcomes, original worktrees unchanged. Launch receipt docs/results/g36-generation/launch.json. No new physical result yet; actual progress verification follows.


G36 completed:56evaluations/64.9minutes, paired support95/94of100per scene. Native expansion speedW1 worsens4.7/12.2% in morning, improves57.2/65.5% later; acceleration also mixed. Higher-accuracy GAF teachers do not consistently improve physical selection: new speed-GAF accelerationW1 delta+8.7,+0.9,-9.6,-4.7% versus Uniform. Full metrics and limits docs/results/g36-generation/STATUS.md. No overall-benefit thesis claim, no additional adaptive sweep started.


2026-11-07 G36 post-hoc statistical audit: original W1 correctly uses class/request/candidate/time weights; retain it. Added complete endpoint MAE/W1 and class stratification, descriptive scene/seed-equal aggregation (no pseudoreplicate inference). Spatial GAF longest-stop paired MAE improves7.6% across four cells, bus20.7% but motorcycle worsens5.3%; adverse speed/stop-frequency distribution metrics retained. This is exploratory mechanism evidence, not overall benefit or significance. Code checks pass; docs/results/g36-statistics/INTERPRETATION.md locks interpretation and outlines independent replication; no new experiment or thesis claim yet.


2026-11-07 thesis v17 completed under explicit user request: integrates G35/G36 and post-hoc audit into abstract, introduction, data, protocol, results and rewritten discussion.87PDF pages/60main text,18figures/20tables/52cited references (47papers/preprints).8new vector figures and7new tables; all generated numerical assets trace frozen aggregate receipts. All87pages rendered/contact-reviewed; heatmap/confusion/class figures checked full-size. No overfull/undefined references; clean Overleaf ZIP build matches delivered PDF page text. Administrative fields remain placeholders. Exploratory stopping effects and adverse endpoints shown together, no universal GAF/in-loop claim. Build/assets receipts docs/results/thesis-v17-*.json; delivered locally under outputs/thesis-v17.

2026-11-07 用户范围调整：论文主线是五类参与者的路网条件扩散轨迹合成，以及空间/速度GAF对生成运动的作用。公交、摩托车、私家车、自行车、出租/网约车均须正文展示轨迹、GAF和独立质量；公交停车不得代表全部成果。小样本/扩容及旧新教师比较降为附录支持诊断，不宣称scaling law。既有完整结果不删除；类/运动特征分层为事后描述，不据此暗中切换最优策略。后续更多SUMO数据优先独立场次、少数类与服务/路网覆盖；新增车型须具备可追溯仿真定义与道路权限，并与原CNR标签明确区分，禁止只改ID/速度参数制造容易分类的新类别。本轮先完成已有五类正文；新数据队列尚未启动。

2026-11-07 用户纠正任务范围：路线选择本身属于车辆运动模式。后续模型目标扩展为OD+类别条件的路线与运动联合合成，公交保留已声明服务站点条件。启动G37有限开发实验：先从地图生成多条合法路线，再由现有扩散生成运动，等8候选比较单路线/多路线及S/V/SV。候选路线暂为图算法，不冒称学习到的选路；已有G36参考只作开发诊断，不作新独立路线选择测试。协议docs/G37_PROTOCOL.md，结果未出前不改论文结论。下一阶段需训练侧路线偏好与独立OD/场次评价。
2026-11-07 G37完成：OD多路线接口20/20覆盖，19有多条候选，72条路线/320条扩散轨迹。空间GAF改选14/20条路线，但多路线池加速度W1改善21.1%同时速度W1恶化29.5%，不能写成整体有效。生成器未读取真实内部路径；路线仍由图算法提议，参考为已用开发数据的一次实现。详见docs/results/g37-od-pilot/REPORT.md。下一步须训练侧学习路线偏好、匹配非GAF对照、独立OD/场次复核；本有限队列已经结束，无新训练启动，不将“改选”当“选得好”。

2026-11-09 用户授权G38：训练目的地/下一公共站点条件的道路选择概率，线性与神经策略各两种子；冻结既有扩散与GAF，在新公开OD及两场SUMO实现上做同候选预算的R/S/V/SV组合对照。按路线分布、候选覆盖、多样性与完整运动指标评价，不把分类自评分当收益。固定协议docs/G38_PROTOCOL.md；当前为实现准备，启动/完成须查回执。

2026-11-09 用户授权尽可能完整推送项目及大文件。实际核验主代码仓库为公开库，新增私有 cnr-road-gaf-artifacts 保存原数据、权重、全部实验快照与源历史；不改变公开仓库可见性，不上传凭据。此项授权只改变归档目的地与覆盖范围，不改变冻结测试、原始数据来源或科研结论。状态与恢复入口 docs/GITHUB_ARCHIVE.md。


2026-11-09 用户授权 G39：先固定计划、四卡并行拟合，再自主检查独立轨迹质量与 GAF 增量。docs/G39_PROTOCOL.md 锁定七臂两种子原生扩散特征监督、八新 SUMO 场次、同路同先验时长的直接输出评价，以及有界机制/增广跟进。尚未启动或形成结果；不把 G28 重试包装成新方法。


G39 continuation fixed before generation outcomes: docs/G39_UTILITY_PROTOCOL.md and bounded isolated g39u/g39f queues. Utility26fits, independent raw-series evaluator, fixed balanced1000real/1000synthetic, separate nonbus OD cohorts; shared bus-template limitation. Factorial old-scene diagnostic preserves route and route×prior-duration marginals with equal expanded pools. Queues wait for prior stages and free GPUs; no unlimited adaptive sweep or premature thesis claim.


Current execution/resumption entry: docs/G39_RUN_STATE.md. Inspect the existing g39/g39u/g39f detached supervisors and completion/failure markers before any new launch. Their dependency chain is active; g39m route-marginal diagnostic is complete. Planned physical/utility interpretation and thesis changes are not yet completed.

2026-11-09 用户要求当前批次先继续完成，之后再优化GPU利用效率。现有G39及已排定后续实验配置保持不变；完成核验与结果审阅后，在隔离目录进行有界性能短测，优先缓存输入构造、减少CPU/GPU同步，保持有效批次、样本与噪声流并核验数值一致性。首轮最多60分钟，不自动追加完整科学训练；以端到端吞吐和一致性决定是否采用，不能仅追求利用率数字。具体顺序见docs/G39_RUN_STATE.md；这是下一阶段授权，尚未启动性能优化任务。

2026-11-09 用户进一步要求审查GAF接入方式，当前整批结束后再设计下一轮。候选机制包括全局中间特征监督与实际输出脱节、路线/总时长固定造成作用位置不匹配，以及整程速度压缩削弱短时运动信息；均为待检验假设。先保留完整主结果、独立增广和路线×时长机制对照，再决定是否直接约束解码输出及其物理时间上的GAF。不得仅换名称重跑G24/G27，不得把硬Hilbert/MTF声称为完整可微。候选审查见docs/GAF_INTEGRATION_REVIEW_PENDING_G39.md。性能候选实现已在新隔离目录通过CPU输入一致性检查，完整短测仍待本批完成与审阅；协议docs/G39_PERFORMANCE_PROTOCOL.md。

2026-11-09 21:39北京时间后G39整批112主单元/26增广拟合/4机制单元完成并核验。固定联合路线/时长分布后速度GAF分数重加权仅约−0.12%速度W1/−0.13%加速度W1，且跨单元不稳定；空间/双视图总体不利，不支持继续加大分类分数guidance。用户改为优先讨论论文故事与具体GAF机制，性能门保持未放行。下一技术设计为G40：保留原空间Hilbert–GAF及独立速度GAF构造，用轨迹去噪与车型识别共同训练编码器，将其表征作为全采样过程的条件，并学习仅训练数据拟合的条件表征先验，解决新生成时无真实未来GAF的问题。原始编码离散步骤不求导，不生成图像再数学反解；正式评价仅用先验采样条件，已知轨迹GAF只允许独立重建诊断。方案、对照与失效条件见docs/G40_GAF_GENERATIVE_CONDITIONING_DESIGN.md；尚未实现、启动或形成收益，执行超参数和源码清单须在运行前锁定。既有负结果及原测试边界保留。
