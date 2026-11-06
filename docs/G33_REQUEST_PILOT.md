# G33 独立 SUMO 参考请求：固定 100 个，直接输出轨迹

状态：本页请求与仿真语义先于执行固定；根协议及代码 `95310bb` 发布后，单次真实仿真与提取已完成。100 个冻结请求得到 97 条合格参考，未补样或按地图构造器筛选。结果与异常见[独立报告](results/g33-reference-pilot/REPORT.md)，下文保留执行前协议。

目标是给完整轨迹扩散提供独立的已知计划条件与 SUMO 参考。SUMO 直接生成轨迹；G32 自制地图构造器不决定哪些请求能进入仿真，不筛掉其失败路线。模拟器参考、扩散输出与解码器覆盖分开记。

## 冻结请求

- 公共源为已校验的 MoST `b29b2f65f1096a9c69a601ec62a724815cb4a43f`，网络和输入逐文件校验。
- 全部 24 条公共公交服务各一条目标车，完整保留原 standalone route 的边序列、站点、duration/until 和 line。不按 G32 的 21 条成功路线过滤。
- 从 `most.pedestrian.rou.xml` 的 `ptw/passenger/bicycle/on-demand` 各取 19 个已有公开、无活动停车的单段 vehicle 路线，共 76 条。先按原 CNR ID/split 元数据继承分区、未知实体沿用 G30 哈希分区，仅取 val；old test 与 train 实体不进入非公交目标选择。只读取 split 元数据，不读原测试轨迹。
- 在每个源 family 的候选池中按固定 SHA256 排序（选择 seed 33001）取前 19 个，**先取请求，再解析端点**；端点无法明确的槽位保留为失败，不拿下一条替补。
- 标签固定为 bus/motorcycle/private_car/bycicle/taxi_or_uber = 0..4。源 ptw 明确请求 motorcycle；其他 family 的实际 vType 由固定哈希选取：五种 passenger、三种 bicycle、taxi/uber。uber 的 SUMO vClass 仍为 passenger，只把参与者标签记为 taxi_or_uber，不让它因此使用 taxi 专属许可。
- 起止 lane 选源首/末 edge 的最低 index 类别许可 lane；起点 `min(20 m, lane_length/2)`，终点为末 lane 声明长度。这是新请求的事前明确条件，不是恢复旧仿真的 random/best 结果。公交站点锚点为公开 endPos，按站序选择前向 edge 实例。
- 新目标使用不重复的 `g33_ref_*` ID，明确出发时刻由固定哈希产生，范围 18000..18600 秒、0.25 秒格。源 triggered departure 被替换为这一控制条件；不能称精确复现历史 native 行程。

准备阶段先保存 `requests.json`、`targets.rou.xml` 和 SHA256，再启动仿真；源输入不改。完整请求含公共路线与实体标识，仅留服务器，不推 Git。

## 一场有限参考仿真

seed=330201，begin=14400，end=23000，step=0.25 秒；保留原生背景全部路线/类型/交通配置，追加 100 条已冻结目标车辆。仅对目标关闭 rerouting device，以对应已给定计划路线的条件运动任务；背景的 rerouting 规则不改。目标加车会改变背景交互，不能称“只输出字段变化的无扰动重放”。

只给目标配置 FCD device，全局 vehicle/person FCD probability=0，输出 x/y/type/speed/lane/edge/pos；保留 tripinfo、vehroute history、stops、summary 作为完成/异常审计。输出后的检查要求 FCD 无非目标记录。目标配额失败不重新仿真补齐，完整 100 槽位始终进入分母。新增文件总量上限 8 GiB；固定目标输出避免全背景 FCD。

公交 stop 的 `until` 在 standalone route 中是相对出发时刻的量。代码复制 standalone route 再让目标 vehicle 引用它，不把相对值错误放成车辆上的绝对时间，也不丢掉源 duration。依据：[SUMO 公交站点与独立路线说明](https://sumo.dlr.de/docs/Simulation/Public_Transport.html)。

## 提取与失败口径

按显式目标 ID 从 FCD 收集原生 0.25 秒坐标；事件由 tripinfo/vehroute/stop 输出读取，不猜测停车边界。没有出发、截止未完成、站点数量不符、少于两点、非有限/不连续或坐标跳跃超过 100 m/s 均单列。未到达的 `vaporized="end"` 是截止状态，不能称异常销毁。

实际路线与冻结计划逐项比较，保留 mismatch 与 history 数。参考轨迹可以被提取并报告实际路线不符，但正式条件生成的可比较覆盖还须另行审计；禁止以 realized route 修补模型条件。末次观测早于到达事件的时间差明确报告，不构造未知终点坐标。

公交按公共服务模板纳入，并未按线路构造训练/验证隔离；具体线路与训练队列的重合须在生成评价中披露，不能仅凭新目标 ID 声称未见线路泛化。非公交源 parent 按已有规则隔离，但路线可能重复。这是同地图、同公共需求族、受控新出发时刻和随机数的独立参考，不是预先设计的未见线路或城市泛化。单场结果不能提供仿真场次级统计置信区间。

## 执行入口

在隔离 `g33-job` 中使用 `scripts/g33_requests.py`，依次 `checks`、`prepare`、`run`、`extract`。默认源 `../g30-job/g30-data/source`，split 元数据 `../data/processed/most64/dataset.json`，SUMO `../g10-env/sumo/bin/sumo`。每个阶段拒绝覆盖既有输出，失败回执保留。正式启动必须先核验当前代码/根协议及准备清单 hash。
