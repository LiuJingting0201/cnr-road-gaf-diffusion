# G31：原生 MoST 完整行程接口小验证

状态：**单场无扰动、事件覆盖、静态计划路径与oracle重构检查已完成；生成接口尚未就绪，未训练生成器**。2026-11-06 已核验所有分阶段回执和`interface-complete.txt`，结果见[独立报告](results/g31-leg-interface/REPORT.md)。9,939,409条记录共同字段完全一致，1644个事件/采样合格段来自1592个parent。静态路径换道/步行适配和长时序表示仍有缺口；oracle使用真实路径，不能合并为计划地图解码成功。首次辅助非有限字段导致的比较失败保留，修复后八项构造检查及全量attempt2通过。本工作不改变G30数据/划分/模型或论文。

## 一场输出增强重放

仅重放已有 pilot seed 300001 到新的隔离目录。固定原 MoST 源文件及路网哈希、SUMO 版本、seed、`begin=14400`、`end=19516`、`step-length=.25` 和全部交通参数；不延长时间、不重新随机 OD。现有 `scripts/g30_native.py` 的命令是基准，不覆盖原 pilot 文件。

最少新增三个选项；建议一次把另两个输出选项也打开，避免再次重放：

```text
--vehroute-output.exit-times true
--stop-output.write-unfinished true
--vehroute-output.stop-edges true
--vehroute-output.route-length true
--vehroute-output.last-route false
```

保留原有 tripinfo/vehroute 的 `write-unfinished=true`。将现有 FCD 属性列表从 `x,y,type,speed,lane,edge,vehicle` 扩为 `x,y,type,speed,lane,edge,vehicle,pos`，便于重复路段和端点定位。无需新增 `personroute-output`：当前 vehroute 输出已含 person，可按 XML 标签解析。新目录保存实际命令、源文件哈希、版本、退出码和完成标记；原始 XML、ID、坐标不进 Git。

部署后从新隔离目录 `/workspace/cnr-experiments/gaf-inloop-v1/g31-job` 顺序运行（不是同时启动）：

```bash
../g29-job/.venv/bin/python ../scripts/g31_replay.py
../g29-job/.venv/bin/python ../scripts/g31_compare.py
```

默认重放目录为 `g31-data/replay/pilot/300001`，对照目录为 `g31-data/compare/pilot-300001`。任一目标目录已存在即拒绝覆盖；失败也保留。重放写 `simulation-complete.json` / `simulation-failed.json`，对照写 `comparison-complete.json` / `comparison-failed.json`，后者必须同时通过共同 FCD 精确一致、完整 20,464 个时刻及新增字段检查。只有仿真完成不代表无扰动检查通过。对照始终先保存聚合的 `fcd-comparison.json`、`field-audit.json`；它不证明完整 leg 已提取或可用于训练。构造检查覆盖同刻记录重排与新增属性、父关系/坐标变化、缺失记录/时刻、重复 ID/倒序时间、命令白名单、输出字段及完成收据/拒绝覆盖。

## 可解析字段与限制

| 输出 | 字段 | 用途与边界 |
|---|---|---|
| vehroute 的 route / walk | `edges`, `exitTimes` | 对齐已行驶的边实例；未经过的尾部 `exitTimes=-1`。重复边不能仅按 edge ID 合并。 |
| vehroute 的 stop | `started`, `ended`, `priorEdges`, `priorEdgesLength` | 停止事件与前段边序列。`priorEdges` 是实现轨迹审计信息；`priorEdgesLength` 累加整边长度，不精确扣首尾位置，不能直接当本段真实里程。 |
| vehroute 的 vehicle / route history | `arrival`, `routeLength`, `replacedAtTime`, `replacedOnEdge` | 区分完成/截尾及改道。**unfinished vehicle 的 routeLength 仍可为正**，不能用 `routeLength=-1` 统一判完成。 |
| vehroute 的 person stage | walk / ride 的 `started`, `ended` | 单个行走/乘坐阶段边界；对照 tripinfo 的 stage depart/arrival。未开始/未结束阶段排除。 |
| stop-output | `started`, `ended`, `lane`, `pos`, `parking`, `busStop` / `parkingArea` | `ended=-1` 表示停止尚未结束，但其 `started` 可结束此前完整出行段。布尔值兼容 `0/1` 与 `false/true`。 |
| FCD | `time`, `x,y`, `speed`, `lane/edge`, `pos`, `vehicle` | 保留真实时间；骑乘 person 不能再次计为 pedestrian。交通等待与活动停车分开。 |

字段语义核对了 [SUMO VehRoutes 文档](https://sumo.dlr.de/docs/Simulation/Output/VehRoutes.html)、[固定 v1_28_0 输出源码](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/devices/MSDevice_Vehroutes.cpp)、[StopOutput](https://sumo.dlr.de/docs/Simulation/Output/StopOutput.html) 和 [FCDOutput](https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html)。正式 parser 仍须对增强重放的实际 XML 做 schema 检查，不能把文档示例当已实测字段。

## 单次行程的最小定义

- 普通车辆：从实际出发/上一活动停车结束，到下一活动停车开始或最终到达；不能把含白天数小时活动停留的整个车辆 ID 当一个 OD。活动边界须来自原需求 stop/parking 的语义及事件，不能通过分类分数或观察低速阈值造边界。存在歧义则单列排除。
- 公交：一次预定线路服务为一个行程，沿线中间站停靠保留在行程内，不能逐站切分后与汽车完整 OD 混比。未完成服务不当完整参考。
- 行人：完整 walk stage 为一个步行段；ride stage 和车辆重复观测排除。一个 person 的整天多模态活动链不是一个步行轨迹。
- 每个 leg 保存 parent base ID、stage/leg 序号、实际起止事件、观测完整性及路径来源。沿用 G30 的 CNR exact-ID split 继承、test-template 排除及整个 base-ID 分组；同一个 parent 的多段不得跨分区。

生成条件只允许来自**仿真前的公开需求路线/站点计划和地图**。最终 vehroute、FCD、exitTimes 和 priorEdges 用于目标提取及审计；不能作为评价时预知的未来路径。先检查计划与实际是否一致，改道样本独立统计。以真实最终路径做的重构只能标为 oracle 接口检查。

## 本轮只做四项检查

1. **输出无扰动**：按 `(time, XML tag, id)` 对齐，新增前后共同的 `x,y,type,speed,lane/edge,vehicle` 记录应完全一致；差异则先调查，不能声称仅扩充输出。输出顺序、gzip 时间戳与 XML 头不用于等同性比较。
2. **完整性与覆盖**：逐类汇总完整 leg、活动链切分、未出发、未完成、间断、乘坐重复、改道及计划匹配失败数量；每个纳入 leg 的起止事件、FCD 时间覆盖和父实体划分必须通过断言。截尾但已到活动停车的前段与真正未到目的地的段分开。
3. **路径接口**：检查预定边序列的方向、vClass 许可、站点顺序和 lane 连接；循环路径按边实例索引对齐。先分别记录机动车与行人成功率，不用 G22 的同一人行道特例声称一般步行拓扑已支持。
4. **真实目标重构**：复用 `scripts/g22_diffusion.py::positions/decode` 和 `scripts/g28_roundtrip.py` 的审计思路，把真实完整 leg 转成时长与单调进度后解码，报告逐类位置/端点、速度、加速度、停留与巡航重构误差和长尾。若表示本身改变主要运动模式，先修接口，不训练生成器。输出原始目标与重构的固定首例，避免只看均值。

以上是数据接口可行性检查，不是 GAF 有效性实验。其结果须先能说明完整行程覆盖、计划条件来源和运动信息保留，再锁定下一轮生成对照。`g22_diffusion.py` 的若干统计/采样循环、`g26_train.py`（G27 使用 `CNR_G26_LAYOUT=flat`）和 `g28_train.py` 都仍有八类假设；原生六类不能直接接入空的货车/配送车分组。空间 GAF 最先应检验其在可选路线间是否有独立信息，固定路径中的速度反馈另作对照；本轮不据 G30 分类分数直接训练或宣称这一分工成立。

执行后的语义修复：SUMO `vaporized="end"` 表示到观察截止仍未完成。旧回执714个此类parent不应称异常销毁；本pilot中其非公交没有已完成活动前leg，所以没有据此重提取。修复代码允许未来此类parent保留已完成前leg，15项构造测试通过，只在独立构造目录验证，未冒领旧真实结果。旧/新脚本哈希和核验见`results/g31-leg-interface/semantic-clarification.json`。
