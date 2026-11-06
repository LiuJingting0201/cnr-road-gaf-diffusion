# G32：事前车辆路线几何接口

## 状态与范围

2026-11-06 新增 `scripts/g32_routes.py`、`scripts/g32_route_checks.py`。v1 的 30 项合成小路网检查通过，全部 24 条公开公交服务 source-only 构造为 11 成功、13 未支持。根据两例具体几何失败锁定 v2 修正后，33 项检查通过；对**字节相同的全部 24 条请求**仅复核一次，21 成功、3 未支持，旧 11 条成功均保留。没有运行新 SUMO、读取目标 FCD、训练或访问冻结测试。回执保留代码 SHA256、Python 3.12.3 与 Shapely 2.1.2 版本及全部旧失败。

这是一个可以接生成器的 **五类车辆事前请求 → 地图折线** 接口。支持 SUMO vClass `passenger/bus/motorcycle/bicycle/taxi`。它尚不等于五类真实 MoST 行程全部可训练，更不等于六类生成已就绪。行人的双向步行、walkingarea 与阶段端点没有实现；只有 OD、重复边活动边界未消歧、站点无法解析的输入仍须另行规划。

## 请求必须先于仿真冻结

输入为确切 network SHA256、公开源的有序 edge **实例** 序列、vClass、起终点 lane+pos、全部计划站点的 edge 实例/lane/pos。代码拒绝 `random` 等未实例化位置、非事前 provenance、错误网络 hash、行人及缺少显式路线的请求。

原 MoST 中 `departLane="best"`、`arrivalPos="random"` 等策略不能从已完成 pilot 的 FCD 中反推为模型已知条件。下一批数据必须先将它们实例化并写入请求 manifest，再将同一组数值写入新 SUMO demand，仿真与扩散共用这些事前条件。输出明确标记 `original_pilot_endpoint_reconstruction=false`。`frozen_before_simulation` 是调用方协议声明，不是对数据来源的密码学证明；任务队列还须保存请求 hash 与仿真启动顺序。

`request_from_source_plan(...)` 检查原公开服务的全部站点数量、顺序、lane 和源区间，不能省略公交中间停站。普通活动行程检查计划停车端点。原源中数值型 depart/arrival lane 与 position 仍要匹配；负位置按 lane 声明长度解析。随机、best、current 或省略的策略在新 demand 中被明确实例化，这不是原 pilot 已实现条件的复原。

## 实现与边界

1. 动态规划沿固定的 edge 实例顺序寻找 lane 链，代价按换道次数、几何长度排序。相邻 lane 才能发生横向转换，沿途每条 lane 的 vClass 许可与驶出 lane 的 `changeLeft/changeRight` 都必须允许。
2. 连接严格通过 `.net.xml` 的 connection 和 internal lane，检查 connection 及 lane 的 allow/disallow。不能用下一条正常 edge 起点的直线替代整个路口。
3. 当前 v2 同 edge 换道按 0.25 m 空间步长寻找最早可完整落在走廊内的局部窗口，在窗口内对两条 lane 中心线作五次平滑插值，前后沿各自中心线。每次在两条 shape 上均至少前进 10 m **实际几何弧长**，不是声明 lane 长度；后一相邻换道保留足够空间。完整折线必须被许可 lane 宽度走廊覆盖。没有窗口就拒绝，不改成瞬时横跳。宽度省略时采用 SUMO 默认 3.2 m；几何数值容差 1e−6 m。连接中心线缝隙至多 2 m，桥接线仍须在走廊内。v1 的整段混合和声明距离口径按原代码归档。
4. lane 的 SUMO `length` 与 shape 几何弧长可能不同；anchor 按 `pos / declared_length` 映射。返回完整分段折线顶点、内部连接、换道数、站点累计进度、固定几何参数和精确折线 decoder。不强压到 2049 节点，避免重新切弯；进度越界明确报错，不静默截到终点。
5. 证明范围为单点轨迹的几何走廊与静态连接/类别许可。它不证明整车尺寸余量、动态车距、碰撞安全、信号通行、换道速度/加速度可行性。公交站点进入路径不代表扩散已经学会停站时刻/时长，后者须作为运动条件与独立指标核验。该路线也未必与 SUMO 参考实际选择的 lane 中心线相同，不能把两者坐标误差当模型唯一质量指标。

SUMO 规则依据：[lane 与换道权限](https://sumo.dlr.de/docs/Networks/PlainXML.html)、[车辆/路线的起终点与停站定义](https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html)。当前公开 MoST 网络 23,800 条 lane 上没有显式 `changeLeft/changeRight` 字段，采用省略字段默认许可；禁止左换/右换和空许可规则均由构造检查覆盖。这不是说所有真实 SUMO lane change 都必然执行成功。

## 接入下一轮有限生成实验

代码接口如下，真实值必须来自公开输入与事前采样器：

```python
network = VehicleRouteNetwork(pinned_net_xml)
request = request_from_source_plan(
    source_plan, vclass, frozen_start, frozen_end,
    all_frozen_stop_anchors, public_stop_locations,
    ordinal=source_activity_leg_ordinal,
    network_sha256=network.network_sha256,
)
# Save/hash all requests before starting any reference simulation.
route = network.construct(request)
polyline_xy = route.vertices
stop_progress = route.manifest()["stop_progress_m"]
xy = route.decode(generated_progress_m)
```

已执行的公共公交 proof 先把全部请求写入 immutable-by-protocol manifest，再开始构造。固定首/末 edge 最低 index 的 bus 许可 lane、起点 20 m、终点为 lane 声明长度；每站采用公开 busStop 的 endPos，按站序选择前向最早的 edge 实例。只读预检显示首边最短 32.83 m，所有站点 endPos 均不超声明 lane 长度。24 个请求均成功冻结，没有从 G31 入选行程或未来轨迹推断条件。

11 条构造成功，保留其中全部 110 个源站点；所有 24 条共 298 站，不能把 110/110 冒充全服务覆盖。成功路线换道数为 0–5。13 条失败分为 7 次停站 lane 不能到达、6 次未留下可继续到下一 edge 的 lane 状态；这些是当前构造器的失败，不等于 SUMO 认为源路线非法。几何构造总用时约 5.42 秒，不包含完整公共输入 checksum 和网络载入；未增加/改变端点或阈值来提高成功率。

按原请求和原参数只读诊断首个停站失败与首个连接失败后，发现二者都源于整段 lane blend 的走廊覆盖检查。`bus_100:Nice` 可用纵向长度 295.77 m，所需 10 m，已有相邻许可 lane，但混合线约 0.293 m 的长度落在走廊外。`bus_1:SaintRoman` 有 58.19 m 可换，目标 lane 存在许可 internal 连接；约 0.0217 m 的混合线在走廊外，使该目标 lane 被丢弃，最终表面上报“无连接”。这是几何构造不足的证据，不能作为车型空间结构缺失或地图路线非法的结论；也不能把这些长度误称为离路距离。

在 v2 执行前，修正写入并提交 `docs/G32_PROTOCOL.md`（e0592b7）。固定改用局部换道窗口，并明确最小 10 m 为两条 shape 的真实纵向几何弧长；端点、站点、许可、连接、0.25 m 步长和 1e−6 m 走廊容差保持不变。新检查包括：声明长度 200 m/shape 长度 100 m 时拒绝只前进 5 m 的换道；实际 10 m 窗口通过；具有中段分离弯道的同一网络中，原全段 blend 越界而前段局部换道保持在走廊内。最初恒定 offset 的 S 曲线测试样例没有复现旧失败，构造断言中止，尚未执行 v2 真实请求；改为明确具有中段分离的构造样例，未改变算法参数，失败记录另存。

| 固定 24 条公共请求 | v1 整段混合 | v2 局部窗口 |
|---|---:|---:|
| 静态几何成功/总数 | 11/24 | 21/24 |
| 成功路线中保留的源站点/全 24 条站点 | 110/298 | 235/298 |
| 成功路线换道数范围 | 0–5 | 0–6 |
| 仅构造时间，不含读网/checksum | 5.42 s | 38.92 s |

同一请求 manifest SHA256：`f38fee3fc0190799a340c29ea81d54ea5c1d82dfbf125aa0b32fddddff7a570e`。v2 仍有 `bus_100:Nice` 在 edge 实例 47 缺少可继续的连接状态，`bus_3:Otto` 不能到达一个站点 lane，`bus_4:Fontvieille` 在实例 51 无可继续连接状态。本轮不再调整参数或剔除这三条。21/24 是当前显式公共公交请求的静态可构造性，不是五类训练数据覆盖率、SUMO 动态成功率或生成质量；11/24 的旧结果也不覆盖。

G33 尚须：冻结一批新需求（与训练/开发/留出 parent 隔离和存储预算一起写协议），将明确端点实例写入 SUMO；检查真实参考能否在**同一事前路线**上单调投影、站点区间是否兼容、是否实际 reroute、偏移/重构尾部有多大。不能用实际未来 lane/edge 补齐失败计划。公交实际停站点可因站内占用而变化，源 stop 区间和构造锚点的差别须报告。完成这一共同条件—真实目标接口后才训练变长 motion diffusion；33 项构造检查和 21 条公开静态路线均不单独赋予 generation-ready 状态。

固定唯一 planned route 的实验主要检验运动合成与 speed-GAF 的运动反馈，不能据此充分检验 spatial-GAF 对路线选择的价值。空间类别反馈需要另列已许可的事前路线候选，控制同候选预算，评价道路与空间模式；不能以同一路线上几厘米的扰动强行制造空间引导优势。

## 服务器复现位置

- 当前 v2 代码：`/workspace/cnr-experiments/gaf-inloop-v1/g32-job/scripts/g32_routes.py` 与 `g32_route_checks.py`。原 v1 逐字节存于 `g32-data/public-bus-map-v1/source/`。
- 固定请求和原 proof：`/workspace/cnr-experiments/gaf-inloop-v1/g32-job/g32-data/public-bus-map-v1/frozen-public-requests.json`、`public-bus-proof.json`。
- 同目录：`constructor-checks.json`、`first-failure-diagnosis.json`、`diagnosis-invocation-failure.json`。诊断首个调用错误使用不存在的 `g32-job/.venv`，在 Python 启动前退出 127，无写入；之后使用已有 `g31-job/.venv/bin/python`，仅在 g32 新目录写入。旧工作树未改动。
- v2 单次结果：`g32-data/public-bus-map-v2/public-bus-proof-v2.json`、`constructor-checks-v2.json`、`implementation-receipt.json`。从当前源码逆向应用本目录的 `v1-to-v2.patch`，已实测逐字节恢复两个旧文件；Git 也可恢复旧实现，不依赖临时目录。
- Git 只收代码、文字、无原始坐标的小型聚合回执。完整请求 manifest 与路径顶点留服务器，不作为论文结果图或数据泄露到仓库。
