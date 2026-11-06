# G34固定时长机制诊断

复用已查看的G33开发参考；不是新盲测。两种子共10项评价，所有W1越低越好。

| 方法/种子 | 请求 | 配对参考 | 原生速度W1 m/s | 原生加速度W1 m/s² | 时长W1 s |
|---|---:|---:|---:|---:|---:|
| native_0_25s-42 | 100 | 94 | 0.6964 | 0.1449 | 57.7782 |
| native_0_25s-42/Uniform | 100 | 94 | 0.7329 | 0.1245 | 57.7782 |
| native_0_25s-42/S | 100 | 94 | 0.8121 | 0.1372 | 57.7782 |
| native_0_25s-42/V | 100 | 94 | 0.7450 | 0.1124 | 57.7782 |
| native_0_25s-42/SV | 100 | 94 | 0.7969 | 0.1430 | 57.7782 |
| native_0_25s-43 | 100 | 94 | 0.5504 | 0.1537 | 51.5908 |
| native_0_25s-43/Uniform | 100 | 94 | 0.5828 | 0.1597 | 51.5908 |
| native_0_25s-43/S | 100 | 94 | 0.5416 | 0.1650 | 51.5908 |
| native_0_25s-43/V | 100 | 94 | 0.5259 | 0.1322 | 51.5908 |
| native_0_25s-43/SV | 100 | 94 | 0.5879 | 0.1412 | 51.5908 |

逐点核验：所有地图支持请求的旧/新Uniform时间与坐标逐元素完全相等；所有K8候选和四种选择共享该训练先验时长。

十项全部指标、逐类支持数及对旧同方法/新Uniform差值见macro.csv、per-class.csv和summary.json。

- Mechanism diagnostic chosen after inspecting G33; the same development reference is reused, not a new blind test.
- No CNN/diffusion retraining, prior fitting or extra strength/step tuning; K8/DDIM100 remains fixed.
- Per-request shared T is the old predetermined Uniform training-prior draw, never the observed reference duration.
- All four selectors have equal duration distributions by construction; this is a control, not a duration-quality improvement.
- Old/new Uniform is checked bitwise on all map-ready requests; other candidates retain noise keys but may have different sequence lengths.
- Equal candidate/DDIM budgets do not imply equal tensor-element compute or wall time.
- Each seed is descriptive on one SUMO episode and known route templates; no confidence intervals or broad generalization claim.
- Teacher target confidence is auxiliary; physical W1 and coverage determine reported quality.
- Equal metric support counts are not a significance test; unsupported metrics remain NA.
