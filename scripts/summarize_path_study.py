"""Aggregate all repeated guidance settings without selecting a winner post hoc."""
import json,statistics
from pathlib import Path
root=Path('docs/results/path-g5-repeat')
reports=[json.loads((root/f'seed{s}/results.json').read_text()) for s in [42,43,44]]
variants=[r['variant'] for r in reports[0]['rows']]
keys=['graph_consistent','gaf_accuracy','tcn_accuracy','speed_w1_mps','within_condition_pairwise_distance_m','route_end_capped']
rows=[]
for v in variants:
 runs=[next(r for r in p['rows'] if r['variant']==v) for p in reports]
 rows.append({'variant':v,**{k:{'mean':statistics.mean(r[k] for r in runs),'seed_sd':statistics.stdev(r[k] for r in runs),'values':[r[k] for r in runs]} for k in keys}})
(root/'summary.json').write_text(json.dumps({'seeds':[42,43,44],'rows':rows},indent=2))
lines=['# G5：地图路径条件扩散与GAF引导','',
 '三模型各20,000步；本轮复用权重，每类16个验证起点、每起点4次噪声采样，共320条/设置/模型。八种设置完整报告。所有模型使用同一批起点、路线与初始噪声，不能把960条输出当作960个独立条件。以下为三种子均值±种子标准差，非置信区间。','',
 '| 引导 | GAF独立一致性% | TCN独立一致性% | 速度W1 m/s↓ | 同条件样本间距m |','|---|---:|---:|---:|---:|']
for r in rows:
 vals=[]
 for k in ['gaf_accuracy','tcn_accuracy','speed_w1_mps','within_condition_pairwise_distance_m']:
  scale=100 if 'accuracy' in k else 1;d=r[k];vals.append(f"{scale*d['mean']:.2f} ± {scale*d['seed_sd']:.2f}")
 lines.append('| '+r['variant']+' | '+' | '.join(vals)+' |')
lines+=['','本轮各设置均通过当前5m候选道路图一致性检查，路径末端截断率均0。约束主要来自显式路径规划与非负进度解码，不能归功于GAF。不是交通合法性证明，未评价信号灯、碰撞或多车交互。','',
 'GAF教师为位移GAF分类器，来源与原Hilbert离散编码的差异已记录。独立审计使用未作为引导教师的GAF/XY序列模型；同架构偏好仍存在。强引导的分数收益必须与速度失真、多样性变化一起解释；样本间距不是越大越好，也不能单独证明分布覆盖。','',
 '范围仅为五类移动且可显式连接匹配的车辆窗口，不覆盖行人、停留、换道；地图路线在验证生成时只用起点和训练统计，匹配真实未来路线仅用于训练监督与表示恢复诊断。选型仍属验证探索，未使用测试集。','',
 '[固定案例：每类第一个起点的全部四次采样](seed42/fixed_examples.png)。共享路线导致平面形状重叠，差别主要在同一时间的行进位置与速度，不能据此声称路线多样性已解决。','',
 '下一步：匹配同一子集的坐标扩散基线，分离约束表示收益；补类别级失败分析与真实留出车辆的分类增广；扩展停留/换道和行人。暂不继续提高GAF权重追逐单一分类分数。']
(root/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:15]))
