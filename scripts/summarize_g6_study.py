import json,statistics
from pathlib import Path
out=Path('docs/results/g6-analysis');out.mkdir(parents=True,exist_ok=True)
rows=[]
for family in ['coordinate-g6','path-g5-repeat','preserved-g6']:
 reports=[json.loads((Path('docs/results')/family/f'seed{s}/results.json').read_text()) for s in [42,43,44]]
 for variant in [r['variant'] for r in reports[0]['rows']]:
  runs=[next(r for r in p['rows'] if r['variant']==variant) for p in reports]
  keys=['graph_consistent','gaf_accuracy','tcn_accuracy','speed_w1_mps','within_condition_pairwise_distance_m']
  rows.append({'family':family,'variant':variant,**{k:{'mean':statistics.mean(r[k] for r in runs),'seed_sd':statistics.stdev(r[k] for r in runs)} for k in keys}})
(out/'summary.json').write_text(json.dumps({'rows':rows},indent=2))
lines=['# G6：公平坐标基线与保持行程的GAF引导','',
 '所有模型使用相同16,779训练窗口；验证为同一批80个起点，每起点4次采样。坐标基线三模型新训各20,000步；路径模型复用G5三模型。报告三种子均值。测试集未用。','',
 '坐标基线与路径模型在生成表示、地图输入与损失上均有差异；这是系统级比较，不能解释成只加一个拓扑模块的因果消融。前者使用地图栅格，后者额外使用规划器的有向路径条件。二者不同维度噪声，仅起点/类别/采样数匹配。','',
 '| 系列 | 引导 | 图一致% | GAF审计% | TCN审计% | 速度W1↓ | 同条件样本间距m |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 vals=[r[k]['mean']*(100 if k in ['graph_consistent','gaf_accuracy','tcn_accuracy'] else 1) for k in ['graph_consistent','gaf_accuracy','tcn_accuracy','speed_w1_mps','within_condition_pairwise_distance_m']]
 lines.append('| '+r['family']+' | '+r['variant']+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
lines+=['','保持行程引导将每个低噪声干净预测的非负弧长增量投影到对应无引导样本总行程，不能使用真实未来长度。约束保持弧长及终点，不保证速度分布或加速度自动合理。其是否有价值必须由独立指标判断。','',
 '所有结果仅为可可靠匹配的五类移动非行人子集，未覆盖停留、换道和行人；5m道路图审计不是完整交通合法性。样本间距越大不必然越好，坐标模型离路也可造成大间距。','']
bootstrap=out/'vehicle_bootstrap.json'
if bootstrap.exists():
 b=json.loads(bootstrap.read_text());lines+=['## 车辆分组的不确定性检查','',b['unit'],'',b['scope'],'','| 系列 | 引导 | 独立TCN配对变化pp | 95%车辆bootstrap区间pp |','|---|---|---:|---:|']
 for r in b['rows']:
  v=r['tcn'];lo,hi=v['bootstrap_percentile_95'];lines.append(f"| {r['family']} | {r['variant']} | {100*v['vehicle_macro_paired_gain']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] |")
(out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))
