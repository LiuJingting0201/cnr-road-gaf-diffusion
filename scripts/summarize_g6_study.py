import json,statistics
from pathlib import Path
out=Path('docs/results/g6-analysis');out.mkdir(parents=True,exist_ok=True)
rows=[]
families=['coordinate-g6','path-g5-repeat','preserved-g6']
if Path('docs/results/route-coordinate-g6/seed44/results.json').exists():families.insert(1,'route-coordinate-g6')
for family in families:
 reports=[json.loads((Path('docs/results')/family/f'seed{s}/results.json').read_text()) for s in [42,43,44]]
 for variant in [r['variant'] for r in reports[0]['rows']]:
  runs=[next(r for r in p['rows'] if r['variant']==variant) for p in reports]
  keys=['graph_consistent','gaf_accuracy','tcn_accuracy','speed_w1_mps','within_condition_pairwise_distance_m']
  rows.append({'family':family,'variant':variant,**{k:{'mean':statistics.mean(r[k] for r in runs),'seed_sd':statistics.stdev(r[k] for r in runs)} for k in keys}})
(out/'summary.json').write_text(json.dumps({'rows':rows},indent=2))
lines=['# G6：公平坐标基线与保持行程的GAF引导','',
 '所有模型使用相同16,779训练窗口；验证为同一批80个起点，每起点4次采样。两组坐标基线共六模型新训各20,000步；路径模型复用G5三模型。报告三种子均值。测试集未用。','',
 'coordinate-g6为栅格地图坐标扩散；route-coordinate-g6为同一路线条件的坐标扩散（追加对照）；path-g5-repeat为沿路线进度扩散。追加对照与进度模型共享路线编码器和噪声MSE目标、无运动辅助损失，区别主要为自由XY与非负进度表示。不同输出维度的噪声不能逐元素配对。栅格基线另有运动辅助损失，仅作为系统比较。','',
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
lines.append("\n\n## \u672c\u8f6e\u5224\u65ad\n\n\u540c\u4e00\u8def\u7ebf\u6761\u4ef6\u7684\u81ea\u7531XY\u5bf9\u7167\u4ecd\u6709\u5927\u91cf\u6574\u8f68\u8ff9\u56fe\u4e0d\u4e00\u81f4\uff0c\u652f\u6301\u6cbf\u8def\u8fdb\u5ea6\u53c2\u6570\u5316\u7684\u7ea6\u675f\u4ef7\u503c\uff1b\u4e0d\u6d89\u53ca\u4ea4\u901a\u4ea4\u4e92\u5b89\u5168\u3002\u5f31GAF\u6709\u8de8\u67b6\u6784\u7684\u5c0f\u5e45\u6b63\u5411\u8bc1\u636e\uff0c\u4f46\u5c1a\u672a\u8bc1\u660e\u72ec\u7279\u4f18\u52bf\u3002\u4fdd\u6301\u884c\u7a0b\u65b9\u6848\u964d\u4f4e\u5f3a\u5f15\u5bfc\u7684\u901f\u5ea6\u5931\u771f\uff0c\u4e5f\u524a\u5f31\u7c7b\u522b\u6536\u76ca\uff0c\u6682\u4f5c\u4e3a\u673a\u5236\u6d88\u878d\uff1b\u9ed8\u8ba4\u4fdd\u7559\u666e\u901a\u5f31GAF0.3\u3002\n\n\u8f66\u8f86bootstrap\u8986\u76d669\u4e2a\u4e0d\u540c\u9a8c\u8bc1\u8f66\u8f86\uff08bus13\u3001motorcycle16\u3001private_car16\u3001bycicle16\u3001taxi8\uff09\uff0c\u4e0d\u662f960\u4e2a\u72ec\u7acb\u6837\u672c\u3002\u63a2\u7d22\u6027\u533a\u95f4\u672a\u6821\u6b63\u591a\u91cd\u6bd4\u8f83\uff0c\u4e5f\u672a\u8986\u76d6\u6559\u5e08\u968f\u673a\u6027\uff0c\u4e0d\u80fd\u5f53\u4f5c\u6700\u7ec8\u6d4b\u8bd5\u7ed3\u8bba\u3002\n\n[\u5f15\u5bfc\u6536\u76ca\u4e0e\u8fd0\u52a8\u8bef\u5dee\u56fe](guidance_tradeoff.png)\u3002G7\u5206\u7c7b\u589e\u5e7f\u5c1a\u672a\u6267\u884c\u3002\n")
(out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))
