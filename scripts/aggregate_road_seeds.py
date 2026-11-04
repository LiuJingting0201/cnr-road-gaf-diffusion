"""Aggregate completed generator seeds; refuse partial or mismatched studies."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

names=['road-g2','road-g2-seed43','road-g2-seed44']
reports=[json.loads((Path('docs/results')/name/'results.json').read_text()) for name in names]
metrics=['road_within_5m','road_distance_mean_m','speed_wasserstein_mps','audit_target_accuracy']
aggregate=[]
for variant in ['relative','map','map_roadloss']:
    rows=[next(x for x in report['rows'] if x['variant']==variant) for report in reports]
    seeds=[row.get('seed',42) for row in rows]
    if seeds!=[42,43,44] or any(row['samples']!=192 for row in rows):raise ValueError('Mismatched study')
    aggregate.append({'variant':variant,'seeds':seeds,'runs':rows,
        'metrics':{m:{'mean':float(np.mean([r[m] for r in rows])),
                      'std':float(np.std([r[m] for r in rows],ddof=1))} for m in metrics}})
out=Path('docs/results/road-g2-multiseed');out.mkdir(parents=True,exist_ok=True)
pairs={}
for first,second in [(1,0),(2,1)]:
    key=aggregate[first]['variant']+' minus '+aggregate[second]['variant']
    pairs[key]={m:[float(a[m]-b[m]) for a,b in zip(aggregate[first]['runs'],aggregate[second]['runs'])] for m in metrics}
report={'scope':'validation-only three training seeds; fixed validation starts and sampling seed, no topology guarantee',
        'aggregate':aggregate,'paired_seed_differences':pairs}
(out/'results.json').write_text(json.dumps(report,indent=2))
lines=['# G2：三种子道路条件扩散对照','',
       '每个变体seed42/43/44，各20,000步；固定末步EMA、同一验证起点、DDIM100及采样种子2026，每模型192条样本。',
       '道路指标排除给定起点，使用独立向量车道精确距离。均值±训练种子标准差；不是置信区间。','',
       '| 模型 | 贴路率 % | 平均道路距离 m | 速度W1 m/s | 独立类别一致性 % |','|---|---:|---:|---:|---:|']
for row in aggregate:
    cells=[]
    for m in metrics:
        scale=100 if m in ['road_within_5m','audit_target_accuracy'] else 1
        v=row['metrics'][m];cells.append(f"{scale*v['mean']:.2f} ± {scale*v['std']:.2f}")
    lines.append('| '+row['variant']+' | '+' | '.join(cells)+' |')
lines+=['','三个训练种子仍不足以证明普适性。相同验证起点意味着只覆盖训练随机性，未覆盖空间划分和采样条件的不确定性。',
        '贴路不等于拓扑合法；G2没有使用GAF，不能将其收益归因于GAF。逐种子配对差值与完整类别统计见results.json。','']
(out/'REPORT.md').write_text('\n'.join(lines))
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,m,title in zip(axes,['road_within_5m','speed_wasserstein_mps'],['Permitted-lane proximity (%)','Speed distribution W1 (m/s)']):
    scale=100 if m=='road_within_5m' else 1
    for i,row in enumerate(aggregate):
        v=row['metrics'][m];ax.bar(i,scale*v['mean'],yerr=scale*v['std'],capsize=5,color='#3478a4',alpha=.7)
        ax.scatter([i]*3,[scale*r[m] for r in row['runs']],color='#172c3b',zorder=3)
    ax.set_xticks(range(3),[r['variant'] for r in aggregate]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.suptitle('G2: three training seeds, held-out-vehicle validation');fig.tight_layout()
fig.savefig(out/'seed_comparison.png',dpi=180);fig.savefig(out/'seed_comparison.svg');plt.close(fig)
