"""Report all registered semantic interventions; no best-only selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('docs/results')
studies=['semantic-g4','semantic-g4-seed43','semantic-g4-seed44']
reports=[json.loads((root/n/'results.json').read_text()) for n in studies]
metrics=['road_within_5m','speed_wasserstein_mps','audit_target_accuracy','gaf_audit_target_accuracy']
aggregate=[]
for variant in ['none','gaf','tcn','shuffled_gaf']:
    rows=[next(x for x in r['rows'] if x['variant']==variant) for r in reports]
    assert [x['seed'] for x in rows]==[42,43,44]
    aggregate.append({'variant':variant,'runs':rows,'metrics':{m:{'mean':float(np.mean([x[m] for x in rows])),
        'std':float(np.std([x[m] for x in rows],ddof=1))} for m in metrics}})
baseline=aggregate[0]['runs'];differences={}
for row in aggregate[1:]:
    differences[row['variant']]={m:[float(x[m]-b[m]) for x,b in zip(row['runs'],baseline)] for m in metrics}
kl=json.loads((root/'semantic-g4b-kl/results.json').read_text())
out=root/'semantic-followup';out.mkdir(parents=True,exist_ok=True)
report={'scope':'validation-selected teachers; validation-only generated samples; no test evaluation',
    'ce_three_seed':aggregate,'paired_ce_minus_none':differences,'kl_seed42':kl,
    'kl_reference_seed42':reports[0],'limits':['only 192 fixed-start samples per model',
        'three generator seeds do not cover teacher uncertainty','auditors may prefer their own architecture',
        'CE and KL equal coefficients do not imply equal gradient strength','no graph validity guarantee']}
(out/'results.json').write_text(json.dumps(report,indent=2))
lines=['# 扩散语义反馈：三种子干预与软一致性对照','',
       '全部为实际训练结果。CE组每变体seed42/43/44、每次20,000步；KL组seed42、每次20,000步。固定末步EMA和相同验证起点，未评价测试集。','',
       '## CE反馈三种子（均值±种子标准差）','',
       '| 反馈 | 贴路率 % | 速度W1 m/s | 独立TCN一致性 % | 独立GAF一致性 % |','|---|---:|---:|---:|---:|']
for row in aggregate:
    cells=[]
    for m in metrics:
        scale=1 if m=='speed_wasserstein_mps' else 100;v=row['metrics'][m]
        cells.append(f"{scale*v['mean']:.2f} ± {scale*v['std']:.2f}")
    lines.append('| '+row['variant']+' | '+' | '.join(cells)+' |')
lines+=['','## 软语义一致性（单种子，不与上表混作三种子结论）','',
        '| 变体 | 贴路率 % | 速度W1 m/s | TCN一致性 % | GAF一致性 % |','|---|---:|---:|---:|---:|']
for row in [reports[0]['rows'][0]]+kl['rows']:
    lines.append(f"| {row['variant']} | {100*row['road_within_5m']:.2f} | {row['speed_wasserstein_mps']:.2f} | {100*row['audit_target_accuracy']:.2f} | {100*row['gaf_audit_target_accuracy']:.2f} |")
lines+=['','两种审计器均未参与训练反馈；仍需警惕架构偏好。GAF自身审计上升不能单独证明通用收益，几何/运动退化亦须报告。',
        '标签打乱是机制负对照。KL以真实训练轨迹的教师分布为目标，不代表真实类别一定正确；其假设是保持真实样本语义而非强推不完美教师的类别边界。',
        '所有权重配置完整保留；不能因一项指标最优就事后宣布总体胜出。下一阶段仍需道路约束主体、多样性/记忆检查和真实数据增广验证。','']
(out/'REPORT.md').write_text('\n'.join(lines))
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,m,title in zip(axes,['road_within_5m','audit_target_accuracy','gaf_audit_target_accuracy'],['Lane proximity','Independent TCN auditor','Independent GAF auditor']):
    for i,row in enumerate(aggregate):
        v=row['metrics'][m];ax.bar(i,100*v['mean'],yerr=100*v['std'],capsize=4,color='#b76a32' if row['variant']=='gaf' else '#397da0')
        ax.scatter([i]*3,[100*r[m] for r in row['runs']],s=18,color='#202b35',zorder=3)
    ax.set_xticks(range(4),['None','GAF','TCN','Shuffled\nGAF']);ax.set_ylim(0,100);ax.set_title(title);ax.set_ylabel('%');ax.grid(axis='y',alpha=.2)
fig.suptitle('Diffusion training interventions: three generator seeds');fig.tight_layout()
fig.savefig(out/'ce_three_seed.png',dpi=180);fig.savefig(out/'ce_three_seed.svg');plt.close(fig)
print('\n'.join(lines))
