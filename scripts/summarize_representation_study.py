"""Write auditable validation tables; never label validation results as test results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--study',default='runs/representation-r1')
p.add_argument('--output',default='docs/results/representation-r1')
a=p.parse_args();root=Path(a.study);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
spec=json.loads((root/'study.json').read_text());rows=[];missing=[]
for task in spec['tasks']:
    name=f"{task['variant']}-seed{task['seed']}";f=root/name/'complete.json'
    if not f.exists():missing.append(name);continue
    d=json.loads(f.read_text());cfg=json.loads((root/name/'config.json').read_text())
    rows.append({'variant':d['variant'],'seed':d['seed'],'steps':d['steps'],'best_step':d['best_step'],
                 'macro_f1':d['best_validation']['macro_f1'],'accuracy':d['best_validation']['accuracy'],
                 'per_class_f1':d['best_validation']['per_class_f1'],'confusion':d['best_validation']['confusion'],
                 'elapsed_s':d['elapsed_s'],'parameters':d['parameters'],'git_commit':cfg['git_commit'],
                 'data_hashes':cfg['hashes']})
aggregate=[]
for variant in sorted(set(r['variant'] for r in rows)):
    rr=[r for r in rows if r['variant']==variant];scores=[r['macro_f1'] for r in rr]
    aggregate.append({'variant':variant,'seeds':len(rr),'macro_f1_mean':float(np.mean(scores)),
                      'macro_f1_std':float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                      'accuracy_mean':float(np.mean([r['accuracy'] for r in rr])),
                      'mean_elapsed_s':float(np.mean([r['elapsed_s'] for r in rr])),
                      'parameters':rr[0]['parameters']})
report={'scope':'validation-only exploratory representation study; checkpoints selected on this validation set',
        'complete':not missing,'missing':missing,'runs':rows,'aggregate':aggregate,
        'limitations':['not paper reproduction','no test-set conclusion','no generated-trajectory benefit established',
                       'different encoders have different parameter counts; budgets matched by updates, not FLOPs']}
(out/'results.json').write_text(json.dumps(report,indent=2))
fields=['variant','seeds','macro_f1_mean','macro_f1_std','accuracy_mean','mean_elapsed_s','parameters']
with (out/'summary.tsv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(aggregate)
lines=[f'# {root.name}：轨迹表示对照结果','',
       '所有数字来自真实运行。按验证集 macro-F1 选择 checkpoint；下表为验证集结果，不是独立测试结论。',
       f"已完成 {len(rows)}/{len(spec['tasks'])} 个任务。",'',
       '| 表示 | 种子数 | Macro-F1 均值 ± 标准差 | Accuracy | 参数量 | 平均秒数 |',
       '|---|---:|---:|---:|---:|---:|']
for r in sorted(aggregate,key=lambda x:-x['macro_f1_mean']):
    sd='—' if r['macro_f1_std'] is None else f"{100*r['macro_f1_std']:.2f}"
    lines.append(f"| {r['variant']} | {r['seeds']} | {100*r['macro_f1_mean']:.2f}% ± {sd} | {100*r['accuracy_mean']:.2f}% | {r['parameters']:,} | {r['mean_elapsed_s']:.1f} |")
lines+=['','协议和局限见 docs/EXPERIMENT_LOG.md；每次运行的混淆矩阵、各类 F1、数据 hash 和代码提交见 results.json。',
        '模型分类能力不等于对生成有帮助，后续必须使用固定候选预算与独立分类器评价。','']
(out/'REPORT.md').write_text('\n'.join(lines))
if aggregate:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered=sorted(aggregate,key=lambda r:r['macro_f1_mean'])
    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.barh([r['variant'] for r in ordered],[100*r['macro_f1_mean'] for r in ordered],
            xerr=[100*(r['macro_f1_std'] or 0) for r in ordered],color='#3478a4',capsize=4)
    for i,r in enumerate(ordered):
        values=[100*x['macro_f1'] for x in rows if x['variant']==r['variant']]
        ax.scatter(values,[i]*len(values),color='#172c3b',s=18,zorder=3)
    ax.set_xlabel('Validation macro-F1 (%)');ax.set_title(f'{root.name} (validation-selected checkpoints)')
    ax.set_xlim(0,100);ax.grid(axis='x',alpha=.2);fig.tight_layout()
    fig.savefig(out/'validation_f1.png',dpi=180);fig.savefig(out/'validation_f1.svg');plt.close(fig)
print(json.dumps({'complete':not missing,'missing':missing,'aggregate':aggregate},indent=2))
