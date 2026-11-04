"""Post-hoc G3 diagnostic: does better road proximity favor short/static tracks?

Thresholds below are descriptive, not vehicle-class physical validity limits.
No selection policy or model is changed after inspecting these metrics.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from cnr.data import CLASSES

root=Path('runs/candidate-selection-g3');r=json.loads((root/'report.json').read_text())
real=np.load('data/processed/most64/val.npz');dt=.25

def features(x):
    vel=np.diff(x.astype(np.float64),axis=1)/dt
    speed=np.linalg.norm(vel,axis=-1)
    acc=np.linalg.norm(np.diff(vel,axis=1)/dt,axis=-1)
    return {'speed':speed,'acceleration':acc,
            'path_length':speed.sum(1)*dt,'displacement':np.linalg.norm(x[:,-1]-x[:,0],axis=-1),
            'nearly_static':(np.linalg.norm(x-x[:,:1],axis=-1).max(1)<1).astype(float)}

references={name:features(real['xy'][real['labels']==i]) for i,name in enumerate(CLASSES)}
rows=[]
for policy in r['policies']:
    sample=np.load(root/f'selected_{policy}.npz');perclass={}
    for i,name in enumerate(CLASSES):
        f=features(sample['xy'][sample['labels']==i]);ref=references[name]
        perclass[name]={'mean_path_length_m':float(f['path_length'].mean()),
            'mean_displacement_m':float(f['displacement'].mean()),'nearly_static_fraction':float(f['nearly_static'].mean()),
            'path_length_w1_m':float(wasserstein_distance(f['path_length'],ref['path_length'])),
            'acceleration_w1_mps2':float(wasserstein_distance(f['acceleration'].ravel(),ref['acceleration'].ravel())),
            'acceleration_p95_mps2':float(np.quantile(f['acceleration'],.95))}
    rows.append({'policy':policy,'per_class':perclass,
                 'macro':{k:float(np.mean([v[k] for v in perclass.values()])) for k in perclass[CLASSES[0]]}})
refsummary={name:{'mean_path_length_m':float(f['path_length'].mean()),
                        'mean_displacement_m':float(f['displacement'].mean()),
                        'nearly_static_fraction':float(f['nearly_static'].mean()),
                        'acceleration_p95_mps2':float(np.quantile(f['acceleration'],.95))} for name,f in references.items()}
out=Path('docs/results/candidate-selection-g3');out.mkdir(parents=True,exist_ok=True)
(out/'motion_audit.json').write_text(json.dumps({'scope':'post-hoc validation diagnostic; no policy retuning',
    'nearly_static_definition':'max distance from supplied start < 1 meter over all 64 points',
    'dt':dt,'reference':'all real validation windows within each class, no test access',
    'limitations':['unconfirmed original time unit, seconds assumed','reference and generated start distributions differ',
                   'descriptive acceleration, no physical acceptance threshold','stationarity may be legitimate'],
    'real_reference':refsummary,'policies':rows},indent=2))
lines=['# G3追加运动诊断（事后探索）','',
       '道路距离筛选可能偏好短轨迹或停留；下表在原策略固定后追加，不用于重选策略。按类别先统计再平均。',
       '近静止定义：64点始终离给定起点不足1m，不代表异常，真实等灯/停留同样可能满足。','',
       '| 策略 | 平均路程m | 平均净位移m | 近静止比例 | 路程分布W1 m | 加速度分布W1 m/s² |',
       '|---|---:|---:|---:|---:|---:|']
for row in rows:
    v=row['macro'];lines.append(f"| {row['policy']} | {v['mean_path_length_m']:.2f} | {v['mean_displacement_m']:.2f} | {100*v['nearly_static_fraction']:.2f}% | {v['path_length_w1_m']:.2f} | {v['acceleration_w1_mps2']:.2f} |")
lines+=['','对照为各类别全部真实验证窗口，不是同一路径的预测误差。时间单位暂按秒解释。',
        '这些是诊断量，不是硬性物理有效率；应结合道路连接、交通状态和完整分布评价。','']
(out/'MOTION_AUDIT.md').write_text('\n'.join(lines))
