"""Independent vector-road, kinematic and classifier audit for the G2 ablation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from cnr.data import CLASSES
from cnr.representations import RepresentationClassifier
from cnr.roads import RoadNetwork

torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--name',default='road-g2');a=p.parse_args()
root=Path('runs')/a.name;out=Path('docs/results')/a.name;out.mkdir(parents=True,exist_ok=True)
roads=RoadNetwork('data/maps/most.net.xml');real=np.load('data/processed/most64/val.npz')
auditors=[]
for seed in [43,44]:
    path=Path('runs/representation-r1')/f'xy_motion_tcn-seed{seed}'
    cfg=json.loads((path/'config.json').read_text());m=RepresentationClassifier(cfg['variant'],cfg['stats'])
    m.load_state_dict(torch.load(path/'best.pt',map_location='cpu',weights_only=True));m.eval();auditors.append(m)
rows=[]
for variant in ['relative','map','map_roadloss']:
    path=root/variant
    if not (path/'complete.json').exists():raise RuntimeError(f'{variant} not complete')
    sample=np.load(path/'validation_samples.npz');xy=sample['xy'];labels=sample['labels']
    with torch.no_grad():
        probabilities=torch.stack([torch.cat([m(torch.from_numpy(xy[i:i+128])).softmax(-1)
                                             for i in range(0,len(xy),128)]) for m in auditors]).mean(0)
    pred=probabilities.argmax(-1).numpy();per_class={}
    for i,name in enumerate(CLASSES):
        tracks=xy[labels==i];d=roads.distances(tracks[:,1:],name)
        speed=np.linalg.norm(np.diff(tracks,axis=1),axis=-1)/.25
        refspeed=np.linalg.norm(np.diff(real['xy'][real['labels']==i],axis=1),axis=-1)/.25
        per_class[name]={'road_within_5m':float(np.mean(d<=5)),'road_distance_mean_m':float(d.mean()),
                         'road_distance_p95_m':float(np.quantile(d,.95)),
                         'speed_wasserstein_mps':float(wasserstein_distance(speed.ravel(),refspeed.ravel())),
                         'speed_p95_mps':float(np.quantile(speed,.95)),
                         'audit_target_accuracy':float(np.mean(pred[labels==i]==i))}
    cfg=json.loads((path/'config.json').read_text())
    row={'variant':variant,'samples':len(xy),'per_class':per_class,'seed':cfg['seed'],
         'scope':'single generator seed; held-out-vehicle validation starts; generated points only (given start excluded)',
         'generation':json.loads((path/'complete.json').read_text()),
         'git_commit':json.loads((path/'config.json').read_text())['git_commit']}
    for key in ['road_within_5m','road_distance_mean_m','speed_wasserstein_mps','audit_target_accuracy']:
        row[key]=float(np.mean([v[key] for v in per_class.values()]))
    rows.append(row)
report={'rows':rows,'scope':'validation-only, single-seed pilot; geometric proximity is not topology validity',
        'auditor':'raw XY/motion TCN seeds43/44; distinct from generators; no semantic guidance used in G2'}
(out/'results.json').write_text(json.dumps(report,indent=2))
lines=['# G2：道路条件扩散预实验','',
       f"三组共享相对坐标、EMA和运动约束，只改变地图条件及道路距离损失。seed={rows[0]['seed']}；训练步数见原始日志。单次种子报告不是最终结论。",
       '使用独立向量车道精确距离；道路指标排除给定起点。语义审计使用独立训练的序列分类器，不参与生成。','',
       '| 变体 | 许可车道5m内 | 平均道路距离m | 每类速度W1均值m/s | 审计目标类别准确率 |',
       '|---|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['variant']} | {100*row['road_within_5m']:.2f}% | {row['road_distance_mean_m']:.2f} | {row['speed_wasserstein_mps']:.2f} | {100*row['audit_target_accuracy']:.2f}% |")
lines+=['','尚需：方向/拓扑验证、多种子、GAF反馈对照、多样性/记忆检查、空间留出和增广实验。','']
(out/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2))
