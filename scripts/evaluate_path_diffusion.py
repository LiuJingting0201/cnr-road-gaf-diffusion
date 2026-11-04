"""Matched-cohort validation of guidance; independent classifier and lane graph audits."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from cnr.data import CLASSES
from cnr.representations import RepresentationClassifier
from cnr.topology import LaneConsistency
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
torch.set_num_threads(4);cfg=json.loads((root/'config.json').read_text());source=np.load('data/processed/most64/val.npz')
m=LaneConsistency('data/maps/most.net.xml',radius=5.);auditors={}
for kind,variant in [('gaf','delta_gaf'),('tcn','xy_motion_tcn')]:
 models=[]
 for seed in [43,44]:
  path=Path('runs/representation-r1')/f'{variant}-seed{seed}';c=json.loads((path/'config.json').read_text());net=RepresentationClassifier(c['variant'],c['stats']).eval()
  net.load_state_dict(torch.load(path/'best.pt',map_location='cpu',weights_only=True));models.append(net)
 auditors[kind]=models
rows=[]
for path in sorted(root.glob('*.npz')):
 s=np.load(path);xy=s['xy'];labels=s['labels'];real=source['xy'][s['source_index']]
 speed=np.linalg.norm(np.diff(xy,axis=1),axis=-1)/.25;rs=np.linalg.norm(np.diff(real,axis=1),axis=-1)/.25
 row={'variant':path.stem,'samples':len(xy),'per_class':{}}
 predictions={}
 with torch.no_grad():
  for kind,models in auditors.items():
   predictions[kind]=torch.stack([torch.cat([net(torch.from_numpy(xy[j:j+32])).softmax(-1) for j in range(0,len(xy),32)]) for net in models]).mean(0).argmax(-1).numpy()
 for c in np.unique(labels):
  keep=labels==c;tracks=xy[keep];graphs=[m.match(x,CLASSES[c]) for x in tracks]
  row['per_class'][CLASSES[c]]={'n':int(keep.sum()),'graph_consistent':float(np.mean([r['graph_consistent'] for r in graphs])),
    'speed_w1_mps':float(wasserstein_distance(speed[keep].ravel(),rs[keep].ravel())),
    'gaf_accuracy':float(np.mean(predictions['gaf'][keep]==c)),'tcn_accuracy':float(np.mean(predictions['tcn'][keep]==c)),
    'endpoint_displacement_le5m':float(np.mean(np.linalg.norm(tracks[:,-1]-tracks[:,0],axis=-1)<=5)),
    'route_end_capped':float(np.mean(s['progress'][keep,-1]>=s['route_length'][keep])),
    'acceleration_abs_mean_mps2':float(np.abs(np.diff(speed[keep],axis=1)/.25).mean()),
    'reference_acceleration_abs_mean_mps2':float(np.abs(np.diff(rs[keep],axis=1)/.25).mean())}
 for key in ['graph_consistent','speed_w1_mps','gaf_accuracy','tcn_accuracy','endpoint_displacement_le5m','route_end_capped']:
  row[key]=float(np.mean([v[key] for v in row['per_class'].values()]))
 rows.append(row);print(json.dumps(row),flush=True)
out=Path('docs/results')/root.parent.name/root.name;out.mkdir(parents=True,exist_ok=True)
report={'rows':rows,'config':cfg,'scope':'validation selected moving nonpedestrian cohort; five-class macro; not comparable to full six-class G4; graph audit is not traffic validity; one draw per supplied condition, within-condition diversity not yet measured'}
(out/'results.json').write_text(json.dumps(report,indent=2))
lines=['# 地图路径条件扩散与采样引导','',report['scope'],'','| 引导 | 图一致率 | GAF独立审计 | TCN独立审计 | 速度W1 | 路径末端截断 |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['variant']} | {r['graph_consistent']:.3f} | {r['gaf_accuracy']:.3f} | {r['tcn_accuracy']:.3f} | {r['speed_w1_mps']:.3f} | {r['route_end_capped']:.3f} |")
lines+=['','路径规划使用训练转移统计，扩散学习条件运动；不能把规划器保证的连接归功于GAF。引导教师未参与独立审计训练；同架构偏好仍可能存在。测试集未使用。']
(out/'REPORT.md').write_text('\n'.join(lines))
