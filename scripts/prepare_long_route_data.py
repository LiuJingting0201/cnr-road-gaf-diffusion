"""G5 pilot: training-only route prior; sampled validation routes contain no future.

Freeze this moving-vehicle subset before training. Keep rejection manifests and
raw references; never write or evaluate a test split.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from cnr.data import CLASSES
from cnr.route_paths import RoutePaths,decode_progress_numpy

p=argparse.ArgumentParser();p.add_argument('--output',default='data/processed/route512');p.add_argument('--limit',type=int,default=0);p.add_argument('--source',default='data/processed/most64');p.add_argument('--route-budget',type=float,default=512.);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
source=Path(a.source);matcher=RoutePaths('data/maps/most.net.xml',radius=2.)
counts=Counter();accepted={};rejections=[]
for split in ['train','val']:
    raw=np.load(source/f'{split}.npz');rows=[]
    for i,(xy,label) in enumerate(zip(raw['xy'],raw['labels'])):
        if a.limit and i>=a.limit:break
        reason=None;name=CLASSES[int(label)]
        if name=='pedestrian':reason='pedestrian_not_in_vehicle_pilot'
        elif np.linalg.norm(xy[-1]-xy[0])<=5:reason='endpoint_displacement_le5m'
        if reason is None:
            trace=matcher.trace(xy,name)
            if trace is None:reason='ambiguous_or_incompatible_trace'
            else:
                rec=matcher.reconstruct(trace,name)
                if rec is None or rec['progress'][-1]>=a.route_budget-12:reason='reconstruction_or_horizon'
        if reason:rejections.append({'split':split,'source_index':i,'reason':reason});continue
        rows.append({'source_index':i,'label':int(label),'xy':xy,'record':rec})
        if split=='train':counts.update((name,int(u),int(v)) for u,v in rec['transitions'])
        if i%2000==0:print(json.dumps({'split':split,'source_index':i,'accepted':len(rows)}),flush=True)
    accepted[split]=rows
prior=[{'class':c,'from':matcher.road.lanes[u]['id'],'to':matcher.road.lanes[v]['id'],'count':n}
       for (c,u,v),n in sorted(counts.items())]
(out/'route_prior.json').write_text(json.dumps(prior))
summary={};rng=np.random.default_rng(2026)
for split,rows in accepted.items():
    arrays={k:[] for k in ['xy','projected_xy','route','oracle_route','route_length','oracle_length','progress','labels','source_index']}
    for row in rows:
        rec=row['record'];name=CLASSES[row['label']];start=row['xy'][0]
        oracle=matcher.extend(rec['points'],rec['last_lane'],rec['last_position'],name,counts,rng,budget=a.route_budget)
        planned=oracle if split=='train' else matcher.sample(start,name,counts,rng,budget=a.route_budget)
        if oracle is None or planned is None:
            rejections.append({'split':split,'source_index':row['source_index'],'reason':'route_extension_failed'});continue
        oroute,olen=oracle;route,length=planned
        projected=decode_progress_numpy(oroute,olen,rec['progress'],start)
        values={'xy':row['xy'],'projected_xy':projected.astype(np.float32),'route':route,'oracle_route':oroute,
            'route_length':length,'oracle_length':olen,'progress':rec['progress'].astype(np.float32),
            'labels':row['label'],'source_index':row['source_index']}
        for k,v in values.items():arrays[k].append(v)
    arrays={k:np.asarray(v,dtype=np.int64 if k in ['labels','source_index'] else np.float32) for k,v in arrays.items()}
    np.savez_compressed(out/f'{split}.npz',**arrays)
    summary[split]={'windows':len(arrays['xy']),'class_counts':{CLASSES[i]:int(np.sum(arrays['labels']==i)) for i in range(6)},
        'projection_mean_error_m':float(np.linalg.norm(arrays['xy']-arrays['projected_xy'],axis=-1).mean())}
meta=json.loads((source/'dataset.json').read_text())
train_xy=np.load(source/'train.npz')['xy']
report={'source_limit':a.limit,'source_directory':str(source),'source_protocol':meta,'scope':'moving non-pedestrian explicit-connection route pilot; not interchangeable with the six-class G4 population',
    'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'summary':summary,
    'route_budget_m':a.route_budget,'route_points':1025,'prior_training_only':True,
    'validation_route':'nearest permitted start lane then training-count Markov transitions with 0.1 smoothing; no true future',
    'oracle_route':'retained separately for reconstruction diagnostics only; default generation must use route',
    'start_offset':'preserve original start via offset decaying over first 10m',
    'mean':train_xy.mean(axis=(0,1)).tolist(),'std':train_xy.std(axis=(0,1)).tolist(),'length':int(train_xy.shape[1]),'prior_sha256':hashlib.sha256((out/'route_prior.json').read_bytes()).hexdigest(),
    'source_hashes':{s:hashlib.sha256((source/f'{s}.npz').read_bytes()).hexdigest() for s in ['train','val']},
    'elapsed_s':time.monotonic()-started}
(out/'dataset.json').write_text(json.dumps(report,indent=2));(out/'rejections.json').write_text(json.dumps(rejections,indent=2))
print(json.dumps(report,indent=2),flush=True)
