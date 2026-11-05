import numpy as np
import json
from pathlib import Path
from scipy.stats import wasserstein_distance
from g8_common import assets,Scores,reference
from cnr.route_paths import decode_progress_numpy

cfg,meta,data=assets();sc=Scores(meta);real=reference(cfg,data)
times=np.linspace(0,1,512)
rows=[]
for i,(p,label) in enumerate(zip(data['progress'],data['labels'])):
    warp=[np.interp(times+a*np.sin(2*np.pi*times),times,p) for a in [-.12,0,.12]]
    xy=np.array([decode_progress_numpy(data['route'][i],data['route_length'][i],q,data['xy'][i,0]) for q in warp])
    assert np.max(abs(xy[:,[0,-1]]-xy[1,[0,-1]]))<1e-4
    assert np.max(abs(xy[1]-data['xy'][i]))<.01
    prob,ok,img=sc.image(xy); raw=sc.raw(xy)
    speed=np.linalg.norm(np.diff(xy,axis=1),axis=-1)/.25
    v=np.linalg.norm(np.diff(real[data['labels']==label],axis=1),axis=-1).ravel()/.25
    acc=np.diff(np.linalg.norm(np.diff(real[data['labels']==label],axis=1),axis=-1)/.25,axis=1).ravel()/.25
    rows.append(dict(i=i,label=int(label),valid=ok.tolist(),unique_images=len(set(x.tobytes() for x in img)),teacher=prob[:,label].tolist(),tcn=raw[:,label].tolist(),speed=[wasserstein_distance(s,v) for s in speed],acc=[wasserstein_distance(np.diff(s)/.25,acc) for s in speed]))
Path('pilot-output/probe.json').write_text(json.dumps(rows))
print('complete',len(rows))
