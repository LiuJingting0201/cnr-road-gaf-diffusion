"""Verify the existing held-out scores before exporting private plot inputs."""
from pathlib import Path
import json
import numpy as np
from porto import arc_length, resample_trajectory,grid
base=Path('/workspace/cnr-experiments/turin-batch56')
def load_case(p):
    draws=np.load(p/'draws.npz'); raw=np.load(p/'observations.npz')
    sizes=raw['length']; ends=np.r_[0,sizes.cumsum()];delta=raw['delta']
    ids=draws['ids']
    ref=[delta[ends[i]:ends[i+1]].cumsum(0)*1e-6 for i in ids]
    syn=[resample_trajectory(t,int(sizes[i])) for t,i in zip(draws['xy'],ids)]
    return ids,ref,syn
saved=json.loads((base/'42/turin-final.json').read_text());rep=json.loads((base/'figures42/geolife-final.json').read_text())
for k in ['density_js','trip_js','length_js','top20_cell_f1','request_index_sha256']: assert rep[k]==saved[k]
ids,ref,gen=load_case(base/'figures42')
ids2,_,old=load_case(Path('/workspace/cnr-experiments/turin-bus-only-v1/42'));assert np.array_equal(ids,ids2)
data=np.load(base/'42/prepared.npz');lo,hi=data['lo'],data['hi']
def counts(trips):
    return np.bincount(grid(np.concatenate(trips),lo,hi),minlength=256)
out=dict(lo=lo,hi=hi)
for label,trips in zip(['Observed','Batch1024','Batch56'],[ref,old,gen]):
    out[label+'counts']=counts(trips);out[label+'km']=[arc_length(t)/1000 for t in trips]
    for i in range(3):out[label+str(i)]=trips[i]
np.savez('turin-plot.npz',**out)
