"""Development-only coverage by number of independent SUMO training taxi requests."""
from pathlib import Path
import json
import sys
import numpy as np
from taxi_preprocessing_forensics import cells

root=Path(sys.argv[1])
original=np.load(root/'taxi10k-base'/'prepared.npz')
low,high=original['lo'],original['hi']
validation=original['h'][original['split']==8,6:].astype(int)
query=validation[:,0]*256+validation[:,1]
old=original['h'][original['split']==0,6:].astype(int)
oldpairs=old[:,0]*256+old[:,1]
newroot=root/'difftraj-five-20k'
index=np.load(newroot/'train'/'index.npz')
norm=np.load(newroot/'normalization.npz')
rows=np.flatnonzero(index['classes']=='taxi')
x=np.load(newroot/'train'/'traj.npy',mmap_mode='r')
native=x[rows][:,[0,-1]].astype(float)*norm['std']+norm['mean']
both=cells(native,low,high)
allpairs=both[:,0]*256+both[:,1]
old_ids=np.load(root/'taxi10k-base'/'observations.npz')['taxi'][original['split']==0]
aligned={k:v for k,v in zip(index['ids'][rows],allpairs)}
changed=sum(aligned[k]!=pair for k,pair in zip(old_ids,oldpairs))
def metrics(pairs):
    counts=np.bincount(pairs,minlength=65536)
    return {str(k):float((counts[query]>=k).mean()) for k in [1,2,5,10,20]}
out=dict(n_development=len(query),grid='original 10k taxi training min/max',
         old_to_new_reconstructed_od_changes=int(changed),
         original_10000=metrics(oldpairs),expanded_20000=metrics(allpairs))
print(json.dumps(out,indent=2))
