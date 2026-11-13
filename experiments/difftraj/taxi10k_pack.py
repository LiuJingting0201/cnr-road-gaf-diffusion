"""Recover full native taxi paths for the exact archived episode-disjoint IDs."""
from pathlib import Path
from collections import defaultdict
import numpy as np
root=Path('/workspace/cnr-experiments/difftraj-five-10k')
assert (root/'COMPLETE').exists()
blocks=[];lens=[];ids=[];times=[];splits=[]
for part,tag,n in [('train',0,10000),('validation',8,1000),('test',9,1000)]:
    index=np.load(root/part/'index.npz')
    keys=index['ids'][index['classes']=='taxi']
    assert len(keys)==n
    episodes=defaultdict(set)
    for key in keys:episodes[int(key.split('_')[0])].add(key)
    for seed,keep in episodes.items():
        with np.load(root/str(seed)/'trips.npz') as d:
            lengths=d['h'][:,3].astype(int);ends=np.r_[0,lengths.cumsum()];raw=d['raw']
            for i,key in enumerate(d['ids']):
                if key not in keep:continue
                track=raw[ends[i]:ends[i+1]]
                assert len(track)>=120 and np.all(np.diff(track[:,0])==1)
                xy=np.rint(track[:,1:3]*100).astype('i4')
                blocks.append(np.diff(xy,axis=0,prepend=np.zeros((1,2),dtype='i4')))
                ids.append(key);lens.append(len(track));times.append(int(track[0,0]));splits.append(tag)
assert len(set(ids))==12000
np.savez_compressed('observations.npz',delta=np.concatenate(blocks),length=lens,taxi=ids,start=times,duration=np.array(lens)-1,timezone='UTC',planar_unit_m=.01,episode_split=splits)
