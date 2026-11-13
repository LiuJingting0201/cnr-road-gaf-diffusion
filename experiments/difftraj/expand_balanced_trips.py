"""Grow training to 20k/class while keeping old training IDs and frozen evaluation IDs."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json,hashlib
import numpy as np
from make_balanced_trips import episode,modes

old=Path('/workspace/cnr-experiments/difftraj-five-10k')
def main():
    groups={}; known=set(); previous=set()
    def insert(store,x,h,k,m):
        digest=hashlib.sha256(x.tobytes()).digest()
        if digest in known:return False
        known.add(digest);store.append((x.copy(),h.copy(),str(k),str(m)));return True
    def read_seed(s,func):
        with np.load(s/'trips.npz') as z:
            for x,h,k,m in zip(z['xy'],z['h'],z['ids'],z['classes']):func(x,h,k,m)
    for part in ['test','validation','train']:
        idx=np.load(old/part/'index.npz'); ids=idx['ids']; want=set(ids);previous.update(want); store=[]
        for s in sorted({k.split('_')[0] for k in ids}):
            read_seed(old/s,lambda x,h,k,m: insert(store,x,h,k,m) if k in want else None)
        assert len(store)==len(ids)
        ordered={r[2]:r for r in store};groups[part]=[ordered[k] for k in ids]
    train=groups['train']; count=Counter(r[3] for r in train)
    def add(x,h,k,m):
        if k not in previous and count[m]<20000 and insert(train,x,h,k,m):count[m]+=1
    for s in sorted(old.glob('41[0-9][0-9][0-9][0-9]')):read_seed(s,add)
    print('After original training surplus',count,flush=True)
    for seed in range(440000,440800,8):
        if min(count.values())>=20000:break
        # Keep all five modes in traffic, oversampling the remaining bicycle deficit.
        with ProcessPoolExecutor(8) as pool:
            list(pool.map(partial(episode,per_class={m:200 if m=='bicycle' else 30 for m in modes}),range(seed,seed+8)))
        for s in range(seed,seed+8):read_seed(Path(str(s)),add)
        print('balanced training inventory',count,flush=True)
    assert set(count.values())=={20000}
    for part in ['train','validation','test']:
        xs,hs,ids,labels=zip(*groups[part]);x=np.asarray(xs);h=np.asarray(hs).copy()
        # Historical shards may encode the episode seed in the departure field.
        # Recover the first observed timestamp from each selected native trip.
        by_episode=defaultdict(dict)
        for i,key in enumerate(ids):by_episode[key.split('_')[0]][key]=i
        for ep,want in by_episode.items():
            pack=old/ep/'trips.npz'
            if not pack.exists():pack=Path(ep)/'trips.npz'
            with np.load(pack) as z:
                starts=np.r_[0,z['h'][:-1,3].astype(int).cumsum()]
                time=z['raw'][starts,0]
                for key,t in zip(z['ids'],time):
                    if key in want:h[want[key],0]=int(t//300)%288
        if part=='train':
            mu=x.mean((0,1),dtype='d');sd=x.std((0,1),dtype='d')
            lo=x.min((0,1));hi=x.max((0,1));hm=h[:,1:].mean(0);hd=h[:,1:].std(0).clip(1e-7)
            np.savez('normalization.npz',mean=mu,std=sd,lo=lo,hi=hi,header_mean=hm,header_std=hd)
        cells=np.clip(np.floor((x[:,[0,-1]]-lo)/(hi-lo)*16),0,15).astype(int)
        h[:,1:]=(h[:,1:]-hm)/hd;h=np.column_stack([h,cells[:,:,0]*16+cells[:,:,1]])
        p=Path(part);p.mkdir(exist_ok=True)
        np.save(p/'traj.npy',((x-mu)/sd).astype('f'));np.save(p/'head.npy',h.astype('f'))
        np.savez_compressed(p/'index.npz',ids=ids,classes=labels)
        (p/'counts.json').write_text(json.dumps(Counter(labels)))
    Path('COMPLETE').touch()
if __name__=='__main__':main()
