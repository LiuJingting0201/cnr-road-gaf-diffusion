"""Recover native records for clock and size contrasts. No held-out SUMO record in fit pack."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys
root=Path('/workspace/cnr-experiments')
def subset(raw,ids):
    size=raw['length']; ends=np.r_[0,size.cumsum()]; delta=raw['delta']
    return dict(delta=np.concatenate([delta[ends[i]:ends[i+1]] for i in ids]),
                **{k:raw[k][ids] for k in ['length','taxi','start']+(['duration'] if 'duration' in raw.files else [])})
def actual_time(data):
    data['start']=np.array([t-int(i.split('_')[0])%1440*60 for i,t in zip(data['taxi'],data['start'])])
    assert min(data['start'])>=0 and max(data['start'])<1500
    return data
def combine(train,ref,tag,planar):
    out={k:np.concatenate([train[k],ref[k]]) for k in train}
    out['episode_split']=np.r_[np.zeros(len(train['length']),dtype='i1'),np.full(len(ref['length']),tag,dtype='i1')]
    if planar:out.update(timezone='UTC',planar_unit_m=.01)
    return out
def native(ids):
    groups=defaultdict(list)
    for idx in ids:groups[idx.split('_')[0]].append(idx)
    result={k:[] for k in ('length','taxi','start','duration','delta')}
    for ep,needed in groups.items():
        pack=root/'difftraj-five-10k'/ep/'trips.npz'
        if not pack.exists():pack=root/'difftraj-five-20k'/ep/'trips.npz'
        with np.load(pack) as z:
            ends=np.r_[0,z['h'][:,3].astype(int).cumsum()];rr=z['raw'];ix=dict(zip(z['ids'],range(len(z['ids']))))
            for key in needed:
                j=ix[key];v=rr[ends[j]:ends[j+1]]
                xy=np.rint(v[:,1:3]*100).astype('int32');t=v[:,0]
                result['delta'].append(np.diff(xy,axis=0,prepend=np.zeros((1,2),dtype='i4')))
                for k,val in zip(('length','taxi','start','duration'),(len(t),key,int(t[0]),int(t[-1]-t[0]))):result[k].append(val)
    return {k:(np.concatenate(v) if k=='delta' else np.array(v)) for k,v in result.items()}
if __name__=='__main__':
    arm=sys.argv[1]; fresh=len(sys.argv)>2
    if arm=='porto':
        r=np.load(root/'porto-cutoff-v1/long-on-long/observations.npz');split=np.load(root/'porto-cutoff-v1/long-on-long/prepared.npz')['split']
        train=subset(r,np.sort(np.random.default_rng(13013).choice(np.flatnonzero(split<8),10000,False)))
        ref=subset(r,np.sort(np.random.default_rng(13113).choice(np.flatnonzero(split==9),1000,False)))
        pack=combine(train,ref,9,False)
    else:
        r=np.load(root/'taxi10k-base/observations.npz')
        if arm in ('10k','oldclock'):
            train=subset(r,np.flatnonzero(r['episode_split']==0))
            if arm=='10k':train=actual_time(train)
        else:
            data=np.load(root/'difftraj-five-20k/train/index.npz')
            train=native(data['ids'][data['classes']=='taxi']);assert len(train['taxi'])==20000
        if fresh:
            f=np.load(root/'taxi-debug-fresh-01/observations.npz')
            ref=actual_time(subset(f,np.arange(len(f['taxi']))))
            # A separate 1000-observation development control for scoring.
            dev=actual_time(subset(r,np.flatnonzero(r['episode_split']==8)))
            pack=combine(train,dev,8,True)
            for k in train:pack[k]=np.concatenate([pack[k],ref[k]])
            pack['episode_split']=np.r_[pack['episode_split'],np.full(len(ref['length']),9)]
        else:
            ref=subset(r,np.flatnonzero(r['episode_split']==8))
            if arm!='oldclock':ref=actual_time(ref)
            pack=combine(train,ref,8,True)
    np.savez_compressed('observations.npz',**pack)
