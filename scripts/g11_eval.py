import hashlib,json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from cnr.topology import LaneConsistency
from cnr.data import CLASSES
from cnr.episode_w1 import episode_w1

root=Path('g11-data');q=np.load(root/'requests.npz');ref=np.load(root/'references.npz')
assert np.array_equal(q['ids'],ref['ids']);ep=q['episode']-260201;y=q['labels']
arms=['mid','terminal','placebo','placebo_terminal'];ds=[dict(np.load(root/f'{s}.npz')) for s in [42,43,44]]
cnt=np.vstack([np.ones(8),np.random.default_rng(53415).multinomial(8,np.ones(8)/8,2000)])
def metrics(x):
    v=np.linalg.norm(np.diff(x,axis=-2),axis=-1)/.25
    return v,np.diff(v,axis=-1)/.25,v.sum(-1)*.25,(v<.1).mean(-1)
def init():
    global matcher
    matcher=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64)
def check(item):
    k,(c,path)=item;g=matcher.match(path,CLASSES[c]);return k,int(g['graph_consistent'] and not g['candidate_truncation_warning'])
def main():
    real=metrics(ref['xy'][:,None]);out={'n':len(y),'episodes':len(np.unique(ep)),'arms':{}};keys={};tasks={};draws={}
    for arm in arms:
        keys[arm]=[]
        for d in ds:
            for c,paths in zip(y,d[arm]):
                for path in paths:
                    k=(int(c),hashlib.sha256(path.tobytes()).digest());tasks[k]=(int(c),path);keys[arm].append(k)
    with mp.get_context('fork').Pool(8,initializer=init) as pool:valid=dict(pool.imap_unordered(check,tasks.items(),chunksize=8))
    for arm in arms:
        res=np.zeros((len(cnt),4));per={};spread=[]
        for d in ds:spread.append(np.mean([np.linalg.norm(d[arm][:,i]-d[arm][:,j],axis=-1).mean() for i,j in [(0,1),(0,2),(1,2)]]))
        vs=[metrics(d[arm]) for d in ds]
        for c in np.unique(y):
            ix=y==c;n=sum(ix);val=np.zeros_like(res)
            for v in vs:
                for j in range(4):val[:,j]+=episode_w1(v[j][ix].reshape(n,-1),real[j][ix].reshape(n,-1),ep[ix],j==3,cnt)/3
            per[CLASSES[int(c)]]=val[0].tolist();res+=val/len(np.unique(y))
        draws[arm]=res;out['arms'][arm]=dict(quality=res[0].tolist(),per_class=per,valid=sum(valid[k] for k in keys[arm]),total=len(keys[arm]),caps=int(sum(d['cap_'+arm].sum() for d in ds)),diversity=float(np.mean(spread)))
    effect=draws['mid']-draws['terminal'];control=draws['placebo']-draws['placebo_terminal']
    for k,a in [('adaptive',effect),('permuted',control),('interaction',effect-control)]:
        out[k]=dict(delta=a[0].tolist(),CI=np.percentile(a[1:],[2.5,97.5],axis=0).T.tolist())
    stages=np.concatenate([d['stage'].ravel() for d in ds]);u,n=np.unique(stages,return_counts=True)
    for d in ds:
        for c in np.unique(y):
            for j in range(3):assert np.array_equal(np.sort(d['stage'][y==c,j]),np.sort(d['placebo_stage'][y==c,j]))
    out['trigger_histogram']={int(i):int(k) for i,k in zip(u,n)}
    out['encoding_failures']=int(sum(d['failures'] for d in ds))
    (root/'results.json').write_text(json.dumps(out,indent=1))
if __name__=='__main__':main()
