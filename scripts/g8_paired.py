"""Descriptive paired vehicle intervals for the finite development pilot."""
import argparse,json
from pathlib import Path
import numpy as np
from g8_common import Scores,assets,reference
from cnr.cluster_wasserstein import cluster_w1

ap=argparse.ArgumentParser();ap.add_argument('--balanced',action='store_true')
prefix='balanced-' if ap.parse_args().balanced else ''
cfg,meta,data=assets(); model=Scores(meta); y=data['labels']; ref=reference(cfg,data)
rng=np.random.default_rng(100817); arms=['identity','uniform','mid_gaf','wrong','post_gaf','post_max']
def metrics(x):
    speed=np.linalg.norm(np.diff(x,axis=1),axis=-1)/.25
    return speed,np.diff(speed,axis=1)/.25,speed.sum(1,keepdims=True)*.25
classes=np.unique(y); observed=metrics(ref);vals={arm:np.zeros((1001,4)) for arm in arms}
counts={c:np.vstack([np.ones(sum(y==c)),rng.multinomial(sum(y==c),np.ones(sum(y==c))/sum(y==c),size=1000)]) for c in classes}
for seed in [42,43,44]:
    z=np.load(f'pilot-output/{prefix}generated-{seed}.npz')
    check=json.loads(Path(f'pilot-output/{prefix}seed{seed}.json').read_text())
    for arm in arms:
        pool,report=z,check
        if arm=='post_max':
            pool=np.load(f'pilot-output/terminal-generated-{seed}.npz')
            report=json.loads(Path(f'pilot-output/terminal-seed{seed}.json').read_text())
        x=pool[arm]; ms=metrics(x);correct=(model.raw(x).argmax(1)==y)
        score=np.zeros((1001,4))
        for c in classes:
            ix=y==c;w=counts[c]
            score[:,0]+=(w@correct[ix])/sum(ix)
            for j in range(3):
                score[:,1+j]+=cluster_w1(ms[j][ix],observed[j][ix],w)
        score/=len(classes)
        np.testing.assert_allclose(score[0],report['arms'][arm]['macro'][:4],rtol=2e-6,atol=1e-5)
        vals[arm]+=score/3
out={}
for other in ['uniform','post_gaf','identity','wrong','post_max']:
    delta=vals['mid_gaf']-vals[other]
    out[other]=dict(point=delta[0].tolist(),interval95=np.percentile(delta[1:],[2.5,97.5],axis=0).T.tolist())
Path(f'pilot-output/{prefix}paired.json').write_text(json.dumps(dict(metrics=['TCN agreement','speed W1','acc W1','travel W1'],direction='mid_gaf minus comparator',replicates=1000,comparisons=out),indent=2))
