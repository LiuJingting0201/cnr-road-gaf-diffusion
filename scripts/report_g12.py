"""Summarize recognition, confidence calibration and exact same-pool ranking separately."""
from pathlib import Path
import json
import numpy as np
root=Path('docs/results/g12-scorer');load=lambda k:json.loads((root/k).read_text())
s=load('scorer.json');quality=load('quality.json');dev=[];fresh=list(s['transfer'].values())
for k,r in s['transfer'].items():
    prefix='' if r['recipe']=='plain' else r['recipe']
    dev.append(load(f'{prefix}fit{k}.json'))
def mean_metrics(rows):
    return {k:np.mean([r[k] for r in rows],axis=0).tolist() for k in ['f1','acc','nll','ece','byclass']}
summary={'development':{k:mean_metrics([z[k] for z in dev]) for k in ['old','final','selected','calibrated']},'external':{k:mean_metrics([z[k] for z in fresh]) for k in ['original','selected','calibrated']}}
summary['external']['raw_tcn']=mean_metrics(load('raw_audit.json'))
cnt=np.vstack([np.ones(8),np.random.default_rng(36240).multinomial(8,[1/8]*8,2000)])
rows=s['within_pool'];ep=np.array([r['episode']-260301 for r in rows]);y=np.array([r['label'] for r in rows])
for key in ['original','selected','calibrated']:
    x=np.asarray([r[key] for r in rows]);v=0
    for c in np.unique(y):
        ix=y==c;weights=cnt[:,ep[ix]];v+=weights@x[ix]/weights.sum(1)[:,None]/5
    np.testing.assert_allclose(v[0],quality[key]['within_vs_uniform'])
    summary[key+'_vs_exact_uniform']=dict(delta=v[0].tolist(),ci=np.percentile(v[1:],[2.5,97.5],axis=0).T.tolist())
(root/'summary.json').write_text(json.dumps(summary,indent=1))
print(summary['calibrated_vs_exact_uniform'])
