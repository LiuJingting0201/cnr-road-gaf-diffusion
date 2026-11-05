"""Rebuild the concise result tables from the finite GPU runs."""
import json
from pathlib import Path
import numpy as np
root=Path('docs/results/g8-development')
read=lambda name:json.loads((root/name).read_text())
seeds=[42,43,44]
probe=read('probe.json')
summary={'probe':{},'class_switch':{},'sampling':{}}
summary['probe']['changed_images']=sum(r['unique_images']>1 for r in probe)
for metric in ['teacher','tcn','speed','acc']:
    summary['probe'][metric+'_best_minus_mean']=float(np.mean([r[metric][np.argmax(r['teacher'])]-np.mean(r[metric]) for r in probe]))
rows=[read('intervention.json' if s==42 else f'intervention-{s}.json')['rows'] for s in seeds]
assert all(len(r)==16 and all(all(v['permitted']) for v in r) for r in rows)
orig=read('seed42.json')['mid']; source=np.array([orig[r['condition']]['label'] for r in rows[0]])
rng=np.random.default_rng(99117)
boot=np.concatenate([rng.choice(np.flatnonzero(source==c),(2000,sum(source==c)),replace=True) for c in [1,3]],1)
for metric in ['travel','stop','tcn_bus','gaf_bus']:
    a=np.array([[[np.mean(v[c][metric]) for c in ['bus','private_car']] for v in r] for r in rows])
    delta=a[:,:,0]-a[:,:,1]
    summary['class_switch'][metric]=dict(bus_car=a.mean((0,1)).tolist(),delta_by_seed=delta.mean(1).tolist(),paired95=np.percentile(delta.mean(0)[boot].mean(1),[2.5,97.5]).tolist(),positive_routes=int(np.sum(delta.mean(0)>0)))
summary['class_switch']['valid_of_768']=sum(v[c]['valid'] for r in rows for v in r for c in ['bus','private_car'])
summary['class_switch']['caps']=sum(v[c]['caps'] for r in rows for v in r for c in ['bus','private_car'])
for prefix in ['','balanced-']:
    ds=[read(f'{prefix}seed{s}.json') for s in seeds]
    for arm in ds[0]['arms']:
        r=[d['arms'][arm] for d in ds]
        summary['sampling'][prefix+arm]=dict(macro=np.mean([x['macro'] for x in r],0).tolist(),diversity=float(np.mean([x['pair_distance'] for x in r])),valid_of_99=sum(x['selected_graph_pass'] for x in r),caps=sum(x['selected_caps'] for x in r))
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['class_switch'],indent=1))
