"""Complete-grid summary; never select a favorable class, seed or metric."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from g33_train import digest,save
from g33_model import CLASSES

ARMS=('B','S','V','SV','P','SH','RAW')
PRIMARY=('selected_native_speed_w1','selected_native_acceleration_w1')


def report(root,out):
    out.mkdir(parents=True,exist_ok=False);rows=[];sources={}
    for episode in range(390201,390209):
        for arm in ARMS:
            for seed in (42,43):
                p=root/str(episode)/'evaluation'/f'{arm}-{seed}'/'complete.json'
                receipt=json.loads(p.read_text());assert receipt['complete'];sources[str(p.relative_to(root))]=digest(p)
                result=receipt['results']['R']
                diagnostic=json.loads(p.with_name('diagnostics.json').read_text())
                assert diagnostic['complete']
                for label,c in enumerate(CLASSES):
                    group=result['by_class'][c]
                    flat={k:v for k,v in group.items() if k!='physical'}
                    flat.update({f'physical/{k}/{stat}':v[stat] for k,v in group['physical'].items() for stat in ('n','w1','mae')})
                    flat.update({f'diagnostic/{k}':v for k,v in diagnostic['by_class'][c].items() if not isinstance(v,list)})
                    rows.append(dict(episode=episode,cohort=(episode-390201)//2,arm=arm,seed=seed,participant=c,**flat))
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'all-class-metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
    measures=sorted({k for r in rows for k in r if k.endswith('_w1') or k.endswith('/w1') or k.endswith('/mae')})
    summary={};deltas={};rng=np.random.default_rng(39090);boot=rng.integers(0,4,(2000,4))
    for arm in ARMS:
        summary[arm]={};deltas[arm]={}
        for metric in measures:
            selected=[r for r in rows if r['arm']==arm]
            values=[r.get(metric) for r in selected]
            summary[arm][metric]=float(np.mean(values)) if all(v is not None for v in values) else None
            cell=[];cohort=[]
            for episode in range(390201,390209):
                for seed in (42,43):
                    a=[r.get(metric) for r in selected if r['episode']==episode and r['seed']==seed]
                    b=[r.get(metric) for r in rows if r['arm']=='B' and r['episode']==episode and r['seed']==seed]
                    cell.append(float(np.mean(a)-np.mean(b)) if len(a)==len(b)==5 and all(v is not None for v in a+b) else None)
            if all(v is not None for v in cell):
                cohort=np.asarray(cell).reshape(4,4).mean(1)
                ci=np.quantile(cohort[boot].mean(1),[.025,.975])
                deltas[arm][metric]={'difference_vs_B':float(cohort.mean()),'cohort_bootstrap95_descriptive':ci.tolist(),
                     'better_cells_of16':int(np.sum(np.array(cell)<0)),'cell_differences':cell,'cohort_differences':cohort.tolist()}
            else:deltas[arm][metric]={'difference_vs_B':None,'missing_support':True}
    primary_better=all(deltas['SV'][k]['difference_vs_B'] is not None and deltas['SV'][k]['difference_vs_B']<0 for k in PRIMARY)
    finding=('SV improves both aggregate primary endpoints; inspect class/scene trade-offs, diversity and controls before any broad claim.'
             if primary_better else 'SV does not improve both aggregate primary endpoints; no broad physical-quality improvement claim.')
    save(out/'complete.json',{'complete':True,'rows':len(rows),'cells':112,'summary':summary,'deltas':deltas,'finding':finding,
        'primary':list(PRIMARY),'weighting':'equal classes, seeds, windows, cohorts','source_sha256':sources,
        'limits':['Four OD cohorts, repeated windows and common bus templates; bootstrap is descriptive.',
                  'Native route and duration marginals fixed by design; no learned duration gain.',
                  'This primary report precedes utility and expanded-pool mechanism follow-up.']})
    lines=['# G39 direct synthesis results','',finding,'',
        'Primary: first direct draw, no terminal GAF ranking. Lower W1 is better.','',
        '| Arm | Speed W1 | Acceleration W1 |','|---|---:|---:|']
    for a in ARMS:
        vals=[summary[a][k] for k in PRIMARY];lines.append('| '+a+' | '+' | '.join('unsupported' if v is None else f'{v:.6f}' for v in vals)+' |')
    lines+=['','All classes, scenes, seeds and secondary metrics are retained in `all-class-metrics.csv` and `complete.json`.',
            'The classifiers are self-trained; confidence alone is not a quality result. RAW differs in architecture and pretraining.',
            'No unseen-city, real-world or universal GAF claim follows from this simulator study.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root,a.output)
