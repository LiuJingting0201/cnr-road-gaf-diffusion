"""Describe a fixed complete-cohort interim snapshot without selecting a method."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


PRIMARY=('selected_native_speed_w1','selected_native_acceleration_w1')


def report(source,output):
    data=json.loads(source.read_text());rows=data['rows'];arms=data['arms']
    assert data['interim'] is True and data['complete_grid'] is False
    episodes=data['episodes'];classes=sorted({r['participant'] for r in rows})
    assert len(classes)==5 and len(episodes)%2==0
    expected={(e,a,s,c) for e in episodes for a in arms for s in (42,43) for c in classes}
    assert {(r['episode'],r['arm'],r['seed'],r['participant']) for r in rows}==expected
    assert len(rows)==len(expected)
    assert all(e+1 in episodes for e in episodes if (e-390201)%2==0)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    measures=[k for k in keys if k not in {'episode','cohort','seed','arm','participant'}]
    def avg(arm,metric,**conditions):
        values=[r[metric] for r in rows if r['arm']==arm and all(r[k]==v for k,v in conditions.items())]
        return mean(values) if values and all(v is not None for v in values) else None
    def effect(arm,metric,**conditions):
        a,b=avg(arm,metric,**conditions),avg('B',metric,**conditions)
        return {'value':a,'baseline':b,'difference':None if a is None or b is None else a-b,
                'relative_percent':100*(a/b-1) if a is not None and b else None}
    summary={a:{m:effect(a,m) for m in measures} for a in arms}
    primary={}
    for a in arms:
        primary[a]={}
        for metric in PRIMARY:
            cells=[effect(a,metric,episode=e,seed=s) for e in episodes for s in (42,43)]
            primary[a][metric]={**effect(a,metric),'better_cells':sum(x['difference']<0 for x in cells),
                'cell_count':len(cells),'by_seed':{s:effect(a,metric,seed=s) for s in (42,43)},
                'by_class':{c:effect(a,metric,participant=c) for c in classes},
                'by_cohort':{c:effect(a,metric,cohort=c) for c in sorted({r['cohort'] for r in rows})}}
    baseline={(r['episode'],r['seed'],r['participant']):r for r in rows if r['arm']=='B'}
    invariant={m:max(abs(r[m]-baseline[(r['episode'],r['seed'],r['participant'])][m]) for r in rows)
               for m in ['physical/duration_s/w1','route_edge_usage_js','expected_detour','effective_route_count']}
    assert all(x==0 for x in invariant.values()),'Route/duration changed across arms'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'all-class-metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    result={'interim':True,'complete_grid':False,'episodes':episodes,'cohort_count':data['cohort_count'],
            'rows':len(rows),'summary':summary,'primary':primary,'invariant_max_absolute_differences':invariant,
            'selection':data['selection'],'limits':data['limits'],'inference':'Descriptive only; no confidence interval, hypothesis test or outcome-driven selection.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# G39 interim: first two complete OD cohorts','',
        '**Incomplete experiment: descriptive interim only, not a thesis success/failure verdict.**','',
        'Four scenes, two complete OD cohorts, seven arms, two model seeds, five classes:56cells/280class rows.',
        'This subset was fixed from completion status before inspecting metric values. The fifth complete scene',
        'was excluded because its paired window/control grid was not yet complete at initial inspection.',
        'Equal class/seed/window/cohort means; primary is the first direct candidate, no GAF ranking.',
        'All arms have identical route and drawn-duration pairing hashes; route/duration metrics agree exactly.','',
        '| Arm | Speed W1 | Change vs B | Acceleration W1 | Change vs B |',
        '|---|---:|---:|---:|---:|']
    for a in arms:
        s,t=(summary[a][m] for m in PRIMARY)
        lines.append(f"| {a} | {s['value']:.6f} | {s['relative_percent']:+.2f}% | {t['value']:.6f} | {t['relative_percent']:+.2f}% |")
    lines+=['','Lower W1 is better; a positive change is worse. S:spatial GAF; V:speed GAF;',
            'SV:both; P:class-centroid targets; SH:another same-class trip feature;',
            'RAW:raw-series auxiliary encoders, different architecture/pretraining.','',
            'Complete secondary, class, seed and cohort results remain in summary.json/all-class-metrics.csv.',
            'No training, sampling, metric or follow-up protocol changes are made from this interim look.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.source,args.output)
