"""Preserve old-control equality and summarize every route-marginal endpoint."""
import argparse
import csv
import json
from pathlib import Path

KEYS=('weighted_native_speed_w1','weighted_native_acceleration_w1','route_edge_usage_js')
METHODS=('R','Global-S','FixedRoute-S','Global-V','FixedRoute-V','Global-SV','FixedRoute-SV')


def report(source,old,out,plot=False):
    out.mkdir(parents=True,exist_ok=True);new=json.loads(source.read_text());previous=json.loads(old.read_text())
    assert new['complete'] and len(new['results'])==4 and set(new['results'])==set(previous['results'])
    rows=[];summary={}
    for cell,rec in new['results'].items():
        before=previous['results'][cell]
        for method,original in [('R','R'),('Global-S','R+S'),('Global-V','R+V'),('Global-SV','R+SV')]:
            assert rec['results'][method]==before['results'][original],(cell,method)
        for method in METHODS:
            m=rec['results'][method]['macro'];base=rec['results']['R']['macro']
            for key,value in m.items():
                rows.append({'cell':cell,'method':method,'class':'macro','metric':key,'value':value,'delta_vs_R':value-base[key] if value is not None and base[key] is not None else None})
            for cls,d in rec['results'][method]['by_class'].items():
                for key,value in d.items():
                    if isinstance(value,(float,int)):rows.append({'cell':cell,'method':method,'class':cls,'metric':key,'value':value})
                    elif key=='physical':
                        for physical,stats in value.items():
                            for stat,v in stats.items():rows.append({'cell':cell,'method':method,'class':cls,'metric':physical+'/'+stat,'value':v})
    with (out/'all-metrics.csv').open('w') as f:
        w=csv.DictWriter(f,['cell','method','class','metric','value','delta_vs_R']);w.writeheader();w.writerows(rows)
    means={m:{k:sum(r['results'][m]['macro'][k] for r in new['results'].values())/4 for k in KEYS} for m in METHODS}
    for m in METHODS:
        summary[m]={k:{'mean':means[m][k],'percent_vs_R':100*(means[m][k]/means['R'][k]-1),
            'improved_cells':sum(r['results'][m]['macro'][k]<r['results']['R']['macro'][k] for r in new['results'].values())} for k in KEYS}
    (out/'summary.json').write_text(json.dumps({'complete':True,'original_controls_exactly_equal':True,'summary':summary},indent=2)+'\n')
    lines=['# Route-marginal mechanism diagnostic','',
        'Completed on the four already-exposed G38 scene/model-seed cells. No new fitting or trajectory generation. The old R and all three global-GAF results match exactly, including every class and endpoint.','',
        '| Rule | Speed W1 | Δ vs R | Acceleration W1 | Δ vs R | Road-edge JS |','|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        a=summary[m];lines.append(f'| {m} | {a[KEYS[0]]["mean"]:.6f} | {a[KEYS[0]]["percent_vs_R"]:+.2f}% | {a[KEYS[1]]["mean"]:.6f} | {a[KEYS[1]]["percent_vs_R"]:+.2f}% | {a[KEYS[2]]["mean"]:.6f} |')
    lines += ['',
        'Lower distances are better. Means weight each class, scene and model seed equally. Expected road usage is unchanged for FixedRoute by construction; it is a control check, not an achieved model improvement.',
        '', 'Spatial GAF retains a small aggregate acceleration improvement after route-marginal preservation, but speed still worsens. Speed-GAF feedback still worsens speed and now slightly worsens acceleration. Each fixed-route variant improves acceleration in only2of4 cells, so this is not stable across scenes. Therefore route marginal distortion is only part of the problem. The data do not establish a broad GAF benefit.',
        '', 'Duration still varies within each route, so this cannot yet attribute the acceleration effect to changed motion at fixed T. The preregistered expanded route×prior-duration×noise experiment addresses that question. It must use the same expanded pool for every method.',
        '', 'Two simulator scenes, shared public templates, four model/scene cells: no independent-four-scenes or unseen-city claim. All adverse results and all per-class physical endpoints are retained in all-metrics.csv. The ongoing G39 train-time alignment study is separate and has no completed generation result yet.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        for ax,key,title in zip(axes,KEYS[:2],['Native speed distribution','Native acceleration distribution']):
            x=np.arange(3)
            ax.bar(x-.18,[summary['Global-'+v][key]['percent_vs_R'] for v in ('S','V','SV')],.35,label='Global GAF weighting',color='#bb684d')
            ax.bar(x+.18,[summary['FixedRoute-'+v][key]['percent_vs_R'] for v in ('S','V','SV')],.35,label='Preserved route marginal',color='#386e9d')
            ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,['Spatial','Speed','Both']);ax.set_ylabel('W1 change versus learned-route baseline (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('GAF reweighting after controlling route choice\nNegative values improve; exposed G38 diagnostic',fontsize=11)
        fig.savefig(out/'route-marginal-control.svg');fig.savefig(out/'route-marginal-control.png',dpi=160);plt.close(fig)
        svg=out/'route-marginal-control.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args();report(a.source,a.old,a.output,a.plot)
