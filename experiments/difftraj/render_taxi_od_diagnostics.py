"""Build a table and figure from the aggregate CPU audits, without resampling draws."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make(target,files):
    target.mkdir(parents=True,exist_ok=True)
    records=[(p.parent.parent.name,p.parent.name,json.loads(p.read_text())) for p in files]
    figs,axes=plt.subplots(1,3,figsize=(11, max(2.7,len(files)*.38+1.2)),layout='constrained',sharey=True)
    y=np.arange(len(records));names=[f'{name}, seed {seed}' for name,seed,d in records]
    for ax,title,getval in zip(axes,('OD divergence','Both endpoints in cell','Destination error (m), median'),(
            lambda d,area:d['common'][area]['od_js'],lambda d,area:d['common'][area]['joint_hit'],
            lambda d,area:d['q10_q50_q90_destination_error_m'][1])):
        if title.startswith('Destination'):
            ax.scatter([getval(d,'full') for _,_,d in records],y,s=38,color='#155F82')
        else:
            for area,offset,marker,color in [('full',-.12,'o','#155F82'),('core',.12,'s','#B95627')]:
                ax.scatter([getval(d,area) for _,_,d in records],y+offset,marker=marker,s=32,color=color,label=area)
        ax.set_title(title,fontsize=10); ax.grid(axis='x',alpha=.25)
    axes[0].legend(loc='best',fontsize=8)
    axes[0].set_yticks(y,labels=names);axes[0].invert_yaxis()
    figs.savefig(target/'od-diagnostics.png',dpi=180)
    plt.close(figs)
    lines=['# Independent taxi diagnostics','',
           'The 10k SUMO training normalization domain supplies common scoring bounds and length bins for comparisons among SUMO fits. The Porto 10k training data supply their own physical bounds and bins. All evaluation points are restored to their original per-trip sample counts before computing occupancy or length. Native scores preserve each model’s own training bounds. Real-coordinate SHA-256 hashes refer to ordered, decoded float64 positions and trip lengths. Equal hashes identify identical reference trips in identical order. The source JSON includes bounds, length-bin edges, overflow, marginals, distance error percentiles and spot checks.', '',
           '| Model | Seed | Requests | Common grid | Density JS | OD JS | Length JS | Hotspot F1 | Origin / dest JS | Exact OD hit | OD within one cell |',
           '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,seed,d in records:
        for area in ('full','core'):
            v=d['common'][area]
            lines.append(f"| {name} | {seed} | {d['n']} | {area} | {v['density_js']:.4f} | {v['od_js']:.4f} | {d['common']['length_js']:.4f} | {v['top20_cell_f1']:.3f} | {v['marginal_origin_js']:.4f} / {v['marginal_destination_js']:.4f} | {v['joint_hit']:.1%} | {v['joint_within_one_cell']:.1%} |")
    lines+=['','Core bounds cover independent 1st–99th percentiles of the training coordinates along each axis. Overflow has its own cell for density and OD divergences, and is excluded from hotspot sets and correct-endpoint counts. Full bounds clip overflow as in the original local comparison. Two endpoints are within one cell when each differs by at most one bin on each spatial axis. Core and full hit rates have different physical tolerances.',
            '','| Model | Seed | Observed distinct OD / N | Most frequent OD count | OD collision probability | Median origin / dest error (m) | Length W1 (km) |',
            '|---|---|---:|---:|---:|---:|---:|']
    for name,seed,d in records:
        q=d['common']['full']['real_od']
        lines.append(f"| {name} | {seed} | {q['distinct']} / {d['n']} | {q['max_count']} | {q['pair_collision_probability']:.5f} | {d['q10_q50_q90_origin_error_m'][1]:.0f} / {d['q10_q50_q90_destination_error_m'][1]:.0f} | {d['length_w1_km']:.2f} |")
    lines+=['','Collision probability is the fraction of ordered pairs of different observed trips sharing the same ordered OD cell pair. This table uses the common full grid.','', '![OD diagnostics](od-diagnostics.png)']
    (target/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('target',type=Path)
    a.add_argument('files',nargs='+',type=Path)
    p=a.parse_args()
    make(p.target,p.files)
