"""Compare spatial occupancy on 1,000 fixed taxi requests at the two final checkpoints."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from plot_supervisor_sumo_occupancy import X, Y, counts
from utils.utils import resample_trajectory


def sampler_source(path):
    module = ast.parse(path.read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    return ast.dump(node, include_attributes=False)


def heatmap(run, old_score_ids, n):
    with np.load(run/'draws.npz') as model:
        ids, xy = model['ids'], model['xy']
    score = json.loads((run/'development.json').read_text())
    np.testing.assert_equal(ids, old_score_ids)
    assert len(ids) == len(np.unique(ids)) == score['requests'] == 1000
    assert score['epoch'] == 200 and np.isfinite(xy).all()
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == score['request_index_sha256']
    positions = np.concatenate([resample_trajectory(tr, int(n[idx])) for tr,idx in zip(xy,ids,strict=True)])
    return counts(positions)


def plot(groups, out):
    fig,axes=plt.subplots(2,3,figsize=(10.6,5.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.985,bottom=.158,top=.83,wspace=.05,hspace=.27)
    stats={}
    points=groups[42]['observed'][1]['points']
    vmax=max(1e6*a.max()/points for row in groups.values() for a,_ in row.values())
    norm=LogNorm(vmin=1e6/points,vmax=vmax)
    cmap=plt.get_cmap('magma_r').copy()
    cmap.set_bad('white')
    for i,seed in enumerate([42,43]):
        stats[str(seed)]={}
        for j,(name,label) in enumerate([('observed','Observed'),('original','Initial training'),('stabilized','Clipping + cosine')]):
            ax=axes[i,j]; a,s=groups[seed][name]
            assert s['points'] == points and a.sum()+s['outside_points'] == points
            img=ax.imshow(np.ma.masked_equal(a*1e6/points,0).T,origin='lower',
                          extent=(X[0]/1000,X[-1]/1000,Y[0]/1000,Y[-1]/1000),
                          interpolation='nearest',aspect='equal',norm=norm,cmap=cmap)
            outside=100*s['outside_points']/points
            ax.text(.984,.951,f'{outside:.2f}% outside',ha='right',va='top',fontsize=8,
                    transform=ax.transAxes,bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=1.5))
            if i==0:
                ax.set_title(label,fontsize=12,weight='semibold',loc='left',pad=7)
            if j==0:
                ax.set_ylabel(f'Seed {seed}\nNorth (km)',fontsize=9)
            if i==1:
                ax.set_xlabel('East (km)',fontsize=9)
            ax.tick_params(length=2.5,labelsize=8,color='#859398')
            for spine in ax.spines.values():
                spine.set_color('#b7c0c3');spine.set_linewidth(.6)
            stats[str(seed)][name]=dict(**s,outside_percent=outside,occupied_cells=int((a>0).sum()))
    fig.text(.077,.972,'Spatial occupancy before and after optimizer changes',va='top',
             fontsize=17,weight='semibold')
    fig.text(.077,.901,'20,000 training journeys  ·  1,000 paired requests  ·  final epoch 200',
             fontsize=10,color='#4b585e')
    cax=fig.add_axes([.30,.042,.48,.017])
    bar=fig.colorbar(img,cax=cax,orientation='horizontal',ticks=[x for x in [10,100,1000,10000] if x<=vmax])
    bar.ax.tick_params(labelsize=8,length=2)
    bar.ax.set_title('Positions per million in each 50 m × 50 m cell (log scale)',fontsize=8.6,pad=3)
    tmp=out/'stability-occupancy.tmp.png'
    fig.savefig(tmp,dpi=190)
    plt.close(fig)
    tmp.replace(out/'stability-occupancy.png')
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('original','stable','out'):
        p.add_argument(k,type=Path)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.wait:
        while not all((args.stable/str(s)/'final/COMPLETE').is_file() for s in (42,43)):
            if (args.stable/'FAILED').exists():
                raise RuntimeError('Stabilized training or evaluation failed before both final outputs were complete')
            time.sleep(30)
    for s in (42,43):
        assert (args.stable/str(s)/'final/COMPLETE').is_file()
    old_sampler=sampler_source(args.original.parent/'porto.py')
    assert old_sampler==sampler_source(args.stable/'porto.py')
    for k in ['observations.npz','prepared.npz']:
        assert (args.original/k).samefile(args.stable/k)
    with np.load(args.original/'42/draws.npz') as raw:
        ids=raw['ids']
    with np.load(args.original/'observations.npz') as raw:
        n=raw['length']; off=np.r_[0,n.cumsum()]
        delta=raw['delta']; unit=float(raw['planar_unit_m'])
        refs=np.concatenate([np.cumsum(delta[off[k]:off[k+1]],0)*unit for k in ids])
        observed=counts(refs)
    groups={}
    for seed in (42,43):
        groups[seed]=dict(observed=observed,original=heatmap(args.original/str(seed),ids,n),
                          stabilized=heatmap(args.stable/str(seed)/'final',ids,n))
    rows=plot(groups,args.out)
    output=dict(reference_sha256=hashlib.sha256(refs.tobytes()).hexdigest(),
                request_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                shared_sampler_ast_sha256=hashlib.sha256(old_sampler.encode()).hexdigest(),
                grid_bounds_m=[int(X[0]),int(X[-1]),int(Y[0]),int(Y[-1])],cell_m=50,
                checkpoint='final epoch 200',requests=1000,rows=rows)
    tmp=args.out/'stability-occupancy.tmp.json'
    tmp.write_text(json.dumps(output,indent=2)+'\n')
    tmp.replace(args.out/'stability-occupancy.json')


if __name__=='__main__':
    main()
