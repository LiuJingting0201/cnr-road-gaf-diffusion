"""Show complete evaluation cohorts in physical 50 m cells with a shared colour scale."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from utils.utils import resample_trajectory

X = np.arange(-50, 10000 + 1, 50)
Y = np.arange(-50, 6400 + 1, 50)
GROUPS = [('Supplied vehicles','sumo-batch32-v1'),('10k taxi model','taxi-repro-debug-v1/10k'),
          ('20k taxi model','taxi-repro-debug-v1/20k')]


def counts(xy):
    arr = np.histogram2d(xy[:,0], xy[:,1], bins=(X,Y))[0].astype('int64')
    return arr, dict(points=len(xy), outside_points=int(len(xy)-arr.sum()))


def panels(run):
    with np.load(run/'observations.npz') as raw:
        n=raw['length']
        offsets=np.r_[0,n.cumsum()]
        delta=raw['delta']
        unit=float(raw['planar_unit_m'])
        names=raw['taxi']
    output={}
    for seed in (42,43):
        with np.load(run/str(seed)/'draws.npz') as draw:
            indices=draw['ids']
            digest=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
            assert len(indices)==len(np.unique(indices))
            if seed==42:
                refs=[delta[offsets[i]:offsets[i+1]].cumsum(0)*unit for i in indices]
                ref=np.concatenate(refs)
                output['observed']=counts(ref)
                ref_hash=hashlib.sha256(ref.tobytes()).hexdigest()
                ref_ids=names[indices]
                order=digest
            else:
                assert order==digest
            gen=np.concatenate([resample_trajectory(g,int(n[i])) for g,i in zip(draw['xy'],indices,strict=True)])
            assert len(gen)==output['observed'][1]['points'] and np.isfinite(gen).all()
            output[str(seed)] = counts(gen)
    return output, len(indices), ref_hash, ref_ids


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    fig,axes=plt.subplots(3,3,figsize=(10.5,7.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.092,right=.979,bottom=.12,top=.882,wspace=.045,hspace=.26)
    all_rows = [(label, *panels(args.runs/run)) for label,run in GROUPS]
    assert all_rows[1][3]==all_rows[2][3]
    np.testing.assert_equal(all_rows[1][4],all_rows[2][4])
    np.testing.assert_equal(all_rows[1][1]['observed'][0],all_rows[2][1]['observed'][0])
    per_million = lambda a,meta: a*1e6/meta['points']
    largest = max(per_million(a,stat).max() for _,r,*_ in all_rows for a,stat in r.values())
    smallest = min(1e6/stat['points'] for _,r,*_ in all_rows for a,stat in r.values())
    norm=LogNorm(vmin=smallest,vmax=largest)
    cmap=plt.get_cmap('magma_r').copy()
    cmap.set_bad('#ffffff')
    stats={}
    for i,(group,result,requests,digest,_) in enumerate(all_rows):
        stats[group]=dict(requests=requests,ordered_reference_sha256=digest)
        for j,(column,title) in enumerate([('observed','Observed'),('42','Generated · seed 42'),('43','Generated · seed 43')]):
            a,stat=result[column]
            ax=axes[i,j]
            image=ax.imshow(np.ma.masked_equal(per_million(a,stat).T,0),origin='lower',
                            extent=(X[0]/1000,X[-1]/1000,Y[0]/1000,Y[-1]/1000),interpolation='nearest',
                            norm=norm,cmap=cmap,aspect='equal')
            pct=100*stat['outside_points']/stat['points']
            ax.text(.985,.955,f'{pct:.2f}% outside',ha='right',va='top',fontsize=7.8,
                    transform=ax.transAxes,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1.6))
            if i==0:
                ax.set_title(title,fontsize=12,weight='semibold',loc='left',pad=8)
            if j==0:
                ax.set_ylabel(f'{group}\nNorth (km)',fontsize=9)
            if i==2:
                ax.set_xlabel('East (km)',fontsize=8.5)
            ax.tick_params(length=2,labelsize=8,color='#8d9ba0')
            for spine in ax.spines.values():
                spine.set_color('#b7c0c3');spine.set_linewidth(.6)
            stats[group][column]={**stat,'occupied_cells':int((a>0).sum()),'outside_percent':pct}
    fig.text(.065,.97,'Spatial occupancy across all evaluation journeys',va='top',fontsize=17,weight='semibold')
    fig.text(.065,.925,'Supplied cohort: 177 requests. Simulated taxi cohort: the same 1,000 requests in both rows.',
             fontsize=10,color='#4b585e')
    cax=fig.add_axes([.265,.043,.53,.014])
    colorbar=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=[10,100,1000,10000])
    colorbar.ax.tick_params(labelsize=8,length=2)
    colorbar.ax.set_title('Positions per million observations in each 50 m × 50 m cell (log scale)',fontsize=8.5,pad=4)
    outfile=args.output/'figures/occupancy.png'
    fig.savefig(outfile,dpi=190)
    plt.close(fig)
    record=dict(bounds_m=[int(X[0]),int(X[-1]),int(Y[0]),int(Y[-1])],cell_m=50,
                normalization='Counts / total positions in that panel, including off-map positions, multiplied by 1,000,000.',
                rows=stats)
    (args.output/'occupancy.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
