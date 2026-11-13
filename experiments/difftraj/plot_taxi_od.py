"""Compare the requested and sampled endpoints for the same taxi journeys."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
d=np.load(sys.argv[1]);out=Path(sys.argv[2]);lo=d['lo']/1000;hi=d['hi']/1000
r=d['obs']/1000; gen=[d[k]/1000 for k in ('s42','s43')]
def cell(x):
    q=np.clip(np.floor((x-lo)/(hi-lo)*16),0,15).astype(int)
    return q[...,0]*16+q[...,1]
rc=cell(r);assert np.array_equal(rc,d['requested'])
err=[np.linalg.norm(g-r,axis=2) for g in gen]
fig,axes=plt.subplots(2,3,figsize=(11,7),sharex=True,sharey=True,layout='constrained')
for ax,i in zip(axes.flat,range(6)):
    for pts,label,col,mark in [(r,'Observed','#222222','o'),(gen[0],'Seed 42','#087c98','s'),(gen[1],'Seed 43','#c45421','^')]:
        xy=pts[i]; ax.scatter(*xy[0],s=68,marker=mark,facecolors='none',edgecolors=col,lw=1.8,label=label+' origin')
        ax.scatter(*xy[1],s=48,marker=mark,c=col,label=label+' destination')
        if pts is not r: ax.add_collection(LineCollection(np.stack([r[i],xy],axis=1),colors=col,linewidths=1,alpha=.6))
    ax.set_title(f'Request {i+1}')
    ax.set_xticks(np.linspace(*[lo[0],hi[0]],17),minor=True);ax.set_yticks(np.linspace(lo[1],hi[1],17),minor=True)
    ax.grid(which='minor',lw=.45,alpha=.35);ax.set_aspect('equal');ax.set_xlim(lo[0]-.35,hi[0]+.35);ax.set_ylim(lo[1]-.35,hi[1]+.35)
fig.supxlabel('SUMO east coordinate (km)');fig.supylabel('SUMO north coordinate (km)')
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
fig.savefig(out/'paired-endpoints.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained');summary={}
for seed,g,e,col in zip((42,43),gen,err,['#087c98','#c45421']):
    hit=cell(g)==rc
    summary[seed]=dict(n=len(r),hits=[int(c.sum()) for c in [hit[:,0],hit[:,1],hit.all(axis=1)]],median_m=np.median(e*1000,axis=0).tolist(),p90_m=np.quantile(e*1000,.9,axis=0).tolist())
    for j,ax in enumerate(axes):
        ax.step(np.sort(e[:,j]),np.arange(1,len(e)+1)/len(e),where='post',label=f'Seed {seed}',color=col)
        ax.set(xlabel='Distance from observed endpoint (km)',title=['Origin','Destination'][j]);ax.grid(alpha=.2)
axes[0].set_ylabel('Fraction of 1,000 requests');axes[1].legend()
fig.savefig(out/'endpoint-errors.png',dpi=165);plt.close(fig)
(out/'endpoints.json').write_text(json.dumps(summary,indent=2));print(summary)
