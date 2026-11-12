"""Render shared-scale city-core density and physical trip lengths from aggregate audit files."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(sys.argv[1])
cases=['42-long-on-long','42-mixed-on-mixed']
records=[json.loads((out/(name+'.json')).read_text())['plot'] for name in cases]
fig,axes=plt.subplots(2,2,figsize=(8,7),layout='constrained')
vmax=max(max(d[:256]) for r in records for d in r['density'])
for i,(r,label) in enumerate(zip(records,['120-point minimum','Including short trips'])):
    lo,hi=r['bounds']
    for j,(d,kind) in enumerate(zip(r['density'],['Observed','Generated'])):
        ax=axes[i,j]
        im=ax.imshow(100*np.reshape(d[:256],(16,16)).T,origin='lower',extent=(lo[0],hi[0],lo[1],hi[1]),vmin=0,vmax=vmax*100)
        ax.set_title(f'{kind} · {label}',fontsize=10)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
fig.colorbar(im,ax=axes,label='Share of all GPS points per cell (%)')
fig.savefig(out/'porto-density.png',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained',sharey=True)
for ax,r,label in zip(axes,records,['120-point minimum','Including short trips']):
    for q,kind in zip(r['km_quantiles'],['Observed','Generated']):
        ax.plot(q,np.arange(101)/100,label=kind)
    ax.set_xlim(0,max(q[99] for q in r['km_quantiles'])*1.08)
    ax.set(xlabel='Trip length (km)',title=label)
    ax.grid(alpha=.2)
axes[0].set_ylabel('Cumulative share of trips')
axes[1].legend()
fig.savefig(out/'porto-length.png',dpi=170)
