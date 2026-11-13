from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
src,out=map(Path,sys.argv[1:])
fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for axrow,kind in zip(axes,['destination','origin']):
    for i,ax in enumerate(axrow):
        for seed,color in [('42','#16839a'),('43','#cc572a')]:
            d=np.load(src/f'{seed}-{kind}.npz')
            for key1,key2,col,lab in [('A','B','black','Requested'),('before','after',color,'Seed '+seed)]:
                if lab=='Requested' and seed=='43':continue
                a,b=[(d[key][i]-d['A'][i])/1000 for key in (key1,key2)]
                ax.plot(*a,'o',mfc='none',mec=col)
                ax.plot(*b,'x',color=col)
                ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=col))
                ax.plot([],[],color=col,label=lab)
            ax.autoscale_view()
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.25)
        ax.set_title(f'{kind.title()} switch {i+1}')
        ax.set_xlabel('East (km)')
    axrow[0].set_ylabel('North (km)')
axes[0,-1].legend(fontsize=8)
fig.savefig(out/'switch.png',dpi=160)
