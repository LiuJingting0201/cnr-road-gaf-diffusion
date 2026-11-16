"""Audit contiguous bus low-speed intervals on the common elapsed-time grid."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from plot_supervisor_sumo_maps import map_shapes

root=Path('/workspace/cnr-experiments')
data=root/'difftraj-five-40k-v1'
saved=root/'guidance-full-v1'
net=root/'gaf-inloop-v1/g27-job/g27-data/kit-check/assets/most.net.xml'
assert hashlib.sha256(net.read_bytes()).hexdigest()==json.loads((data/'manifest.json').read_text())['source_network_sha256']
with np.load(data/'validation/index.npz') as z:
    keep=np.flatnonzero(z['classes']=='bus')
    ids=z['ids'][keep]
assert len(np.unique(ids))==1000
with np.load(data/'normalization.npz') as n:
    obs=np.load(data/'validation/traj.npy',mmap_mode='r')[keep]*n['std']+n['mean']
    dur=np.rint(np.load(data/'validation/head.npy')[keep,2]*n['header_std'][1]+n['header_mean'][1])
dt=dur/199
paths={'SUMO':obs}
for seed in (42,43):
    for method in ('base','speed'):
        with np.load(saved/f'draws-{method}-{seed}.npz') as d:
            ix={k:i for i,k in enumerate(d['ids'])}
            paths[f'{method}-{seed}']=d['xy'][[ix[k] for k in ids]].astype('d')

def speeds(tracks):
    return np.linalg.norm(np.diff(tracks,axis=1),axis=2)/dt[:,None]

def describe(tracks):
    event=[]; entered=[]; interior=[]; shares=[]; rates=[]; fragments=[]
    for i,mask in enumerate(speeds(tracks)<1):
        borders=np.diff(np.r_[False,mask,False].astype(int))
        a,b=np.flatnonzero(borders==1),np.flatnonzero(borders==-1)
        length=(b-a)*dt[i]
        eligible=(a>0)&(b-a>=2)
        event.extend(length)
        entered.extend(length[eligible])
        interior.extend(length[eligible & (b<199)])
        shares.append([100*length[(length>=lo)&(length<hi)].sum()/dur[i] for lo,hi in [(0,10),(10,30),(30,60),(60,np.inf)]])
        rates.append(eligible.sum()*600/dur[i])
        fragments.append([100*length[eligible].sum()/dur[i],100*length[a==0].sum()/dur[i],100*length[(a>0)&(b-a==1)].sum()/dur[i]])
    q=lambda x: np.percentile(x,[50,75,90,95]).tolist()
    return np.sort(entered),dict(all_events=len(event),eligible=len(entered),all_q=q(event),entry_q=q(entered),
        interior_q=q(interior),mean_entry_s=float(np.mean(entered)),over30=float(np.mean(np.array(entered)>=30)),
        shares=np.mean(shares,axis=0).tolist(),rate=float(np.mean(rates)),parts=np.mean(fragments,axis=0).tolist())
if __name__ == '__main__':
    summary={}; events={}
    for name,track in paths.items():
        events[name],summary[name]=describe(track)
    Path('summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    colors=['#202932','#9370a6','#007a74']
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True,sharex='col',sharey='col')
    for row,s in enumerate((42,43)):
        for j,(key,label) in enumerate(zip(('SUMO',f'base-{s}',f'speed-{s}'),('SUMO','Original','Guided'))):
            x=events[key]; ax=axes[row,0]
            ax.step(x,np.arange(len(x))/len(x),where='post',color=colors[j],label=label,ls=['-','--','-.'][j])
            axes[row,1].bar(np.arange(4)+(j-1)*.25,summary[key]['shares'],width=.25,color=colors[j],label=label)
        ax.set(xscale='log',ylabel=f'Seed {s}   |   cumulative fraction',ylim=(0,1.02))
        axes[row,1].set_ylabel('Mean share of journey time (%)')
        axes[row,1].set_xticks(range(4),['<10','10–<30','30–<60','≥60'])
    axes[0,0].set_title('Duration of counted low-speed episodes')
    axes[0,1].set_title('Time spent in all low-speed spells')
    axes[1,0].set_xlabel('Consecutive seconds below 1 m/s')
    axes[1,1].set_xlabel('Spell duration (s)')
    axes[0,0].legend()
    fig.savefig('durations.png',dpi=180);plt.close(fig)
    roads=map_shapes(net)
    for seed in (42,43):
        fig,axes=plt.subplots(3,3,figsize=(10,9),constrained_layout=True)
        norm=Normalize(vmin=0,vmax=15,clip=True)
        for row,i in enumerate((0,499,999)):
            subset=[paths[k][i]/1000 for k in ('SUMO',f'base-{seed}',f'speed-{seed}')]
            allxy=np.concatenate(subset)
            lo,hi=allxy.min(0),allxy.max(0)
            mid=(lo+hi)/2; r=max(.18,(hi-lo).max()*.56)
            for col,xy in enumerate(subset):
                ax=axes[row,col]
                ax.add_collection(LineCollection(roads,color='#c5cbd1',linewidths=.48))
                seg=LineCollection(np.stack((xy[:-1],xy[1:]),axis=1),cmap='plasma',norm=norm,linewidths=2)
                seg.set_array(np.linalg.norm(np.diff(xy,axis=0),axis=1)*1000/dt[i]);ax.add_collection(seg)
                ax.scatter(*xy[0],s=26,facecolor='white',edgecolor='black',zorder=5)
                ax.scatter(*xy[-1],s=24,c='black',marker='x',zorder=5)
                ax.set(aspect='equal',xlim=(mid[0]-r,mid[0]+r),ylim=(mid[1]-r,mid[1]+r))
                ax.tick_params(labelsize=8)
                if col==0: ax.set_ylabel(f'Bus request {i+1}\nNorth (km)')
                if row==0: ax.set_title(['SUMO','Original DiffTraj','Speed guidance'][col])
                if row==2: ax.set_xlabel('East (km)')
        fig.suptitle(f'Bus trajectories on the MoST network   ·   seed {seed}\nOpen circle = start     × = end')
        fig.colorbar(seg,ax=axes,location='bottom',shrink=.6,aspect=38,label='Interval speed (m/s; scale capped at 15)')
        fig.savefig(f'map-{seed}.png',dpi=170);plt.close(fig)
