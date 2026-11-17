"""Compare saved observed, baseline and endpoint-loss trips on the same SUMO map."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

from five_class_evaluate import MODES, read_observations
from plot_supervisor_sumo_maps import map_shapes


def plot(args, seed):
    paths = []
    for directory in [args.baseline, args.run]:
        with np.load(directory/f'velocity-{seed}/eval/draw-cache.npz') as z:
            labels = z['true_labels']
            slots = np.array([np.flatnonzero(labels==i)[0] for i in range(len(MODES))])
            np.testing.assert_array_equal(z['draw_rows'][slots], slots)
            np.testing.assert_array_equal(z['draw_labels'][slots], labels[slots])
            identity = tuple(z[k].item() for k in ('noise_sha256','head_sha256','request_sha256'))
            ids = z['ids'][slots]
            rows = z['validation_rows'][slots]
            if paths:
                assert identity == baseline_id
                np.testing.assert_array_equal(base_ids, ids)
            baseline_id, base_ids = identity, ids
            paths.append(z['xy_m'][slots])
    raw = read_observations(args.data, ids)
    with np.load(args.data/'normalization.npz') as z:
        lo, width = z['lo'], (z['hi']-z['lo'])/16
    cells = np.load(args.data/'validation/head.npy')[rows,7].astype(int)
    roads = map_shapes(args.net)
    fig, axes = plt.subplots(5,3,figsize=(8.8,12.5))
    for r,(mode,track,cell) in enumerate(zip(MODES,raw,cells)):
        time = np.linspace(0,len(track)-1,200)
        obs = np.stack([np.interp(time,np.arange(len(track)),a) for a in track.T], axis=1)
        trips = [obs,paths[0][r],paths[1][r]]
        speeds = [np.linalg.norm(np.diff(p,axis=0),axis=1)/(time[1]-time[0]) for p in trips]
        scale = Normalize(0,np.ceil(max(v.max() for v in speeds)))
        corner = (lo+np.array([cell//16,cell%16])*width)/1000
        extent = np.vstack([np.concatenate(trips)/1000,corner,corner+width/1000])
        center = (extent.max(0)+extent.min(0))/2
        radius = max(.2,.54*np.ptp(extent,axis=0).max())
        for col,(ax,path,v) in enumerate(zip(axes[r],trips,speeds)):
            xy = path/1000
            ax.add_collection(LineCollection(roads,colors='#d5d9dc',linewidths=.5))
            ax.add_patch(Rectangle(corner,*(width/1000),fill=False,ls=':',ec='#b52534',lw=1.4))
            line = LineCollection(np.stack([xy[:-1],xy[1:]],axis=1),cmap='viridis',norm=scale,linewidths=2)
            line.set_array(v)
            ax.add_collection(line)
            ax.plot(*xy[0],marker='o',ms=4,mec='black',mfc='white')
            ax.plot(*xy[-1],marker='x',ms=6,color='black')
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),aspect='equal')
            ax.set_axis_off()
            error = np.linalg.norm(path[-1]-obs[-1])
            name = ('SUMO','Velocity','Velocity + endpoint loss')[col]
            ax.set_title(f'{mode} · {name}\n'+(str(ids[r]) if not col else f'Endpoint gap {error:.0f} m'),fontsize=8.5)
        fig.colorbar(line,ax=axes[r].tolist(),fraction=.018,pad=.014,label='m/s')
    fig.suptitle(f'Same request and noise · seed {seed}\nCircle: start   ×: end   dotted box: requested destination cell',fontsize=12)
    fig.savefig(args.output/f'map-{seed}.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','baseline','net','output','data'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for seed in (42,43):
        plot(args,seed)
