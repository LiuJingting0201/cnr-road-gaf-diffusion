"""Plot the first two validation requests per class across the speed experiments."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import torch

from five_class_evaluate import MODES, read_observations, stratify
from plot_supervisor_sumo_maps import map_shapes
from probe_sampling_guidance import DATA, OLD, draw, make_model


def main(run, netfile, out):
    out.mkdir(exist_ok=True, parents=True)
    with np.load(DATA / 'validation/index.npz') as index:
        rows, _ = stratify(index['classes'])
        ids, labels = index['ids'][rows], index['classes'][rows]
    slots = np.concatenate([np.flatnonzero(labels == cls)[:2] for cls in MODES])
    codes = np.searchsorted(MODES, labels[slots])
    head = np.load(DATA / 'validation/head.npy')[rows[slots]]
    noise = np.random.default_rng(42).standard_normal((len(rows), 2, 200), dtype='f')[slots]
    norm = dict(np.load(DATA / 'normalization.npz'))
    observed = read_observations(DATA, ids[slots])
    paths = {}
    for arm, title in [('C', 'Coordinates'), ('D', 'Speed supervision')]:
        with np.load(OLD / f'runs/all-{arm}-42/eval/draws.npz') as z:
            np.testing.assert_array_equal(z['ids'], ids)
            paths[title] = z['xy_m'][slots]
    model = make_model('C').eval()
    model.load_state_dict(torch.load(OLD/'runs/all-C-42/final.pt', map_location='cpu', weights_only=True)['weights'])
    paths['Speed guidance'] = draw(model, 'speed', head, codes, noise, norm)
    for rep, title in [('coordinate', 'Coordinates (matched budget)'), ('velocity', 'Velocity diffusion')]:
        with np.load(run/f'{rep}-42/eval/draw-cache.npz') as z:
            np.testing.assert_array_equal(z['ids'], ids)
            np.testing.assert_array_equal(z['draw_labels'][slots], codes)
            np.testing.assert_array_equal(z['draw_rows'][slots], slots)
            paths[title] = z['xy_m'][slots]
    roads = map_shapes(netfile)
    for k, raw in enumerate(observed):
        duration = len(raw)-1
        time = np.linspace(0, duration, 200)
        obs = np.column_stack([np.interp(time, np.arange(duration+1), x) for x in raw.T])
        examples = [('SUMO', obs)] + [(title, tracks[k]) for title, tracks in paths.items()]
        speeds = [np.linalg.norm(np.diff(x, axis=0), axis=1)*199/duration for _, x in examples]
        scale = Normalize(0, max(1, np.ceil(max(v.max() for v in speeds))))
        all_xy = np.concatenate([x for _, x in examples])/1000
        center = (all_xy.max(0)+all_xy.min(0))/2
        radius = max(.2, np.ptp(all_xy, axis=0).max()*.55)
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.0))
        for ax, (name, xy), speed in zip(axes.flat, examples, speeds):
            xy = xy/1000
            ax.add_collection(LineCollection(roads, colors='#d4d8db', linewidths=.42))
            trace = LineCollection(np.stack([xy[:-1],xy[1:]],axis=1), cmap='viridis', norm=scale, linewidths=1.9)
            trace.set_array(speed)
            ax.add_collection(trace)
            ax.plot(xy[0,0],xy[0,1],'o',mfc='white',mec='black',ms=4)
            ax.plot(xy[-1,0],xy[-1,1],'x',color='black',ms=5)
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),aspect='equal')
            ax.set_axis_off()
            ax.set_title(name, fontsize=10)
        fig.subplots_adjust(top=.85,bottom=.03,left=.01,right=.88,hspace=.19,wspace=.04)
        fig.colorbar(trace,cax=fig.add_axes([.90,.2,.017,.48]),label='Speed (m/s)')
        fig.suptitle(f'{ids[slots[k]]}   |   {duration} seconds',fontsize=12)
        fig.savefig(out/f'{labels[slots[k]]}-{k%2+1}.png',dpi=175)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run','net','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    main(args.run,args.net,args.output)
