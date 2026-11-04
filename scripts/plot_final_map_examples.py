"""Fixed final-test examples: seed42, first condition per class, no quality selection."""
import json
from pathlib import Path
import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from cnr.data import CLASSES
from cnr.roads import RoadNetwork

root = Path('runs/final-diffusion-e7')
pool = np.load(root / 'seed42/candidates.npz')
report = json.loads(Path('docs/results/final-diffusion-e7/seed42/results.json').read_text())
methods = ['random', 'gaf', 'tcn']
selected = {m: np.load(root / f'selection/seed42/{m}.npz') for m in methods}
roads = RoadNetwork('data/maps/most.net.xml')
classes = np.unique(pool['labels'])
fig, axes = plt.subplots(len(classes), 3, figsize=(12, 16), squeeze=False)
records = []
for row, label in enumerate(classes):
    first = np.flatnonzero(pool['labels'] == label)[0]
    condition = pool['condition_id'][first]
    group = np.flatnonzero(pool['condition_id'] == condition)
    tracks = pool['xy'][group]
    name = CLASSES[int(label)]
    allxy = tracks.reshape(-1, 2)
    center = (allxy.min(0) + allxy.max(0)) / 2
    radius = max(float(np.ptp(allxy, axis=0).max()) / 2 + 30, 50)
    lo, hi = center - radius, center + radius
    roads.distances(tracks[0, :1], name)
    tree = roads.trees[name]
    nearby = tree.geometries[tree.query(shapely.box(*lo, *hi))]
    segments = [np.asarray(line.coords) for line in nearby]
    record = {'class': name, 'condition': int(condition), 'candidate_indices': group.tolist(), 'selected': {}}
    for col, method in enumerate(methods):
        match = np.flatnonzero(np.isin(selected[method]['indices'], group))
        assert len(match) == 1
        ix = int(selected[method]['indices'][match[0]])
        track = pool['xy'][ix]
        record['selected'][method] = ix
        ax = axes[row, col]
        ax.add_collection(LineCollection(segments, colors='#c9cdd1', linewidths=.6))
        for candidate in tracks:
            ax.plot(*candidate.T, color='#9bbed4', lw=.8, alpha=.65)
        ax.plot(*track.T, color='#b84b27', lw=1.8)
        ax.scatter(*track[0], s=28, color='#21824a', zorder=5)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal', title=f'{name} | {method}')
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel('SUMO y (m)')
        if row == len(classes) - 1:
            ax.set_xlabel('SUMO x (m)')
    records.append(record)
fig.suptitle('Road-conditioned diffusion: fixed held-out test examples, seed42\nGray: permitted lanes; blue: K=8 candidates; orange: selected; green: given start', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .96))
out = Path('docs/results/final-test-e7')
fig.savefig(out / 'map_examples.png', dpi=150)
fig.savefig(out / 'map_examples.pdf')
plt.close(fig)
(out / 'map_examples.json').write_text(json.dumps({'rule': 'seed42, first condition per class; fixed before viewing; no real future supplied', 'examples': records}, indent=2))
