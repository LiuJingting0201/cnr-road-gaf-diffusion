"""Evaluate stored windows. Never presents point proximity as route validity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cnr.data import CLASSES
from cnr.roads import RoadNetwork, CLASS_TO_VCLASSES, distance_summary

p = argparse.ArgumentParser()
p.add_argument('--samples', required=True)
p.add_argument('--map', default='data/maps/most.net.xml')
p.add_argument('--output', required=True)
p.add_argument('--dt', type=float, default=.25)
a = p.parse_args()
if a.dt <= 0:
    raise ValueError('dt must be positive')
out = Path(a.output)
if out.exists():
    raise FileExistsError(out)
data = np.load(a.samples)
xy, labels = data['xy'], data['labels']
if xy.ndim != 3 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
    raise ValueError('Invalid XY samples')
roads = RoadNetwork(a.map)
report = {'samples': a.samples, 'map_sha256': hashlib.sha256(Path(a.map).read_bytes()).hexdigest(),
          'lane_count': len(roads.lanes), 'connection_count': len(roads.connections),
          'dt_assumed_seconds': a.dt, 'vclass_hypotheses': CLASS_TO_VCLASSES,
          'scope': 'point-to-centerline geometry only; not direction, connectivity, collision or signal validity',
          'per_class': {}}
for i, name in enumerate(CLASSES):
    tracks = xy[labels == i]
    if not len(tracks):
        continue
    speed = np.linalg.norm(np.diff(tracks, axis=1), axis=-1) / a.dt
    report['per_class'][name] = {
        'windows': len(tracks), 'all_lanes': distance_summary(roads.distances(tracks)),
        'permitted_lanes_hypothesis': distance_summary(roads.distances(tracks, name)),
        'speed_mps': {'median': float(np.median(speed)), 'p95': float(np.quantile(speed, .95)),
                      'over_100_fraction': float(np.mean(speed > 100))}}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
