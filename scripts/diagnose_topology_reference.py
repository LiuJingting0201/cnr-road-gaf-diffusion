"""Post-hoc diagnostic of motion state versus full-validation coverage.

Low displacement is not proof of parking. Class2 is not a known SUMO vClass.
"""
import json
from pathlib import Path
import numpy as np
from cnr.roads import RoadNetwork

root=Path('docs/results/topology-t1b-full');report=json.loads((root/'results.json').read_text())
data=np.load('data/processed/most64/val.npz');roads=RoadNetwork('data/maps/most.net.xml')
rows=report['calibration']['rows'];indices=np.array([r['index'] for r in rows])
xy=data['xy'][indices];displacement=np.linalg.norm(xy[:,-1]-xy[:,0],axis=-1)
summaries={}
for name,mask in [('moving_endpoint_gt5m',displacement>5),('low_endpoint_displacement_le5m',displacement<=5)]:
    rr=[r for r,m in zip(rows,mask) if m];tracks=xy[mask];perclass={}
    for cls in sorted(set(r['class'] for r in rr)):
        cm=np.array([r['class']==cls for r in rr]);xx=tracks[cm];cr=[r for r in rr if r['class']==cls]
        permitted=roads.distances(xx,cls);any_lane=roads.distances(xx)
        perclass[cls]={'windows':len(cr),'whole_window_geometry_covered':float(np.mean([r['geometry_covered'] for r in cr])),
            'whole_window_graph_consistent':float(np.mean([r['graph_consistent'] for r in cr])),
            'permitted_point_distance_p95_m':float(np.quantile(permitted,.95)),
            'any_lane_point_distance_p95_m':float(np.quantile(any_lane,.95)),
            'mean_path_length_m':float(np.linalg.norm(np.diff(xx,axis=1),axis=-1).sum(1).mean())}
    summaries[name]={'windows':len(rr),'whole_window_geometry_covered':float(np.mean([r['geometry_covered'] for r in rr])),
                     'whole_window_graph_consistent':float(np.mean([r['graph_consistent'] for r in rr])),
                     'per_class':perclass}
result={'scope':'post-hoc real-validation diagnostic; no thresholds retuned',
        'split':'endpoint displacement >5m, matching the existing reversal control',
        'limitations':['endpoint displacement is not path length','low displacement does not prove parking',
                      'vClass mapping remains unconfirmed','all-lane distance includes lanes unavailable to assumed class'],
        'groups':summaries}
(root/'reference_motion_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
