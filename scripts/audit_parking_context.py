"""Exploratory proximity to public scenario parking corridors, not a match proof.

Reconstruct approximate parallel-parking center corridors. Does not reproduce
SUMO slot placement, vehicle-front offsets or historical configuration versions.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import shapely
from shapely import STRtree
from shapely.ops import substring
from cnr.roads import RoadNetwork

network=RoadNetwork('data/maps/most.net.xml');index={x['id']:i for i,x in enumerate(network.lanes)}
source=Path('data/maps/most.parking.allvisible.add.xml');areas=ET.parse(source).getroot().findall('parkingArea')
shapes=[];ids=[];skipped=[]
for area in areas:
    if area.get('lane') not in index or float(area.get('angle',0))!=0 or area.findall('space'):
        skipped.append(area.get('id'));continue
    i=index[area.get('lane')];lane=network.lanes[i];shape=network.shapes[i]
    lane_length=float(lane['length']);start=float(area.get('startPos',0));end=float(area.get('endPos',lane_length))
    if start<0:start+=lane_length
    if end<0:end+=lane_length
    start=np.clip(start,0,lane_length);end=np.clip(end,0,lane_length)
    segment=substring(shape,start/lane_length*shape.length,end/lane_length*shape.length)
    if segment.geom_type!='LineString':skipped.append(area.get('id'));continue
    offset=0. if area.get('onRoad','false')=='true' else (float(lane.get('width',3.2))+float(area.get('width',3.2)))/2
    sign=1 if area.get('lefthand','false')=='true' else -1
    corridor=shapely.offset_curve(segment,sign*offset)
    if corridor.is_empty:skipped.append(area.get('id'));continue
    shapes.append(corridor);ids.append(area.get('id'))
tree=STRtree(shapes);rows=json.loads(Path('docs/results/topology-t1b-full/results.json').read_text())['calibration']['rows']
real=np.load('data/processed/most64/val.npz');result=[]
for row in rows:
    if row['geometry_covered']:continue
    x=real['xy'][row['index']]
    if np.linalg.norm(x[-1]-x[0])>5:continue
    center=x.mean(0);point=shapely.Point(center);near=int(tree.nearest(point));d=shapely.distance(point,tree.geometries[near])
    result.append({'validation_index':row['index'],'class':row['class'],
        'nearest_parking_id':ids[near],'distance_to_approximate_corridor_m':float(d)})
d=np.array([x['distance_to_approximate_corridor_m'] for x in result])
report={'scope':'post-hoc source-context proximity check; not confirmed original simulation or parking occupancy',
    'source_url':'https://github.com/lcodeca/MoSTScenario/blob/master/scenario/in/add/most.parking.allvisible.add.xml',
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'config_url':'https://github.com/lcodeca/MoSTScenario/blob/master/scenario/most.sumocfg',
    'sumo_reference':'https://sumo.dlr.de/docs/Simulation/ParkingArea.html',
    'approximation':'lane-subsegment offset by half lane width plus half parking width; no exact slot/vehicle-front reconstruction',
    'areas':len(areas),'corridors_used':len(shapes),'skipped_area_ids':skipped,
    'selected_windows':'non-pedestrian validation windows with endpoint displacement <=5m and incomplete 2m permitted-lane candidate coverage',
    'windows':len(d),'within_2m':float(np.mean(d<=2)),'within_5m':float(np.mean(d<=5)),
    'distance_median_m':float(np.median(d)),'distance_p95_m':float(np.quantile(d,.95)),
    'rows':result,'limitations':['many windows share vehicles and locations; not independent samples',
                                 'proximity alone does not prove parking','source config may differ from data generator',
                                 'parking corridors are for context only, not used to modify pass/fail metrics']}
out=Path('docs/results/topology-t1b-full/parking_context.json');out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
