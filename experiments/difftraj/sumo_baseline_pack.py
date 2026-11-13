"""Pack native SUMO planar positions as integer centimetres, retaining full tracks."""
import csv
import json
import sys
from array import array
from collections import defaultdict,Counter
import numpy as np
groups=defaultdict(lambda:array('d'))
reasons=Counter()
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        if row['class2']=='pedestrian':
            continue
        groups[row['vehicle_id']].extend(float(row[k]) for k in ('timestep_time','vehicle_x','vehicle_y'))
delta=[]; length=[]; starts=[]; duration=[]; vehicles=[];clock=[]
for vehicle,buf in sorted(groups.items()):
    g=np.frombuffer(buf).reshape(-1,3)
    if not np.isfinite(g).all():
        reasons['nonfinite']+=1;continue
    t,idx=np.unique(g[:,0],return_index=True)
    if len(t)<120:
        reasons['short']+=1;continue
    if np.diff(t).max()>2:
        reasons['gap']+=1;continue
    pos=np.rint(g[idx,1:]*100).astype('i4')
    if np.all(pos==pos[0]):
        reasons['stationary']+=1;continue
    delta.append(np.diff(pos,axis=0,prepend=np.zeros((1,2),dtype='i4')))
    length.append(len(t));starts.append(t[0]);duration.append(t[-1]-t[0]);vehicles.append(vehicle)
    clock.append(np.median(np.diff(t)))
np.savez_compressed(sys.argv[2],delta=np.concatenate(delta),length=length,start=starts,duration=duration,taxi=vehicles,planar_unit_m=.01,timezone='UTC')
print(json.dumps(dict(vehicle_ids=len(groups),retained=len(vehicles),excluded=reasons,median_dt_s=np.median(clock),median_points=np.median(length))))
