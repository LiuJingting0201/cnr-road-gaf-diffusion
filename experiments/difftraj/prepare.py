"""Prepare a paired, vehicle-disjoint development pilot on the supplied SUMO export."""
import hashlib
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
rows = []
for vehicle, g in df.groupby('vehicle_id', sort=True):
    g = g.sort_values('timestep_time').drop_duplicates('timestep_time')
    if len(g) < 20 or g.class2.nunique() != 1 or g.class2.iloc[0] == 'pedestrian':
        continue
    t = g.timestep_time.to_numpy()
    xy = g[['vehicle_x','vehicle_y']].to_numpy()
    grid = np.linspace(t[0], t[-1], 200)
    distance = np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()
    if distance < 1 or np.diff(t).max() > 2:
        continue
    pos = np.stack([np.interp(grid,t,v) for v in xy.T])
    # Interval mean of simulator speed, integrated on the observed clock.
    v = g.vehicle_speed.to_numpy()
    if not np.isfinite(v).all() or not np.isfinite(xy).all():
        continue
    arc = np.r_[0,np.cumsum((v[:-1]+v[1:])*.5*np.diff(t))]
    speed = np.diff(np.interp(grid,t,arc))/np.diff(grid)
    dur = t[-1]-t[0]
    tag = int(hashlib.sha256(str(vehicle).encode()).hexdigest()[:8],16)%5
    rows.append((pos,speed,[int(t[0]//300)%288,distance,dur,len(g),distance/(len(g)-1),distance/dur],g.class2.iloc[0],tag))
pos,v,attrs,labels,split=map(np.asarray,zip(*rows))
train=split!=0
lo=pos[train].min(axis=(0,2)); hi=pos[train].max(axis=(0,2))
def cell(x):
    q=np.clip(((x-lo)/(hi-lo)*16).astype(int),0,15)
    return q[:,0]*16+q[:,1]
h=np.column_stack([attrs,cell(pos[:,:,0]),cell(pos[:,:,-1])])
h[:,1:6]=(h[:,1:6]-h[train,1:6].mean(0))/h[train,1:6].std(0).clip(1e-6)
classes,y=np.unique(labels,return_inverse=True)
np.savez('pilot.npz',xy=pos.astype('f'),v=v.astype('f'),h=h.astype('f'),dt=(attrs[:,2]/199).astype('f'),y=y,train=train,classes=classes)
print(dict(zip(classes,np.bincount(y))),train.sum(),(~train).sum())
