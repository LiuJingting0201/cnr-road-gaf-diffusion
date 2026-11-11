"""Sampling variability of speed/length distances between actual development trips."""
import json
import numpy as np
from scipy.stats import wasserstein_distance
d=np.load('pilot.npz')
rng=np.random.default_rng(42)
valid=np.flatnonzero(~d['train'])
length=np.linalg.norm(np.diff(d['xy'],axis=-1),axis=1)
speed=length/d['dt'][:,None]
length=length.sum(1)
stats=[]
for _ in range(100):
    a,b=np.split(rng.choice(valid,size=256,replace=False),2)
    stats.append([wasserstein_distance(speed[a].ravel(),speed[b].ravel()),wasserstein_distance(length[a],length[b])])
print(json.dumps({'repeats':100,'trips_per_half':128,'quantiles':[.05,.5,.95],'speed_w1_mps':np.quantile(np.array(stats)[:,0],[.05,.5,.95]).tolist(),'length_w1_m':np.quantile(np.array(stats)[:,1],[.05,.5,.95]).tolist()},indent=2))
