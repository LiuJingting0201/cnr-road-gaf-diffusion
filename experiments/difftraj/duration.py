"""Describe displacement and speed error by validation duration on the saved paired draws."""
import json
import numpy as np
from scipy.stats import wasserstein_distance
d=np.load('private-B-8000.npz')
dt=d['dt']
truth=d['reference']
draw=d['generated']
rows=[]
for ii in np.array_split(np.argsort(dt),4):
    s=np.linalg.norm(np.diff(truth[ii]),axis=1)
    g=np.linalg.norm(np.diff(draw[ii]),axis=1)
    rows.append(dict(n=len(ii),duration_s_range=[float(dt[ii].min()*199),float(dt[ii].max()*199)],median_reference_step_m=float(np.median(s)),median_generated_step_m=float(np.median(g)),speed_w1_mps=float(wasserstein_distance((s/dt[ii,None]).ravel(),(g/dt[ii,None]).ravel()))))
print(json.dumps(rows,indent=2))
