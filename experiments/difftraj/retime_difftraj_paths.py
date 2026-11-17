"""Apply a generated velocity clock to a separately generated coordinate polyline."""
import argparse
import json
from pathlib import Path

import numpy as np
from five_class_evaluate import MODES, features, occupancy, read_observations
from pooled_velocity_evaluate import scores
from velocity_label_balance import coverage
from velocity_label_contrasts import SIGNATURE, motion_signature


def arclength(p):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0),axis=1))]


def retime(route, motion):
    arc, clock = arclength(route), arclength(motion)
    if arc[-1] <= 0 or clock[-1] <= 0:
        raise ValueError('Degenerate trip')
    keep = np.r_[True, np.diff(arc)>0]
    s = clock/clock[-1]*arc[-1]
    xy = np.column_stack([np.interp(s,arc[keep],v[keep]) for v in route.T])
    xy[[0,-1]] = route[[0,-1]]
    return xy, arc[-1]/clock[-1]


def run(data, models, seed):
    with np.load(models/f'coordinate-{seed}/eval/draw-cache.npz') as c, np.load(models/f'velocity-{seed}/eval/draw-cache.npz') as v:
        for key in ('ids','validation_rows','draw_rows','draw_labels','true_labels','keys','head_sha256','noise_sha256'):
            np.testing.assert_array_equal(c[key],v[key])
        route, vel = (z['xy_m'].astype('d') for z in (c,v))
        ids, rows, draw_rows, target, actual = (c[k] for k in ('ids','validation_rows','draw_rows','draw_labels','true_labels'))
    paths, factors = zip(*(retime(p,q) for p,q in zip(route,vel)))
    xy = np.stack(paths)
    n=len(ids)
    np.testing.assert_array_equal(draw_rows[:n],np.arange(n))
    np.testing.assert_array_equal(target[:n],actual)
    scale=dict(np.load(data/'normalization.npz'))
    h=np.load(data/'validation/head.npy')[rows]
    raw=read_observations(data,ids)
    duration=np.array([len(p)-1 for p in raw])
    reference=np.load(data/'validation/traj.npy',mmap_mode='r')[rows]*scale['std']+scale['mean']
    tr=np.load(data/'train/head.npy',mmap_mode='r')[:,1].astype('d')*scale['header_std'][0]+scale['header_mean'][0]
    bins=np.unique(np.quantile(tr,np.arange(1,20)/20))
    per_class,macro=scores(raw,features(reference,duration),occupancy(raw,scale['lo'],scale['hi']),bins,
                           xy[:n],h,actual,duration,scale)
    matched,info=coverage(h,h[:,4].astype('d')*scale['header_std'][3]+scale['header_mean'][3],actual,ids)
    a,b=MODES.index('bicycle'),MODES.index('bus')
    left,right=matched[a,b]
    grid=np.full((n,len(MODES),len(SIGNATURE)),np.nan)
    grid[draw_rows,target]=motion_signature(xy,duration[draw_rows])
    contrast=((grid[left,b]-grid[left,a])+(grid[right,b]-grid[right,a])).mean(0)/2
    return dict(seed=seed,requests=n,macro=macro,per_class=per_class,bus_bicycle_pairs=len(left),
                bus_minus_bike=dict(zip(SIGNATURE,contrast)),matched_ids=info['bus_minus_bicycle']['matched_ids_sha256'],
                length_scale_p10_p50_p90=np.quantile(factors[:n],[.1,.5,.9]).tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','models','output'): p.add_argument(k,type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps([run(a.data,a.models,s) for s in (42,43)],indent=2,allow_nan=False))
