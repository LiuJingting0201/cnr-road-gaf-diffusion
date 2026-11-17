"""Check alignment of training-route descriptors with the trip supplying the speed target."""
import json
from pathlib import Path
import argparse
import numpy as np
from route_clock_features import describe


def summarize(data, run):
    x=np.load(data/'train/traj.npy',mmap_mode='r')
    rows=np.random.default_rng(171120).choice(len(x),5000,replace=False)
    norm=dict(np.load(data/'normalization.npz'))
    obs=describe(x[rows].astype('d')*norm['std']+norm['mean'])
    stats={}
    for seed in (42,43):
        folder=run/f'features-{seed}'
        z=dict(np.load(folder/'scale.npz'))
        cond=np.load(folder/'train.npy',mmap_mode='r')[rows]*z['sd']+z['mean']
        a=obs[:,1:]-obs[:,1:].mean(1,keepdims=True)
        b=cond[:,1:]-cond[:,1:].mean(1,keepdims=True)
        denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        keep=denom>1e-8
        corr=(a[keep]*b[keep]).sum(1)/denom[keep]
        ratio=np.exp(cond[:,0]-obs[:,0])
        stats[str(seed)]=dict(rows=len(rows),median_generated_to_observed_polyline_length=float(np.median(ratio)),
            p10_p90_length_ratio=np.quantile(ratio,[.1,.9]).tolist(), within_trip_bend_profile_comparable=int(keep.sum()),
            median_within_trip_bend_profile_correlation=float(np.median(corr)),correlation_quartiles=np.quantile(corr,[.25,.75]).tolist())
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path); p.add_argument('run',type=Path)
    args=p.parse_args()
    print(json.dumps(summarize(args.data,args.run),indent=2))
