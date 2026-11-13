"""CPU post-selection scores using native traces, one training grid and an independent reference."""
import argparse
from pathlib import Path
import json

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from porto import grid, resample_trajectory

BASE = Path('/workspace/cnr-experiments')
METRICS = ('density_js','od_js','length_js','hotspot_f1','length_w1_km')


def load_trajectories(path, ids=None):
    with np.load(path) as raw:
        assert raw['planar_unit_m']==.01
        lens=raw['length']; ends=np.r_[0,lens.cumsum()]; delta=raw['delta']
        if ids is None:
            ids=np.arange(len(lens))
        tracks=[delta[ends[i]:ends[i+1]].cumsum(0)*.01 for i in ids]
        return tracks,raw['taxi'][ids]


def lengths(tracks):
    return np.array([np.linalg.norm(np.diff(t,axis=0),axis=1).sum() for t in tracks])


def features(tracks,lo,hi):
    cells=[grid(t,lo,hi) for t in tracks]
    occupancy=np.asarray([np.bincount(c,minlength=256) for c in cells],dtype=np.int32)
    od=np.asarray([c[0]*256+c[-1] for c in cells])
    return occupancy,od,lengths(tracks)


def hist(features,ids,bins):
    occ,od,metres=features
    x=metres[ids]
    return occ[ids].sum(0),np.bincount(od[ids],minlength=256**2),np.bincount(np.digitize(x,bins),minlength=1+len(bins)),x


def compare(a,b):
    js=[float(jensenshannon(x,y)**2) for x,y in zip(a[:3],b[:3])]
    top=lambda counts:set(np.argsort(-counts,kind='stable')[:20])
    return js+[len(top(a[0]) & top(b[0]))/20,float(wasserstein_distance(a[3],b[3])/1000)]


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('score_dirs',nargs='+',type=Path,help='Final evaluation directories containing draws.npz and observations.npz')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()

    with np.load(BASE/'taxi10k-base/prepared.npz') as p:
        lo,hi=p['lo'],p['hi']
        train=np.flatnonzero(p['split']<8)
    assert len(train)==10000
    training,_=load_trajectories(BASE/'taxi10k-base/observations.npz',train)
    bins=np.unique(np.quantile(lengths(training),np.linspace(0,1,21)[1:-1]))
    del training

    fresh,keys=load_trajectories(BASE/'taxi-debug-fresh-01/observations.npz')
    independent,ref_keys=load_trajectories(BASE/'taxi-debug-fresh-reference-01/observations.npz')
    assert len(keys)==1000 and len(ref_keys)==3000 and not set(keys)&set(ref_keys)
    lookup={k:t for k,t in zip(keys,fresh)}
    ref_f=features(independent,lo,hi)
    full=np.arange(1000)
    real_h=hist(features(fresh,lo,hi),full,bins)
    gen_h={}; paired={}
    for path in args.score_dirs:
        path=path.resolve()
        key=f'{path.parent.name}/{path.name}'
        assert key not in gen_h
        with np.load(path/'draws.npz') as draws:
            ids=draws['ids']; xy=draws['xy']
        actual,observed_ids=load_trajectories(path/'observations.npz',ids)
        assert len(observed_ids)==1000 and set(observed_ids)==set(keys)
        assert all(np.array_equal(t,lookup[k]) for k,t in zip(observed_ids,actual))
        assert xy.shape==(1000,200,2) and np.isfinite(xy).all()
        gen=[resample_trajectory(t,len(real)) for t,real in zip(xy,actual)]
        gen_h[key]=hist(features(gen,lo,hi),full,bins)
        paired[key]=dict(zip(METRICS,compare(real_h,gen_h[key])))

    repeats={'fixed_test_real_vs_independent_real':[], 'disjoint_independent_real_vs_independent_real':[]}
    repeats.update({key:[] for key in gen_h})
    for rep in range(50):
        perm=np.random.default_rng(13765+rep).permutation(3000)
        target=hist(ref_f,perm[:1000],bins)
        another=hist(ref_f,perm[1000:2000],bins)
        repeats['fixed_test_real_vs_independent_real'].append(compare(real_h,target))
        repeats['disjoint_independent_real_vs_independent_real'].append(compare(another,target))
        for key,histograms in gen_h.items():
            repeats[key].append(compare(histograms,target))
    result=dict(requests=1000, independent_population=3000, draws=50, grid_min_m=lo.tolist(), grid_max_m=hi.tolist(),length_bin_edges_m=bins.tolist(),
                model_vs_own_request_reference=paired, versus_independent_observed={})
    for label,rows in repeats.items():
        quantiles=np.quantile(rows,[.1,.5,.9],axis=0).T
        result['versus_independent_observed'][label]=dict(zip(METRICS,quantiles.tolist()))
    result['quantile_order']=['q10','q50','q90']
    result['interpretation']='Repeated reference subsets share observations. Quantiles measure sensitivity to the reference sample, not confidence intervals or training-seed uncertainty.'
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
