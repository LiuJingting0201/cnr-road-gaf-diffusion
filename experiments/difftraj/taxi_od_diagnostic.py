"""Audit saved DiffTraj draws against native samples on both fit-specific and shared grids."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from porto import grid, resample_trajectory, track_geometry


def pack_dir(path):
    for folder in (path, path.parent):
        if (folder/'observations.npz').exists() and (folder/'prepared.npz').exists():
            return folder
    raise FileNotFoundError(f'Missing native/prepared data near {path}')


def native_xy(raw, ids):
    scale, length = track_geometry(raw)
    delta = raw['delta']
    ends = np.r_[0, raw['length'].cumsum()]
    return [delta[ends[i]:ends[i+1]].cumsum(axis=0)*scale for i in ids], length


def measurement_basis(folder):
    with np.load(folder/'prepared.npz') as prep, np.load(folder/'observations.npz') as raw:
        ids = np.flatnonzero(prep['split'] < 8)
        points = prep['xy'][ids].reshape(-1,2)
        lo,hi = prep['lo'],prep['hi']
        core = np.quantile(points, [.01,.99], axis=0)
        tracks,length = native_xy(raw,ids)
        bins = np.unique(np.quantile([length(t) for t in tracks], np.arange(1,20)/20))
        return {'full':(lo,hi),'core':tuple(core)},bins


def diverg(a,b):
    return float(jensenshannon(a,b)**2)


def quant(a):
    return np.quantile(a,[.1,.5,.9]).tolist()


def occupancy(tracks,low,high,overflow):
    point_hist = np.zeros(257, dtype='i8')
    endpoints = []
    outside = 0
    for t in tracks:
        c = grid(t,low,high)
        outside_mask = ((t<low)|(t>high)).any(axis=1)
        outside += int(outside_mask.sum())
        if overflow: c[outside_mask]=256
        point_hist += np.bincount(c,minlength=257)
        endpoints.append(c[[0,-1]])
    return point_hist,np.asarray(endpoints),outside


def grid_report(real,gen,low,high,overflow):
    (r,rc,ro),(g,gc,go) = [occupancy(t,low,high,overflow) for t in [real,gen]]
    # Keep overflow in all divergence counts. On the city core exclude overflow from
    # request hits, so two physically distant out-of-bounds endpoints cannot match.
    M=257
    rpair=np.bincount(rc[:,0]*M+rc[:,1],minlength=M*M)
    gpair=np.bincount(gc[:,0]*M+gc[:,1],minlength=M*M)
    marg=[diverg(np.bincount(rc[:,j],minlength=M),np.bincount(gc[:,j],minlength=M)) for j in [0,1]]
    hit=(rc==gc)&(rc<256)
    steps=np.maximum(abs(rc//16-gc//16),abs(rc%16-gc%16))
    nearby=(steps<=1)&(rc<256)&(gc<256)
    top=lambda h: np.argsort(-h[:256],kind='stable')[:20]
    def structure(freq):
        k=len(rc)
        return {'distinct':int(np.count_nonzero(freq)), 'max_count':int(freq.max()),
                'pair_collision_probability':float(np.dot(freq,freq-1)/(k*(k-1))),
                'singleton_trips':int((freq==1).sum())}
    return dict(n=len(rc),lo=low.tolist(),hi=high.tolist(),
                density_js=diverg(r,g),od_js=diverg(rpair,gpair),marginal_origin_js=marg[0],marginal_destination_js=marg[1],
                top20_cell_f1=len(set(top(r))&set(top(g)))/20,
                real_nonempty_cells=int(np.count_nonzero(r[:256])),generated_nonempty_cells=int(np.count_nonzero(g[:256])),
                real_od=structure(rpair),generated_od=structure(gpair),
                generated_mass_on_observed_od=float(gpair[rpair>0].sum()/len(rc)),
                origin_hit=float(hit[:,0].mean()),destination_hit=float(hit[:,1].mean()),joint_hit=float(hit.all(1).mean()),
                origin_within_one_cell=float(nearby[:,0].mean()),destination_within_one_cell=float(nearby[:,1].mean()),joint_within_one_cell=float(nearby.all(1).mean()),
                outside_points_real=float(ro/r.sum()),outside_points_generated=float(go/g.sum()))


def audit(run,common,check):
    train=pack_dir(run); canonical=pack_dir(common)
    with np.load(train/'observations.npz') as raw, np.load(run/'draws.npz') as draws:
        ids=draws['ids']
        if len(np.unique(ids))!=len(ids): raise ValueError('Repeated evaluation index')
        real,length=native_xy(raw,ids)
        synth=[resample_trajectory(t,len(r)) for t,r in zip(draws['xy'],real,strict=True)]
    digest=hashlib.sha256()
    for t in real:
        digest.update(len(t).to_bytes(8,'little')); digest.update(t.astype('<f8').tobytes())
    rlen,glen=[np.array([length(t) for t in tracks]) for tracks in (real,synth)]
    errors=np.array([[length(np.stack((r[j],g[j]))) for j in (0,-1)] for r,g in zip(real,synth)])
    result=dict(n=len(ids),reference_sha256=digest.hexdigest(),q10_q50_q90_origin_error_m=quant(errors[:,0]),
                q10_q50_q90_destination_error_m=quant(errors[:,1]),length_w1_km=wasserstein_distance(rlen,glen)/1000,
                q10_q50_q90_real_length_km=quant(rlen/1000),q10_q50_q90_generated_length_km=quant(glen/1000))
    for name,folder in [('native',train),('common',canonical)]:
        if name=='common' and (folder/'prepared.npz').samefile(train/'prepared.npz'):
            result[name]=result['native']
            continue
        bounds,bins=measurement_basis(folder)
        h=lambda a:np.bincount(np.digitize(a,bins),minlength=len(bins)+1)
        stats=dict(length_js=diverg(h(rlen),h(glen)),bin_edges_m=bins.tolist())
        for area,(low,high) in bounds.items():
            stats[area]=grid_report(real,synth,low,high,area=='core')
        result[name]=stats
    if check is not None:
        old=json.loads(check.read_text())
        assert old['requests']==result['n']
        for local,field in [('density_js','density_js'),('od_js','trip_js'),('top20_cell_f1','top20_cell_f1')]:
            np.testing.assert_allclose(result['native']['full'][local],old[field],rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(result['native']['length_js'],old['length_js'],rtol=1e-9,atol=1e-12)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--common-pack',type=Path)
    parser.add_argument('--check',type=Path,help='Verify four native metrics against archived score file')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    item=audit(args.run,args.common_pack or args.run,args.check)
    (args.output or args.run/'od-diagnostic.json').write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
