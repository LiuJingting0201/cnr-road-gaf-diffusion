"""Re-score saved Porto draws on one training-only grid and length scale."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from porto import arc_length, grid, resample_trajectory


def features(tracks, lo, hi, core):
    hist, od = [], []
    for track in tracks:
        cells = grid(track, lo, hi)
        if core:
            cells[np.any((track < lo) | (track > hi), axis=1)] = 256
        hist.append(np.bincount(cells, minlength=257))
        od.append(cells[0] * 257 + cells[-1])
    return np.asarray(hist), np.asarray(od)


def compare(a, b, selection, other, bins):
    ha, oa, la = a
    hb, ob, lb = b
    h1, h2 = ha[selection].sum(0), hb[other].sum(0)
    rank = lambda h: set(np.argsort(-h[:256], kind='stable')[:20])
    js = lambda x, y: float(jensenshannon(x, y)**2)
    lengths = lambda arr: np.bincount(np.digitize(arr, bins), minlength=len(bins)+1)
    x, y = la[selection], lb[other]
    return dict(density=js(h1,h2), trip=js(np.bincount(oa[selection],minlength=257**2),
                                          np.bincount(ob[other],minlength=257**2)),
                length=js(lengths(x),lengths(y)), pattern=len(rank(h1)&rank(h2))/20,
                length_w1_km=wasserstein_distance(x,y)/1000)


def audit(run, root):
    canonical = np.load(root/'42-mixed-on-mixed/prepared.npz')
    train = canonical['xy'][canonical['split']<8].reshape(-1,2)
    ranges = {'full':(canonical['lo'],canonical['hi']), 'core':np.quantile(train,[.01,.99],axis=0)}
    bins = np.load(root/'shared-bins.npz')['mixed']
    raw = np.load(run/'observations.npz')
    draws = np.load(run/'draws.npz')
    ids = draws['ids']
    ends = np.r_[0,raw['length'].cumsum()]
    delta = raw['delta']
    real = [delta[ends[i]:ends[i+1]].cumsum(0)*1e-6 for i in ids]
    syn = [resample_trajectory(track,len(ref)) for track,ref in zip(draws['xy'],real)]
    digest = hashlib.sha256()
    for track in real:
        digest.update(len(track).to_bytes(8,'little'))
        digest.update(track.astype('<f8').tobytes())
    assert len(real)==len(syn)==3000
    lengths = [np.array([arc_length(t) for t in tracks]) for tracks in (real,syn)]
    archive = json.loads((run/'porto-final.json').read_text())
    assert np.isclose(np.median(lengths[0]),archive['reference_median_m'])
    assert np.isclose(np.median(lengths[1]),archive['generated_median_m'])
    if run.name.startswith('42-'):
        old_dir = Path('/workspace/cnr-experiments/porto-stabilized-v1') / run.name[3:]
        old = json.loads((old_dir/'porto-final.json').read_text())
        for field in ('density_js','trip_js','length_js','top20_cell_f1','request_index_sha256'):
            if old[field] != archive[field]:
                raise ValueError(f'Fixed-weight replay mismatch: {field}')
    # Project local coordinate differences to metres using a single training latitude.
    scale = np.array([np.cos(np.radians(np.median(train[:,1]))),1.]) * (6371000*np.pi/180)
    euclid = lambda t: np.linalg.norm(np.diff(t,axis=0)*scale,axis=1).sum()
    planar = [np.array([euclid(t) for t in tracks]) for tracks in (real,syn)]
    hist = lambda a: np.bincount(np.digitize(a,bins),minlength=len(bins)+1)
    result = dict(requests=len(ids),ref_sha256=digest.hexdigest(), archived=archive,grids={})
    result['projected_length_js_same_metre_bins'] = float(jensenshannon(*map(hist,planar))**2)
    fig = {}
    for key,(lo,hi) in ranges.items():
        a,b = [(*features(tracks,lo,hi,key=='core'),l) for tracks,l in zip((real,syn),lengths)]
        both = np.arange(3000)
        stats = {'all':compare(a,b,both,both,bins)}
        controls = {'real_real':[], 'real_generated':[]}
        for seed in range(20):
            x,y = np.split(np.random.default_rng(2300+seed).permutation(both),2)
            controls['real_real'].append(compare(a,a,x,y,bins))
            controls['real_generated'].append(compare(a,b,x,x,bins))
        for name,rows in controls.items():
            stats[name] = {k:np.quantile([r[k] for r in rows],[.1,.5,.9]).tolist() for k in rows[0]}
        stats['occupied'] = [int((v[0].sum(0)[:256]>0).sum()) for v in (a,b)]
        result['grids'][key]=stats
        if key=='core':
            fig['bounds'] = np.array([lo,hi]).tolist()
            fig['density'] = [(v[0].sum(0)/v[0].sum()).tolist() for v in (a,b)]
    fig['km_quantiles']=[(np.quantile(l,np.arange(101)/100)/1000).tolist() for l in lengths]
    result['plot']=fig
    (run/'audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    path=Path(sys.argv[1]).resolve()
    audit(path,path.parent)
