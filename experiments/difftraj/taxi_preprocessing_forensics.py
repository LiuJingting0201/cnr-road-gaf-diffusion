"""Read-only audit of original taxi-10k features against independent native traces."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree.ElementTree import parse

import numpy as np


def cells(x, low, high):
    ix = np.searchsorted(np.linspace(low[0], high[0], 17)[1:-1], x[..., 0], side='right')
    iy = np.searchsorted(np.linspace(low[1], high[1], 17)[1:-1], x[..., 1], side='right')
    return ix * 16 + iy


def interpolate(x, n):
    t = np.linspace(0, len(x)-1, n)
    return np.column_stack([np.interp(t, np.arange(len(x)), col) for col in x.T])


def length(x):
    return np.hypot(*np.diff(x, axis=0).T).sum()


def quantiles(x):
    return np.quantile(x, [.1,.5,.9,.99]).tolist()


def audit(root):
    fit = root/'taxi10k-base'
    pack = root/'difftraj-five-10k'
    with np.load(fit/'observations.npz') as f:
        obs = dict(f)
    with np.load(fit/'prepared.npz') as f:
        prepared = dict(f)
    xy = prepared['xy']
    hdr = prepared['h']
    splits = prepared['split']
    ids = obs['taxi']
    sizes = obs['length']
    offs = np.r_[0,sizes.cumsum()]
    assert np.array_equal(prepared['sizes'],sizes) and np.array_equal(splits,obs['episode_split'])
    assert len(set(ids))==len(ids)==12000
    assert obs['planar_unit_m']==.01
    idmap = {key:i for i,key in enumerate(ids)}
    episodes = defaultdict(list)
    for key in ids:
        episodes[key.split('_')[0]].append(key)
    numeric = np.zeros((len(ids),5))
    time_bins = np.zeros((len(ids),2),dtype=int)
    ratios = np.zeros((len(ids),2))
    observed_endpoints = np.zeros((len(ids),2,2))
    max_source_delta = max_resample_delta = 0.
    check = Counter()
    hashes = set()
    for episode, keys in episodes.items():
        with np.load(pack/episode/'trips.npz') as f:
            index = {key:j for j,key in enumerate(f['ids'])}
            h = f['h']
            orig_size = h[:,3].astype(int)
            pos = np.r_[0,orig_size.cumsum()]
            raw = f['raw']
        arrival = {v.get('id'): v for v in parse(pack/episode/'arrivals.xml').getroot().findall('tripinfo')}
        for key in keys:
            i,j = idmap[key],index[key]
            t,x,y,speed = raw[pos[j]:pos[j+1]].astype(float).T
            track = np.column_stack((x,y))
            stored = obs['delta'][offs[i]:offs[i+1]].cumsum(0)*.01
            max_source_delta = max(max_source_delta,np.abs(track-stored).max())
            assert len(stored)==sizes[i]==len(track)
            event = arrival[key]
            check['non_arrival_or_gap'] += int(float(event.get('depart'))!=t[0] or float(event.get('arrival'))!=t[-1]+1 or np.any(np.diff(t)!=1))
            check['negative_speed_or_nonfinite'] += int(np.min(speed)<0 or not np.isfinite(track).all() or not np.isfinite(speed).all())
            check['duration_mismatch'] += int(t[-1]-t[0]!=obs['duration'][i])
            check['under_120_native_points'] += int(len(t)<120)
            check['original_class_not_taxi'] += int('_taxi_' not in key)
            check['synthetic_start_mismatch'] += int(obs['start'][i] != t[0]+(int(episode)%1440)*60)
            time_bins[i] = [int(obs['start'][i]//300)%288, int(t[0]//300)%288]
            L = length(stored); duration=t[-1]-t[0]
            numeric[i]=[L,duration,len(t),L/(len(t)-1),L/duration]
            arr=interpolate(stored,200); restored=interpolate(arr,len(t))
            max_resample_delta=max(max_resample_delta,np.abs(xy[i]-arr).max())
            ratios[i]=[length(arr)/L,length(restored)/L]
            observed_endpoints[i]=stored[[0,-1]]
            digest=hashlib.sha256(np.ascontiguousarray(xy[i]).tobytes()).hexdigest()
            check['duplicate_200_point_geometry']+=int(digest in hashes)
            hashes.add(digest)
    train=splits==0
    nominal=(numeric-numeric[train].mean(0))/numeric[train].std(0)
    low,high=prepared['lo'],prepared['hi']
    endpoint_cells=cells(observed_endpoints,low,high)
    actual=hdr[:,6:].astype(int); od=actual[:,0]*256+actual[:,1]
    result=dict(spec='Native XY and elapsed time in metres and seconds; regular interval 1 s. Each count refers to selected episode-local taxi IDs.',check=dict(check),
                input_shape=list(xy.shape),max_component_rounding_m=float(max_source_delta),
                max_interpolated_component_error_m=float(max_resample_delta),
                independently_recomputed_header_max_abs_diff=np.max(np.abs(hdr[:,1:6]-nominal),axis=0).tolist(),
                endpoint_grid_independent_mismatches=int(np.sum(actual!=endpoint_cells)),
                header_clock_bin_mismatches=int(np.sum(time_bins[:,0]!=hdr[:,0])),
                training_stat_diff_mean=np.max(np.abs(xy[train].mean((0,1),dtype=float)-prepared['mean'])).item(),
                training_stat_diff_std=np.max(np.abs(xy[train].std((0,1),dtype=float)-prepared['std'])).item(),
                global_grid_lo=low.tolist(),global_grid_hi=high.tolist(),grid_cell_width_m=((high-low)/16).tolist(),splits={})
    episode_sets=[]
    for k,name in ((0,'train'),(8,'validation'),(9,'test')):
        mask=splits==k
        episode_sets.append(set(i.split('_')[0] for i in ids[mask]))
        counts=np.bincount(od[mask],minlength=65536)
        native_counts=np.array(sorted(set(time_bins[mask,1])))
        old_counts=np.unique(hdr[mask,0]).astype(int)
        result['splits'][name]=dict(n=int(mask.sum()),episodes=len(episode_sets[-1]),native_points_p10_p50_p90_p99=quantiles(sizes[mask]),
            native_km_p10_p50_p90_p99=quantiles(numeric[mask,0]/1000),
            trip_mean_mps_p10_p50_p90_p99=quantiles(numeric[mask,4]),
            retention_200_p10_p50_p90_p99=quantiles(ratios[mask,0]),retention_roundtrip_p10_p50_p90_p99=quantiles(ratios[mask,1]),
            current_clock_bins=old_counts.tolist(),local_simulation_clock_bins=native_counts.tolist(),current_time_bin_seen_train=float(np.isin(time_bins[mask,0],time_bins[train,0]).mean()),
            local_simulation_time_bin_seen_train=float(np.isin(time_bins[mask,1],time_bins[train,1]).mean()),
            distinct_OD=int((counts>0).sum()),largest_OD_share=float(counts.max()/counts.sum()),od_covered_by_training=float(np.isin(od[mask],od[train]).mean()),
            origin_covered_by_training=float(np.isin(actual[mask,0],actual[train,0]).mean()),dest_covered_by_training=float(np.isin(actual[mask,1],actual[train,1]).mean()))
    result['episode_split_intersection_sizes']=[len(episode_sets[a]&episode_sets[b]) for a,b in ((0,1),(0,2),(1,2))]
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('experiment_root',type=Path)
    print(json.dumps(audit(p.parse_args().experiment_root),indent=2))
