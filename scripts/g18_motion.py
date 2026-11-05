"""Audit temporal motion without reading either test split or changing a generator."""
import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

KEYS = ["cruise_s","braking_events","restarts","lag4_mps","stop_fraction"]


def measure(xy):
    xy = np.asarray(xy,dtype=float)
    assert xy.shape==(512,2) and np.isfinite(xy).all()
    step = np.linalg.norm(np.diff(xy,axis=0),axis=1)
    speeds = step[:508].reshape(127,4).sum(1)
    arc = np.r_[0,np.cumsum(step)]
    centers = np.arange(10,arc[-1]-10,5.)
    if not len(centers):
        return speeds,None
    keep = np.r_[True, np.diff(arc)>0]
    def point(s):
        return np.stack([np.interp(s,arc[keep],v[keep]) for v in xy.T],axis=1)
    before = point(centers)-point(centers-10)
    after = point(centers+10)-point(centers)
    norms = np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)
    angle = np.rad2deg(np.arccos(np.clip((before*after).sum(1)/np.maximum(norms,1e-12),-1,1)))
    # Local speed over five metres; includes time spent in any interior dwell.
    ticks = np.interp(np.r_[centers-2.5,centers+2.5],arc[keep],np.arange(512)[keep]*.25)
    elapsed = ticks[len(centers):]-ticks[:len(centers)]
    local = 5/np.maximum(elapsed,1e-9)
    bend = (angle>=10)&(norms>1)
    straight = (angle<=3)&(norms>1)
    gap = float(local[bend].mean()-local[straight].mean()) if bend.any() and straight.any() else None
    return speeds,gap


def read_tracks(bundle, name, wanted):
    tracks = defaultdict(list)
    with ZipFile(bundle) as z, z.open(name) as raw:
        rows = csv.reader(io.TextIOWrapper(raw))
        assert next(rows) == ["time","id","x","y","type"]
        for t, ident, x, y, _ in rows:
            if ident in wanted:
                tracks[ident].append((float(t),float(x),float(y)))
    assert set(tracks)==wanted
    out = {}
    for ident, track in tracks.items():
        arr = np.array(track)
        assert arr.shape==(512,3) and np.isfinite(arr).all() and np.allclose(np.diff(arr[:,0]),.25)
        out[ident] = measure(arr[:,1:])
    return out


def runs(mask):
    edges = np.diff(np.r_[0,mask.astype(int),0])
    return np.flatnonzero(edges==1),np.flatnonzero(edges==-1)


def describe(v, threshold):
    dv = np.diff(v)
    state = np.ones(len(dv),int)
    state[dv>=threshold] = 2
    state[dv<=-threshold] = 3
    state[v[1:]<.3] = 0
    a,b = runs(state==1)
    start,end = runs(state==3)
    lo,hi = runs(v<.3)
    restart = sum(e-l>=2 and e+2<=len(v) and np.all(v[e:e+2]>=.5) for l,e in zip(lo,hi))
    joint = np.bincount(4*state[:-1]+state[1:],minlength=16)/(len(state)-1)
    return np.r_[max(b-a,default=0), np.sum(end-start>=2), restart, np.mean(abs(v[4:]-v[:-4])),
                 np.mean(v<.3), joint]


def summarize(rows, episodes):
    if not rows:
        return {"requests":0}
    obs, syn = np.mean(np.array(rows),axis=0)
    cluster = np.array(episodes)
    delta = np.array([g[0]-r[0] for r,g in rows])
    unique = np.unique(cluster)
    draws = np.random.default_rng(1826).multinomial(len(unique),np.ones(len(unique))/len(unique),2000)
    sums = np.array([delta[cluster==e].sum() for e in unique])
    sizes = np.array([(cluster==e).sum() for e in unique])
    ci = np.quantile((draws@sums)/(draws@sizes),[.025,.975]).tolist()
    return dict(requests=len(rows),episodes=len(unique),cruise_difference_ci=ci,reference=dict(zip(KEYS,obs[:5])),synthetic=dict(zip(KEYS,syn[:5])),
                transition_TV=float(np.abs(obs[5:]-syn[5:]).sum()/2))


def audit(corpus, samples):
    with ZipFile(samples) as z:
        meta=json.loads(z.read("conditions.json"))
    generated=read_tracks(samples,"trajectories.csv",{r["id"] for r in meta})
    reference=read_tracks(corpus,"validation.csv",{r["request"] for r in meta})
    groups=defaultdict(list)
    for r in meta:
        groups[r["mode"],r["request"]].append(generated[r["id"]])
    result={}
    for threshold in (.3,.5,.7):
        tables=defaultdict(lambda:defaultdict(list))
        matched_counts=defaultdict(int)
        episodes=defaultdict(lambda:defaultdict(list))
        turns=defaultdict(list)
        for (mode,ident),pool in groups.items():
            assert len(pool)==8
            v,ref_turn=reference[ident]
            possible_turns = [g for _,g in pool if g is not None]
            pool = [vgen for vgen,_ in pool]
            if ref_turn is not None and possible_turns:
                turns[mode].append((ref_turn,np.mean(possible_turns),len(possible_turns)))
            ep=int(ident.split("_")[0])
            for group in ("all","shuffled"):
                episodes[mode][group].append(ep)
            ref=describe(v,threshold)
            feats=np.array([describe(x,threshold) for x in pool])
            tables[mode]["all"].append((ref,feats.mean(0)))
            seed=int.from_bytes(hashlib.sha256(ident.encode()).digest()[:4],"little")
            shuffled=np.random.default_rng(seed).permutation(v)
            assert np.array_equal(np.sort(v),np.sort(shuffled))
            tables[mode]["shuffled"].append((ref,describe(shuffled,threshold)))
            mask=np.abs(np.mean(pool,axis=1)-v.mean())<=1
            matched_counts[mode]+=int(mask.sum())
            if mask.any():
                episodes[mode]["matched"].append(ep)
                tables[mode]["matched"].append((ref,feats[mask].mean(0)))
        result[str(threshold)]={m:{k:summarize(table[k],episodes[m][k]) for k in ("all","matched","shuffled")}
                                | {"matched_candidates":matched_counts[m],"curve_vs_straight":{
                                    "paired_requests":len(turns[m]),
                                    "means_ref_gen_candidate_count":np.mean(turns[m],axis=0).tolist() if turns[m] else None}} for m,table in tables.items()}
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("corpus","samples","output"):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(audit(a.corpus,a.samples),indent=2,allow_nan=False))
