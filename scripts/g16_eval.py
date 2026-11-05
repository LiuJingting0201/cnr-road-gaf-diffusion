"""Independent shape/dynamics audit, using reference futures only after generation."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from cnr.episode_w1 import episode_w1
from cnr.topology import LaneConsistency
from g16_data import BASE, NAMES, ORIGINAL


def descriptors(x):
    speed = 4 * np.linalg.norm(np.diff(x.astype(np.float64),axis=-2),axis=-1)
    return [speed.sum(-1)/4,speed.mean(-1),abs(4*np.diff(speed,axis=-1)).mean(-1),(speed<.1).mean(-1)]


def audit(args):
    domain,arm,seed = args
    directory = BASE / f"{arm}-{seed}"
    sample = np.load(directory / (domain+".npz"))
    gen = sample["xy"]
    if domain != "cnr":
        prefix = "g14-" if domain=="g14" else ""
        req = np.load(BASE/(prefix+"requests.npz"))
        ref = np.load(BASE/(prefix+"references.npz"))
        assert np.array_equal(ref["ids"],req["ids"])
        lookup = {ident:i for i,ident in enumerate(req["ids"])}
        ix = np.array([lookup[k] for k in sample["ids"]])
        labels, episodes = req["labels"][ix], req["episodes"][ix]
        truth, lanes = ref["xy"][ix,None], req["lane"][ix]
    else:
        ref = np.load(ORIGINAL/"val.npz")
        assert np.array_equal(ref["source_index"],sample["ids"])
        labels, episodes = ref["labels"], np.arange(len(gen))
        truth, lanes = ref["xy"][:,None], np.full(len(gen),-1)
    graph = LaneConsistency("data/maps/most.net.xml",radius=5.,max_candidates=64)
    valid = np.empty(gen.shape[:2],bool)
    import shapely
    for i,label in enumerate(labels):
        for j,xy in enumerate(gen[i]):
            if label == 0:
                distances = shapely.distance(shapely.points(xy), graph.road.shapes[lanes[i]])
                valid[i,j] = bool(np.max(distances)<=2.01)
            else:
                result = graph.match(xy,NAMES[label])
                valid[i,j] = result["graph_consistent"] and not result["candidate_truncation_warning"]
    div = np.mean([np.linalg.norm(gen[:,a]-gen[:,b],axis=-1).mean(-1)
                   for a in range(4) for b in range(a)], axis=0)
    return (domain,arm,seed),dict(real=descriptors(truth), gen=descriptors(gen), labels=labels,
                                  episodes=episodes, valid=valid, caps=sample["caps"], div=div)


def summarize(rows,domain):
    common = range(1,6)
    summaries, curves = {}, {}
    for arm in "ABDC":
        per_seed = [rows[domain,arm,s] for s in (42,43)]
        first = per_seed[0]
        labs, ep = first["labels"],first["episodes"]
        number = len(np.unique(ep))
        e = np.unique(ep,return_inverse=True)[1]
        rng = np.random.default_rng(160320)
        counts = np.vstack([np.ones(number),rng.multinomial(number,np.ones(number)/number,2000)]) if domain!="cnr" else None
        by_class, estimates = {}, {}
        for label in np.unique(labs):
            mask = labs==label
            estimates[label] = []
            for metric in range(4):
                vals = []
                for data in per_seed:
                    assert np.array_equal(labs,data["labels"])
                    a, b = data["gen"][metric][mask],data["real"][metric][mask]
                    if domain!="cnr":
                        v = episode_w1(a,b,e[mask],metric==3,counts)
                    else:
                        v = np.array([abs(a.mean()-b.mean()) if metric==3 else wasserstein_distance(a.ravel(),b.ravel())])
                    vals.append(v)
                estimates[label].append(np.mean(vals,axis=0))
            by_class[NAMES[label]] = dict(conditions=int(mask.sum()),
                discrepancy=[float(v[0]) for v in estimates[label]],
                pass_count=int(sum(d["valid"][mask].sum() for d in per_seed)),
                total=int(sum(d["valid"][mask].size for d in per_seed)),
                caps=int(sum(d["caps"][mask].sum() for d in per_seed)),
                pairwise_distance_m=float(np.mean([d["div"][mask].mean() for d in per_seed])))
        macro = np.mean([estimates[c] for c in common],axis=0)
        curves[arm] = macro
        summaries[arm] = dict(five_macro=macro[:,0].tolist(), classes=by_class)
    contrasts = {}
    for advanced,base in [("B","A"),("D","A"),("B","D"),("C","B")]:
        delta = curves[advanced]-curves[base]
        contrasts[advanced+"-"+base] = dict(delta=delta[:,0].tolist())
        if domain!="cnr":
            contrasts[advanced+"-"+base]["interval"] = np.quantile(delta[:,1:],[.025,.975],axis=1).T.tolist()
    return dict(arms=summaries, contrasts=contrasts)


if __name__=="__main__":
    domains = ["g15","cnr"] + (["g14"] if (BASE/"g14-requests.npz").exists() else [])
    tasks = [(domain,arm,seed) for domain in domains for arm in "ABDC" for seed in (42,43)]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = dict(pool.map(audit,tasks))
    results = dict(metrics=["travel_W1_m","mean_speed_W1_mps","mean_abs_accel_W1_mps2","near_stop_abs_diff"],
                   **{domain:summarize(rows,domain) for domain in domains})
    (BASE/"results.json").write_text(json.dumps(results,indent=2))
