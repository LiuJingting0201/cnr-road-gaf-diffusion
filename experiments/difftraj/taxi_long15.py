"""Generate complete SUMO long taxi trips and match Porto training duration quartiles."""
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sumolib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BASE = Path('/workspace/cnr-experiments')
SOURCE = BASE / 'gaf-inloop-v1/g30-job/g30-data/source/scenario/in'
BINARY = BASE / 'gaf-inloop-v1/g10-env/sumo/bin/sumo'
PORTO = BASE / 'taxi-repro-debug-v1/porto/observations.npz'
WORKERS = int(os.getenv('LONG15_WORKERS', '12'))
EPISODE_CAP = 2400
TAXIS = 256
SEEDS = {0: 811000, 8: 821000, 9: 831000}
TARGETS = {0: 10000, 8: 1000, 9: 1000}
CLOCK = [0, .25, .5, .75, 1]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def setup():
    global edges_by_mode, bus_routes, taxi_edges, centers
    net = sumolib.net.readNet(str(SOURCE / 'most.net.xml'))
    edges_by_mode = {m: [e for e in net.getEdges() if e.allows(m) and e.getLength() > 25]
                     for m in ['motorcycle', 'passenger', 'bicycle']}
    bus_routes = ET.parse(SOURCE / 'route/most.buses.flows.xml').getroot().findall('route')
    for route in bus_routes:
        for stop in route.findall('stop'):
            stop.attrib.pop('until', None)
    edges = [e for e in net.getEdges() if e.allows('taxi')]
    index = {e: i for i, e in enumerate(edges)}
    links = [(i, index[b]) for i, a in enumerate(edges) for b, cs in a.getOutgoing().items()
             if b in index and any(c.getFromLane().allows('taxi') and c.getToLane().allows('taxi') for c in cs)]
    rows, cols = zip(*links)
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(edges), len(edges)))
    _, labels = connected_components(graph, directed=True, connection='strong')
    largest = np.bincount(labels).argmax()
    taxi_edges = [e for i, e in enumerate(edges) if labels[i] == largest and e.getLength() > 25]
    centers = np.array([e.getShape()[0] for e in taxi_edges])


def write_routes(seed, path):
    rng = np.random.default_rng(seed)
    root = ET.Element('routes')
    for mode in ['taxi', 'bus', *edges_by_mode]:
        ET.SubElement(root, 'vType', id=mode, vClass=mode)
    root.extend(bus_routes)
    demand = []
    for i in range(TAXIS):
        ident = f'{seed}_taxi_{i}'
        dep = int(rng.integers(900))
        chain = [int(rng.integers(len(taxi_edges)))]
        # Broad demand over multiple waypoints. Each vehicle follows one continuous SUMO trip.
        for _ in range(int(rng.integers(2, 13))):
            d = np.linalg.norm(centers - centers[chain[-1]], axis=1)
            options = np.flatnonzero((d >= 1200) & (d <= 5500))
            if len(options) == 0:
                options = np.flatnonzero(d > np.quantile(d, .65))
            chain.append(int(rng.choice(options)))
        route = [taxi_edges[j].getID() for j in chain]
        options = {'id': ident, 'type': 'taxi', 'depart': str(dep), 'departLane': 'best',
                   'from': route[0], 'via': ' '.join(route[1:-1]), 'to': route[-1]}
        demand.append((dep, ident, ET.Element('trip', options)))
    for mode in ['bus', *edges_by_mode]:
        for i in range(100):
            dep = int(rng.integers(900))
            ident = f'{seed}_{mode}_{i}'
            common = dict(id=ident, type=mode, depart=str(dep))
            if mode == 'bus':
                common['route'] = rng.choice([r.get('id') for r in bus_routes])
                vehicle = ET.Element('vehicle', common)
            else:
                edges = edges_by_mode[mode]
                while True:
                    a, b = (edges[j] for j in rng.choice(len(edges), 2, replace=False))
                    if 1100 < np.linalg.norm(np.subtract(a.getShape()[0], b.getShape()[-1])) < 6500:
                        break
                common.update({'from': a.getID(), 'to': b.getID(), 'departLane': 'best'})
                vehicle = ET.Element('trip', common)
            demand.append((dep, ident, vehicle))
    root.extend(v for _, _, v in sorted(demand))
    ET.ElementTree(root).write(path)


def episode(seed):
    cache = Path('episodes') / f'{seed}.npz'
    if cache.exists():
        return seed
    with tempfile.TemporaryDirectory(dir='episodes') as name:
        wd = Path(name).resolve()
        write_routes(seed, wd / 'routes.xml')
        args = [str(BINARY), '-n', str(SOURCE / 'most.net.xml'), '-r', 'routes.xml', '-a',
                str(SOURCE / 'add/most.busstops.add.xml'), '--seed', str(seed), '--step-length', '1',
                '--end', '16000', '--time-to-teleport', '-1', '--ignore-route-errors', 'true',
                '--device.fcd.period', '15', '--fcd-output', 'fcd.xml.gz', '--fcd-output.attributes', 'x,y',
                '--tripinfo-output', 'arrivals.xml', '--no-step-log', 'true']
        with (wd / 'sumo.log').open('w') as log:
            result = subprocess.run(args, cwd=wd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            raise RuntimeError((seed, (wd / 'sumo.log').read_text()[-3000:]))
        arrivals = {t.get('id'): (float(t.get('depart')), float(t.get('arrival')))
                    for t in ET.parse(wd / 'arrivals.xml').getroot().findall('tripinfo')
                    if '_taxi_' in t.get('id') and t.get('vaporized', '') in ('', 'false', '0')}
        obs = defaultdict(list)
        with gzip.open(wd / 'fcd.xml.gz') as file:
            for _, node in ET.iterparse(file, events=['end']):
                if node.tag == 'timestep':
                    t = int(float(node.get('time')))
                    for v in node:
                        ident = v.get('id')
                        if ident in arrivals:
                            obs[ident].append((t, float(v.get('x')), float(v.get('y'))))
                    node.clear()
        ids, sizes, starts, deltas, margins = [], [], [], [], []
        rejects = Counter()
        for ident in sorted(arrivals):
            arr = np.asarray(obs[ident]); dep, end = arrivals[ident]
            if len(arr) < 120:
                rejects['under_120'] += 1
                continue
            t, xy = arr[:, 0], arr[:, 1:]
            if not (0 <= t[0] - dep < 15 and 0 < end - t[-1] <= 15 and np.all(np.diff(t) == 15)):
                rejects['incomplete_clock'] += 1
                continue
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            if not np.isfinite(xy).all() or steps.max() > 65 * 15 or steps.sum() < 100:
                rejects['invalid_geometry'] += 1
                continue
            cm = np.rint(xy * 100).astype('int32')
            deltas.append(np.diff(cm, axis=0, prepend=np.zeros((1, 2), 'i4')))
            sizes.append(len(t)); starts.append(int(t[0])); ids.append(ident)
            margins.append([int(t[0]-dep), int(end-t[-1])])
        np.savez_compressed(wd/'episode.npz', taxi=ids, length=sizes, start=starts,
                            delta=np.concatenate(deltas) if deltas else np.empty((0, 2), 'i4'), margins=margins)
        (wd/'episode.npz').replace(cache)
        write_json(cache.with_suffix('.json'), dict(seed=seed, requested=TAXIS, arrived=len(arrivals),
                                                   eligible=len(ids), rejected=dict(rejects)))
    return seed


def main():
    started = time.monotonic()
    with np.load(PORTO) as reference:
        lens = reference['length'][reference['episode_split'] < 8]
    duration = (lens-1)*15
    bounds = np.quantile(duration, [.25,.5,.75])
    porto_bins = np.bincount(np.searchsorted(bounds, duration, side='right'), minlength=4)
    summary = dict(eligible_observations='15s, >=120 points, completed taxi vehicles; up to 14 s before first and 15 s after last GPS read',
                   selection='training Porto quartile cutoffs and proportions; independent SUMO seed blocks',
                   porto_training_quartile_edges_seconds=bounds.tolist(), porto_counts=porto_bins.tolist(),
                   episode_cap_per_split=EPISODE_CAP, workers=WORKERS, per_episode_taxi_demand=TAXIS, splits={})
    Path('episodes').mkdir(exist_ok=True)
    setup()
    selected = []; seen = set()
    with ProcessPoolExecutor(WORKERS) as pool:
        for split, target in TARGETS.items():
            expected = target*porto_bins/porto_bins.sum()
            quotas = np.floor(expected).astype(int)
            for index in np.argsort(-(expected-quotas))[:target-quotas.sum()]:
                quotas[index] += 1
            counts = np.zeros(4, int)
            stats = Counter(); all_sizes=[]
            for offset in range(0, EPISODE_CAP, WORKERS):
                batch = [SEEDS[split]+n for n in range(offset, min(offset+WORKERS, EPISODE_CAP))]
                for seed in pool.map(episode, batch):
                    path = Path('episodes') / f'{seed}.npz'
                    meta=json.loads(path.with_suffix('.json').read_text()); stats.update({k:meta[k] for k in ['requested','arrived','eligible']})
                    with np.load(path) as raw:
                        sizes=raw['length']; delta=raw['delta']; idx=np.r_[0,sizes.cumsum()].astype(int)
                        taxis=raw['taxi']; starts=raw['start']; margins=raw['margins']
                        for i,n in enumerate(sizes):
                            key=int(np.searchsorted(bounds,(n-1)*15,side='right'))
                            if counts[key] == quotas[key]:
                                continue
                            d=delta[idx[i]:idx[i+1]].copy(); digest=hashlib.sha256(d.tobytes()).digest()
                            if digest in seen:
                                continue
                            seen.add(digest); counts[key]+=1; all_sizes.append(n)
                            selected.append((split, taxis[i], int(starts[i]), int(n), d, margins[i].copy()))
                    summary['splits'][split] = dict(target=target, selected=int(counts.sum()), bins=counts.tolist(),
                        quotas=quotas.tolist(), observed_duration_quantiles_seconds=np.quantile((np.array(all_sizes)-1)*15,CLOCK).tolist() if all_sizes else [],
                        last_seed=seed, totals=dict(stats))
                    summary['elapsed_seconds']=round(time.monotonic()-started,2)
                    write_json('progress.json',summary)
                if np.array_equal(counts,quotas):
                    break
            else:
                raise RuntimeError(f'Bounded episode budget exhausted: {counts} of {quotas} for split {split}')
    split, ids, starts, sizes, delta, margins = zip(*selected)
    n=np.asarray(sizes); margin=np.asarray(margins)
    assert len(set(ids))==len(ids)==sum(TARGETS.values()) and n.min()>=120
    summary['max_seconds_between_trip_and_first_last_observation']=margin.max(axis=0).tolist()
    summary['elapsed_seconds']=round(time.monotonic()-started,2)
    np.savez_compressed('observations.tmp.npz',delta=np.concatenate(delta),length=n,taxi=ids,start=starts,duration=(n-1)*15,
                        episode_split=np.array(split,'i1'),planar_unit_m=.01,timezone='UTC')
    Path('observations.tmp.npz').replace('observations.npz')
    write_json('packing-summary.json', summary)


if __name__=='__main__':
    main()
