"""Audit already sampled SUMO trajectories on the taxi-permitted directed MoST graph."""
import argparse
import csv
import hashlib
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from cnr.roads import allows

RADIUS = 50.
MAX_CANDIDATES = 16
MAX_ROUTE = 1500.
SPEED = 40.


class TaxiNetwork:
    def __init__(self, path, distance_file):
        xml = ElementTree.parse(path).getroot()
        edge_index = {}
        lane_edge = {}
        geometry = []
        geom_edge = []
        edge_m = []
        for edge in xml.findall('edge'):
            allowed = []
            for lane in edge.findall('lane'):
                coords = [tuple(map(float, pair.split(',')[:2])) for pair in lane.get('shape', '').split()]
                if allows(lane.attrib, 'taxi') and len(coords) >= 2:
                    line = shapely.LineString(coords)
                    if line.length > 0:
                        allowed.append((lane, line))
            if not allowed:
                continue
            ei = len(edge_m)
            edge_index[edge.get('id')] = ei
            edge_m.append(float(np.mean([float(lane.get('length')) for lane, _ in allowed])))
            for lane, line in allowed:
                lane_edge[lane.get('id')] = ei
                geometry.append(line)
                geom_edge.append(ei)
        self.length = np.asarray(edge_m, dtype=float)
        self.shapes = np.asarray(geometry, dtype=object)
        self.tree = shapely.STRtree(self.shapes)
        self.scale = self.length[geom_edge] / shapely.length(self.shapes)
        self.edge = np.asarray(geom_edge)
        transitions = set()
        for con in xml.findall('connection'):
            source_lane = con.get('from') + '_' + con.get('fromLane')
            dest_lane = con.get('to') + '_' + con.get('toLane')
            inter_lane = con.get('via', dest_lane)
            if all(lane in lane_edge for lane in (source_lane, inter_lane, dest_lane)) and allows(con.attrib, 'taxi'):
                transitions.add((lane_edge[source_lane], lane_edge[inter_lane]))
        src, dst = np.asarray(sorted(transitions)).T
        self.info = dict(edges=len(edge_m), taxi_lane_shapes=len(self.shapes), directed_transitions=len(src),
                         map_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        graph = csr_matrix((self.length[src], (src,dst)), shape=(len(edge_m),len(edge_m)))
        self.mid = np.memmap(distance_file, dtype='float32', mode='w+', shape=graph.shape)
        for lo in range(0, len(edge_m), 96):
            ids = np.arange(lo, min(len(edge_m),lo+96))
            # Dijkstra from edge starts; subtract the source length for end-to-start distance.
            dist = dijkstra(graph, indices=ids, limit=MAX_ROUTE + self.length.max())
            self.mid[ids] = dist - self.length[ids, None]
        self.mid.flush()

    def candidates(self, xy):
        points = shapely.points(xy)
        ix, lane = self.tree.query(points, predicate='dwithin', distance=RADIUS)
        dist = shapely.distance(self.shapes[lane], points[ix])
        along = shapely.line_locate_point(self.shapes[lane], points[ix]) * self.scale[lane]
        groups = [[] for _ in points]
        seen = [set() for _ in points]
        truncated = set()
        for k in np.lexsort((lane,dist,ix)):
            p = ix[k]
            edge = self.edge[lane[k]]
            if edge in seen[p]:
                continue
            seen[p].add(edge)
            if len(groups[p]) < MAX_CANDIDATES:
                groups[p].append((edge,along[k],dist[k]))
            else:
                truncated.add(p)
        return groups,len(truncated)

    def match(self, xy, dt):
        groups,truncated = self.candidates(xy)
        options = [np.array(group if group else [(-1,0.,RADIUS+1)]) for group in groups]
        missing = np.array([not g for g in groups])
        previous_cost = options[0][:,2]**2
        failures = np.zeros(len(previous_cost), dtype=int)
        trail = []
        for i,(a,b) in enumerate(zip(options,options[1:])):
            ae,be = a[:,0].astype(int), b[:,0].astype(int)
            same = (ae[:,None] == be[None,:])
            advance = b[None,:,1]-a[:,None,1]
            route = self.length[ae,None]-a[:,None,1]+self.mid[ae[:,None],be[None,:]]+b[None,:,1]
            # Dijkstra assigns zero start-to-start cost to identical edges. A backward move
            # on an edge needs an actual cycle. Conservatively reject that alternative.
            route = np.where(same, np.where(advance >= -1, np.maximum(0,advance), np.inf), route)
            chord = np.linalg.norm(xy[i+1]-xy[i])
            valid = (route <= min(MAX_ROUTE, 3*chord+100)) & ~(missing[i] | missing[i+1])
            new_fail = failures[:,None] + (~valid)
            min_fail = new_fail.min(axis=0)
            distance_cost = np.where(valid, (np.minimum(abs(route-chord),500)/5)**2, 0)
            cost = previous_cost[:,None] + distance_cost
            prev = np.argmin(np.where(new_fail == min_fail, cost, np.inf), axis=0)
            k = np.arange(len(b))
            previous_cost = cost[prev,k] + b[:,2]**2
            failures = min_fail
            trail.append((prev,valid[prev,k],route[prev,k]))
        current = np.argmin(np.where(failures == failures.min(), previous_cost, np.inf))
        linked = np.zeros(len(dt),bool)
        speeds = np.full(len(dt), np.inf)
        for i in reversed(range(len(dt))):
            prev,ok,length = trail[i]
            linked[i] = ok[current]
            speeds[i] = max(0,length[current])/dt[i]
            current = prev[current]
        plausible = linked & (speeds<=SPEED)
        midtimes = np.cumsum(dt) - .5*dt
        central = (midtimes >= .1*dt.sum()) & (midtimes <= .9*dt.sum())
        return dict(intervals=len(dt),linked=int(linked.sum()),reasonable=int(plausible.sum()),
                    walk=int(linked.all()),passed=int(plausible.all()),observations=len(xy),
                    central_intervals=int(central.sum()),central_linked=int(linked[central].sum()),
                    central_walk=int(linked[central].all()),central_reasonable=int(plausible[central].sum()),
                    projected_within50=int((~missing).sum()),capped_points=truncated)


def interp(points, times, duration):
    source = np.linspace(0, duration, len(points))
    return np.column_stack([np.interp(times, source, a) for a in points.T])


def match_case(case):
    row_index,order,label,points,times,dense = case
    result = net.match(points, np.diff(times))
    dense_pts=shapely.points(dense[1:-1])
    nearest=net.tree.nearest(dense_pts)
    offsets=shapely.distance(net.shapes[nearest],dense_pts)
    return dict(row_index=int(row_index),request_order=int(order),arm=label,**result,
                interior_points=len(offsets),interior_within10=int((offsets<=10).sum()),
                interior_within50=int((offsets<=50).sum()),interior_all50=int((offsets<=50).all()))


def jobs(arm):
    with np.load(arm/'observations.npz') as raw:
        offsets=np.r_[0,raw['length'].cumsum()]
        delta=raw['delta']; duration=raw['duration']; scale=float(raw['planar_unit_m'])
        split=raw['episode_split']
    for seed in [42,43]:
        with np.load(arm/str(seed)/'draws.npz') as draws:
            ids=draws['ids']
            synthetic=draws['xy']
        assert len(ids)==len(set(ids))==1000 and np.all(split[ids]==8)
        if seed==42:
            first_ids=ids
        else:
            np.testing.assert_array_equal(ids,first_ids)
        for j,k in enumerate(ids):
            xy=delta[offsets[k]:offsets[k+1]].cumsum(0)*scale
            T=int(duration[k]); assert T+1==len(xy)
            times=np.r_[np.arange(0,T,15),T]
            if seed==42:
                yield k,j,'real',xy[times],times,interp(xy,np.linspace(0,T,200),T)
            pred=synthetic[j]
            yield k,j,str(seed),interp(pred,times,T),times,pred


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('network',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--max-candidates',type=int,default=16)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    global net, MAX_CANDIDATES
    MAX_CANDIDATES = args.max_candidates
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        net=TaxiNetwork(args.network, Path(temp)/'traversal.dat')
        print(net.info,flush=True)
        cases=list(jobs(args.source))
        with ProcessPoolExecutor(args.workers,mp_context=get_context('fork')) as pool:
            rows=list(pool.map(match_case,cases,chunksize=8))
    totals={}
    for label in ('real','42','43'):
        arm=[r for r in rows if r['arm']==label]
        assert len(arm)==1000
        totals[label]=dict(trips=len(arm),**{k:sum(r[k] for r in arm) for k in arm[0] if k not in ('row_index','request_order','arm')})
    with (args.out/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    params=dict(network=net.info, observation_period_seconds=15,retain_terminal=True, radius_m=RADIUS,
                maximum_candidate_edges_per_point=MAX_CANDIDATES, speed_ceiling_mps=SPEED,
                route_limit='min(1500 m, 3 times observed chord + 100 m)', reverse_tolerance_m=1.,
                whole_trip='all intervals supported by one candidate sequence; same-edge lane positioning and choice are approximated at edge level',
                central_intervals='actual elapsed interval midpoint in inclusive [10%,90%] of duration',
                density_diagnostic='198 interior points of the common representation (observed xy interpolated to 200 points, generated original 200) to nearest taxi-permitted lane')
    result=dict(parameters=params,counts=totals)
    (args.out/'summary.tmp').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.out/'summary.tmp').replace(args.out/'summary.json')
    print(totals,flush=True)


if __name__=='__main__':
    main()
