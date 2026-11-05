"""Pre-plan a legal directed route from user-supplied coordinates, without a future trace."""
import heapq
import numpy as np
import shapely
from shapely.ops import substring
from cnr.roads import allows
from g22_sim import VC,NODES


def from_endpoints(graph,c,start,end):
    label=["pedestrian","bus","motorcycle","private_car","bycicle","taxi_or_uber","truck","delivery"][c]
    candidates,_=graph.candidates(np.array([start,end],dtype=float),label)
    first,last=([t for t in xs if allows(graph.road.lanes[t[0]],VC[c])] for xs in candidates)
    if not first or not last:
        raise ValueError("Both endpoints must be within 2 m of permitted geometry.")
    best=None
    def consider(pieces):
        nonlocal best
        pts=[p for piece in pieces for p in piece.coords]
        if len(pts)<2:return
        line=shapely.LineString(pts)
        if line.length>=10 and (best is None or line.length<best.length):best=line
    for src,s,_ in first:
        for dst,e,_ in last:
            if src==dst and (c==0 or e>s):
                consider([substring(graph.road.shapes[src],s,e)])
        if c==0:continue
        costs={};parent={};queue=[]
        def extend(prev,near):
            for nxt in graph.successors[prev]:
                if nxt not in graph.permitted[label] or not allows(graph.road.lanes[nxt],VC[c]):continue
                gap=np.linalg.norm(np.subtract(graph.road.shapes[prev].coords[-1],graph.road.shapes[nxt].coords[0]))
                if gap>2:continue
                price=near+gap
                if price<costs.get(nxt,np.inf):
                    costs[nxt]=price
                    parent[nxt]=prev
                    heapq.heappush(queue,(price,nxt))
        extend(src,graph.lengths[src]-s)
        while queue:
            cost,node=heapq.heappop(queue)
            if cost!=costs[node]:continue
            if best is not None and cost>=best.length:break
            for dst,e,_ in last:
                if node!=dst:continue
                chain=[node]
                while parent[chain[-1]]!=src:chain.append(parent[chain[-1]])
                chain.reverse()
                consider([substring(graph.road.shapes[src],s,graph.lengths[src]),
                          *[graph.road.shapes[k] for k in chain[:-1]],substring(graph.road.shapes[dst],0,e)])
            if node!=src:extend(node,cost+graph.lengths[node])
    if best is None:raise ValueError("No supported directed path. Pedestrians currently require a shared sidewalk.")
    pts=shapely.get_coordinates(shapely.line_interpolate_point(best,np.linspace(0,best.length,NODES)))
    return pts,best.length
