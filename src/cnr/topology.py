"""Conservative vehicle-lane consistency diagnostic, not a SUMO validity oracle.

Use explicit connections, preserving via/internal lanes. Match multiple candidates
jointly so nearest-lane jitter is not itself labelled a graph violation. Same-edge
adjacent lane changes are a permissive approximation; traffic, signals, lane-change
safety and pedestrian walking-area routing are outside this diagnostic.
"""
from functools import lru_cache
import heapq
import numpy as np
import shapely
from shapely import STRtree
from cnr.roads import RoadNetwork, CLASS_TO_VCLASSES, allows


class LaneConsistency:
    def __init__(self,path,radius=5.,max_candidates=12,reverse_tolerance=.05,
                 route_factor=2.,route_slack=5.,route_cap=200.):
        self.road=RoadNetwork(path);self.radius=radius;self.max_candidates=max_candidates
        self.reverse_tolerance=reverse_tolerance;self.route_factor=route_factor
        self.route_slack=route_slack;self.route_cap=route_cap
        self.lengths=shapely.length(self.road.shapes)
        self.ids={lane['id']:i for i,lane in enumerate(self.road.lanes)}
        self.by_edge_index={(x['edge'],x.get('index','0')):i for i,x in enumerate(self.road.lanes)}
        self.successors=[set() for _ in self.road.lanes];self.unresolved_connections=0
        for c in self.road.connections:
            src=self.by_edge_index.get((c['from'],c['fromLane']))
            dst=self.ids.get(c['via']) if c.get('via') else self.by_edge_index.get((c['to'],c['toLane']))
            if src is None or dst is None:self.unresolved_connections+=1
            else:self.successors[src].add(dst)
        self.class_indices={};self.trees={};self.permitted={}
        for label,vc in CLASS_TO_VCLASSES.items():
            keep=np.array([i for i,x in enumerate(self.road.lanes) if any(allows(x,v) for v in vc)],dtype=int)
            self.class_indices[label]=keep;self.permitted[label]=set(keep.tolist())
            self.trees[label]=STRtree(self.road.shapes[keep])

    @lru_cache(maxsize=32768)
    def reachable_starts(self,src,label):
        """Distance from end(src) to start(dst), including intermediate lanes."""
        allowed=self.permitted[label]
        dist={n:0. for n in self.successors[src] if n in allowed}
        queue=[(0.,n) for n in dist];heapq.heapify(queue)
        while queue:
            d,node=heapq.heappop(queue)
            if d!=dist[node]:continue
            nd=d+float(self.lengths[node])
            if nd>self.route_cap:continue
            for nxt in self.successors[node]:
                if nxt in allowed and nd<dist.get(nxt,np.inf):
                    dist[nxt]=nd;heapq.heappush(queue,(nd,nxt))
        return dist

    def route_distance(self,src,s,dst,t,label):
        if label=='pedestrian':raise ValueError('Pedestrian walking-area routing is not implemented')
        options=[]
        if src==dst and t-s>=-self.reverse_tolerance:options.append(max(0.,t-s))
        a,b=self.road.lanes[src],self.road.lanes[dst]
        if src!=dst and a['edge']==b['edge'] and a['function']=='normal':
            di=int(b.get('index',0))-int(a.get('index',0))
            permission=a.get('changeLeft' if di>0 else 'changeRight')
            allowed_change=permission is None or any(v in permission.split() or 'all' in permission.split()
                                                     for v in CLASS_TO_VCLASSES[label])
            # Align relative progress for parallel lanes with unequal shape lengths.
            progress=t/max(self.lengths[dst],1e-9)*self.lengths[src]-s
            if abs(di)==1 and allowed_change and progress>=-self.reverse_tolerance:
                pa=shapely.line_interpolate_point(self.road.shapes[src],s)
                pb=shapely.line_interpolate_point(self.road.shapes[dst],t)
                options.append(float(shapely.distance(pa,pb)))
        middle=self.reachable_starts(src,label).get(dst,np.inf)
        options.append(float(self.lengths[src]-s+middle+t))
        return min(options)

    def candidates(self,xy,label):
        points=shapely.points(xy);tree=self.trees[label]
        pairs=tree.query(points,predicate='dwithin',distance=self.radius)
        groups=[[] for _ in xy];truncated=0
        if pairs.shape[1]:
            ix=self.class_indices[label][pairs[1]]
            distances=shapely.distance(points[pairs[0]],self.road.shapes[ix])
            positions=shapely.line_locate_point(self.road.shapes[ix],points[pairs[0]])
            for j,lane,d,s in zip(pairs[0],ix,distances,positions):groups[j].append((int(lane),float(s),float(d)))
        for j,g in enumerate(groups):
            g.sort(key=lambda x:(x[2],x[0]));truncated+=int(len(g)>self.max_candidates)
            groups[j]=g[:self.max_candidates]
        return groups,truncated

    def match(self,xy,label):
        if label=='pedestrian':return {'status':'unsupported_pedestrian','reason':'walking-area geometry and bidirectionality require a separate model'}
        xy=np.asarray(xy,dtype=np.float64)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():raise ValueError('Invalid trajectory')
        groups,truncated=self.candidates(xy,label)
        missing=sum(not g for g in groups);breaks=0;cost=None;prev=None;prev_xy=None
        # Lexicographic objective: minimum number of incompatible transitions,
        # then minimum accumulated squared geometric distance. No arbitrary penalty.
        for point,group in zip(xy,groups):
            if not group:
                if cost is not None:breaks+=min(cost)[0]
                cost=None;prev=None;prev_xy=None;continue
            if cost is None:cost=[(0,x[2]**2) for x in group]
            else:
                budget=min(self.route_cap,self.route_factor*np.linalg.norm(point-prev_xy)+self.route_slack)
                updated=[]
                for dst,t,emission in group:
                    choices=[]
                    for (src,s,_),(b,c) in zip(prev,cost):
                        valid=self.route_distance(src,s,dst,t,label)<=budget
                        choices.append((b+int(not valid),c+emission**2))
                    updated.append(min(choices))
                cost=updated
            prev=group;prev_xy=point
        if cost is not None:breaks+=min(cost)[0]
        covered=sum(bool(g) for g in groups)
        # This is graph consistency within a geometric envelope, not proof that
        # the original continuous curve follows lanes between observed points.
        return {'status':'audited','points':len(xy),'candidate_covered_points':covered,
            'generated_point_coverage':float(np.mean([bool(g) for g in groups[1:]])),
            'missing_candidate_points':missing,'truncated_candidate_points':truncated,
            'geometry_covered':missing==0,'minimum_incompatible_transitions':int(breaks),
            'graph_consistent':missing==0 and breaks==0,
            'candidate_truncation_warning':truncated>0}
