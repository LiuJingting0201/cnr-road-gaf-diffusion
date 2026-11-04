"""Training-only route prior and explicit road-path reconstruction.

This pilot covers moving non-pedestrian windows. Lane-changing training windows are rejected in this pilot; traffic validity is not claimed.
"""
from collections import Counter,defaultdict
from functools import lru_cache
import heapq
import numpy as np
import shapely
from shapely.ops import substring
from cnr.topology import LaneConsistency


class RoutePaths(LaneConsistency):
    def trace(self,xy,label):
        xy=np.asarray(xy,dtype=np.float64);groups,truncated=self.candidates(xy,label)
        if truncated or any(not g for g in groups):return None
        costs=np.array([p[2]**2 for p in groups[0]]);back=[]
        for step,group in enumerate(groups[1:],1):
            budget=min(self.route_cap,self.route_factor*np.linalg.norm(xy[step]-xy[step-1])+self.route_slack)
            new=np.full(len(group),np.inf);parents=np.full(len(group),-1,dtype=int)
            for j,(dst,t,d) in enumerate(group):
                for i,(src,s,_) in enumerate(groups[step-1]):
                    if costs[i]<np.inf and self.route_distance(src,s,dst,t,label)<=budget:
                        value=costs[i]+d*d
                        if value<new[j]:new[j]=value;parents[j]=i
            if not np.isfinite(new).any():return None
            costs=new;back.append(parents)
        j=int(costs.argmin());trace=[groups[-1][j]]
        for step in range(len(groups)-2,-1,-1):
            j=int(back[step][j]);trace.append(groups[step][j])
        return trace[::-1]

    @lru_cache(maxsize=32768)
    def lane_path(self,src,dst,label):
        queue=[];distance={};parent={}
        for n in sorted(self.successors[src]):
            if n in self.permitted[label]:distance[n]=0.;parent[n]=None;heapq.heappush(queue,(0.,n))
        while queue:
            d,node=heapq.heappop(queue)
            if d!=distance[node]:continue
            if node==dst:
                path=[node]
                while parent[path[-1]] is not None:path.append(parent[path[-1]])
                return tuple([src]+path[::-1])
            nd=d+float(self.lengths[node])
            if nd>self.route_cap:continue
            for nxt in sorted(self.successors[node]):
                if nxt in self.permitted[label] and nd<distance.get(nxt,np.inf):
                    distance[nxt]=nd;parent[nxt]=node;heapq.heappush(queue,(nd,nxt))
        return None

    def reconstruct(self,trace,label):
        points=[];progress=[0.];transitions=[];distance=0.
        def add(coords):
            nonlocal distance
            for point in coords:
                point=np.asarray(point,dtype=np.float64)
                if points:
                    step=float(np.linalg.norm(point-points[-1]))
                    if step<1e-8:continue
                    distance+=step
                points.append(point)
        src,s,_=trace[0];add([shapely.line_interpolate_point(self.road.shapes[src],s).coords[0]])
        for dst,t,_ in trace[1:]:
            if src==dst and t-s>=-self.reverse_tolerance:
                t=max(t,s);piece=substring(self.road.shapes[src],s,t);add(piece.coords)
            elif (self.road.lanes[src]['edge']==self.road.lanes[dst]['edge']
                  and self.road.lanes[src]['function']=='normal'):
                return None  # First pilot uses explicit lane connections only, no lateral shortcut.
            else:
                path=self.lane_path(src,dst,label)
                if path is None:return None
                add(substring(self.road.shapes[src],s,self.lengths[src]).coords)
                for lane in path[1:-1]:add(self.road.shapes[lane].coords)
                add(substring(self.road.shapes[dst],0,t).coords)
                transitions.extend(zip(path[:-1],path[1:]))
            progress.append(distance);src,s=dst,t
        return {'points':np.array(points),'progress':np.array(progress),'last_lane':src,'last_position':s,
                'transitions':transitions,'first_lane':trace[0][0]}

    def extend(self,points,last_lane,last_position,label,counts,rng,budget=512.,max_hops=256):
        points=[np.asarray(p) for p in points]
        length=sum(np.linalg.norm(b-a) for a,b in zip(points[:-1],points[1:]))
        def add(coords):
            nonlocal length
            for point in coords:
                point=np.asarray(point);d=np.linalg.norm(point-points[-1])
                if d>1e-8:length+=d;points.append(point)
        lane=last_lane;position=last_position
        for _ in range(max_hops):
            if length>=budget:break
            add(substring(self.road.shapes[lane],position,self.lengths[lane]).coords)
            if length>=budget:break
            successors=sorted(n for n in self.successors[lane] if n in self.permitted[label])
            if not successors:break
            weights=np.array([counts.get((label,lane,n),0)+.1 for n in successors],dtype=np.float64)
            lane=int(rng.choice(successors,p=weights/weights.sum()));position=0.
        if len(points)<2:return None
        line=shapely.LineString(points)
        if line.length>budget:line=substring(line,0,budget)
        if line.length<1:return None
        distance=np.linspace(0,line.length,1025)
        route=shapely.get_coordinates(shapely.line_interpolate_point(line,distance)).astype(np.float32)
        return route,float(line.length)

    def sample(self,start,label,counts,rng,budget=512.):
        groups,truncated=self.candidates(np.asarray(start,dtype=np.float64)[None],label)
        if not groups[0] or truncated:return None
        # Nearest permitted candidate uses only given start and class, never heading/future.
        lane,s,_=groups[0][0];point=shapely.line_interpolate_point(self.road.shapes[lane],s).coords[0]
        return self.extend([point],lane,s,label,counts,rng,budget)


def decode_progress_numpy(route,length,progress,start):
    u=np.clip(np.asarray(progress)/length*(len(route)-1),0,len(route)-1)
    lo=np.minimum(u.astype(int),len(route)-2);w=(u-lo)[:,None]
    xy=route[lo]*(1-w)+route[lo+1]*w
    offset=np.asarray(start)-route[0]
    return xy+np.maximum(0,1-np.asarray(progress)/10)[:,None]*offset
