"""Whole-trace Valhalla matches on unmodified input points, retaining failed requests."""
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np
from valhalla import Actor

PROFILES={'car':'auto','bus':'bus','taxi':'taxi','bike':'bicycle'}
CAP={'car':40.,'taxi':40.,'bus':35.,'bike':15.}

def init(config):
    global actor
    actor=Actor(str(config))

def match(job, *, with_interior=False):
    points,dt,mode=job
    n=len(points)
    row=dict(mode=mode,n=n,offsets=[],linked=0,reasonable=0,walk=0,pass08=0,passed=0,pass12=0)
    if with_interior:
        fraction = (np.arange(n-1) + 0.5) / (n-1)
        interior = (fraction >= 0.1) & (fraction <= 0.9)
        row.update(interior_intervals=int(interior.sum()), interior_linked=0, interior_walk=0)
    request=dict(shape=[dict(lon=float(x),lat=float(y)) for x,y in points],
                 costing=PROFILES[mode],shape_match='map_snap',units='kilometers',
                 trace_options=dict(gps_accuracy=10,search_radius=50),
                 filters=dict(action='include',attributes=['edge.length','matched.type','matched.edge_index','matched.distance_along_edge','matched.distance_from_trace_point','matched.begin_route_discontinuity','matched.end_route_discontinuity']))
    try:
        res=actor.trace_attributes(request)
    except RuntimeError as e:
        row['error']=str(e)[:160]
        return row
    edges=res['edges']; matches=res['matched_points']
    if len(matches)!=n: raise ValueError('Input point count changed')
    length=np.array([e['length']*1000 for e in edges])
    starts=np.r_[0.,np.cumsum(length)]
    pos=np.full(n,np.nan); breaks=np.zeros(n,dtype=bool)
    for i,m in enumerate(matches):
        if m.get('type')=='unmatched' or not 0 <= m.get('edge_index',-1) < len(edges): continue
        error=m['distance_from_trace_point']
        row['offsets'].append(error)
        if error>50: continue
        j=m['edge_index']; edge=edges[j]
        a=edge.get('source_percent_along',0.); b=edge.get('target_percent_along',1.)
        fraction=(m['distance_along_edge']-a)/(b-a) if b>a else 0.
        pos[i]=starts[j]+length[j]*np.clip(fraction,0,1)
        # Protect both adjacent intervals at a marked discontinuity.
        breaks[i]=bool(m.get('begin_route_discontinuity') or m.get('end_route_discontinuity'))
    dist=np.diff(pos)
    connected=np.isfinite(dist)&(dist>=-1)&~(breaks[:-1]|breaks[1:])
    speed=np.maximum(dist,0)/dt
    row['linked']=int(connected.sum())
    row['reasonable']=int(np.sum(connected&(speed<=CAP[mode])))
    row['walk']=int(connected.all())
    if with_interior:
        row['interior_linked'] = int(connected[interior].sum())
        row['interior_walk'] = int(connected[interior].all())
    for factor,key in [(0.8,'pass08'),(1.,'passed'),(1.2,'pass12')]:
        row[key]=int(np.all(connected&(speed<=factor*CAP[mode])))
    return row

def summarize(rows):
    errors=Counter(r.get('error') for r in rows if r.get('error'))
    offsets=np.array([x for r in rows for x in r['offsets']])
    totals={k:sum(r[k] for r in rows) for k in ['n','linked','reasonable','walk','pass08','passed','pass12']}
    return dict(trips=len(rows),intervals=totals['n']-len(rows),**totals,projected=len(offsets),
                projected_within50=int((offsets<=50).sum()),matched_offset_p50_p95=np.percentile(offsets,[50,95]).tolist() if offsets.size else None,errors=dict(errors))

if __name__=='__main__':
    name=sys.argv[1]; city=name.split('-')[0]
    d=np.load('tracks/'+name+'.npz')
    p=d['points']; ends=d['ends']
    jobs=[(p[a:b],float(dt),str(m)) for a,b,dt,m in zip(ends[:-1],ends[1:],d['dt'],d['mode'])]
    with ProcessPoolExecutor(int(sys.argv[2]),initializer=init,initargs=(Path(city+'.json').resolve(),)) as pool:
        rows=list(pool.map(match,jobs,chunksize=4))
    result=dict(all=summarize(rows),classes={m:summarize([r for r in rows if r['mode']==m]) for m in sorted(set(d['mode']))})
    Path('metrics').mkdir(exist_ok=True)
    np.savez('tracks/'+name+'-status.npz', passed=[r['passed'] for r in rows], walk=[r['walk'] for r in rows],
             reasonable=[r['reasonable'] for r in rows], intervals=[r['n']-1 for r in rows])
    Path('metrics',name+'.json').write_text(json.dumps(result,indent=2))
    print(name,result['all']['walk'],result['all']['passed'],flush=True)
