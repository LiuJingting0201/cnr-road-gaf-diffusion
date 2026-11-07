"""OD-only route proposals. No reference itinerary is accepted by propose()."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import networkx as nx
import numpy as np
from g32_routes import VehicleRouteNetwork,UnsupportedRoute,SCHEMA,PROVENANCE
from g33_evaluate import curve_xy,digest,save
from g33_model import CLASSES
FIELDS={'id','label','vclass','network_sha256','start','end','stops'}
def redact(r):
 return {k:copy.deepcopy(r[k]) for k in ['id','label','vclass','network_sha256','start','end']}|{'stops':[{k:v for k,v in s.items() if k in ('lane','pos','source_stop_index')} for s in r.get('stops',[])]}
def prepare(source,out):
 out.mkdir(parents=True,exist_ok=False);d=json.loads(source.read_text());rows=[]
 for c in range(5):
  for r in sorted([r for r in d['requests'] if r['label']==c],key=lambda r:r['id'])[:4]:
   try:rows.append(redact(r))
   except KeyError as e:rows.append({'id':r['id'],'label':c,'preparation_failure':str(e)})
 assert len(rows)==20
 save(out/'od.json',{'requests':rows,'source_manifest_sha256':digest(source),'redaction':'endpoints,class,ordered stops only; interior edge plan excluded'})
def graph(net,vc):
 g=nx.DiGraph();length={}
 for e in sorted(net.edges):
  if net.edges[e]['function']=='normal' and net.permitted(e,vc):
   length[e]=min(net.length(l) for l in net.permitted(e,vc));g.add_node(e)
 for e in list(g):
  targets=sorted({a.get('to') for l in net.permitted(e,vc) for _,a in net.successors[l] if a.get('to') in g})
  for target in targets:
   if any(net.transitions(e,target,vc,True).values()):g.add_edge(e,target,weight=length[target])
 return g,length

def edge_anchor(net,a):return net.lanes[a['lane']]['edge']
def bind_stops(net,edges,stops):
 bound=[];cursor=0;pos=-1
 for a in stops:
  edge=edge_anchor(net,a);ix=next((i for i in range(cursor,len(edges)) if edges[i]==edge and (i>cursor or a['pos']>=pos)),None)
  if ix is None:raise UnsupportedRoute('Cannot bind ordered public stop')
  bound.append(dict(a,edge_index=ix));cursor,pos=ix,a['pos']
 return bound

def paths(g,anchors,used,attempt,key):
 # Positive deterministic weights; route interiors and reference outcomes absent.
 def weight(u,v,attrs):
  jitter=int.from_bytes(hashlib.sha256(f'{key}:{attempt}:{u}:{v}'.encode()).digest()[:4],'little')/2**32
  return attrs['weight']*(1+used.get(v,0)*.6+(jitter*.15 if attempt else 0))
 route=[]
 for a,b in zip(anchors,anchors[1:]):
  segment=nx.shortest_path(g,a,b,weight=weight);route+=segment if not route else segment[1:]
 return route or anchors[:1]
def propose(net,g,length,r):
 if set(r)!=FIELDS:raise ValueError('OD input schema rejects route interiors / reference data')
 if r['network_sha256']!=net.network_sha256:raise ValueError('Network mismatch')
 anchors=[edge_anchor(net,r['start'])]+[edge_anchor(net,s) for s in r['stops']]+[edge_anchor(net,r['end'])]
 used={};seen=set();kept=[];attempts=[]
 for attempt in range(12):
  try:edges=paths(g,anchors,used,attempt,r['id'])
  except (nx.NetworkXNoPath,nx.NodeNotFound) as e:attempts.append({'attempt':attempt,'status':'no_graph_path','error':str(e)});break
  for e in set(edges):used[e]=used.get(e,0)+1
  if tuple(edges) in seen:attempts.append({'attempt':attempt,'status':'duplicate'});continue
  seen.add(tuple(edges))
  req={'schema':SCHEMA,'provenance':PROVENANCE,'frozen_before_simulation':True,'network_sha256':net.network_sha256,'vclass':r['vclass'],'edges':edges,'start':r['start'],'end':r['end'],'stops':bind_stops(net,edges,r['stops'])}
  try:
   geom=net.construct(req);m=geom.manifest();assert len(m['stop_progress_m'])==len(r['stops'])
   xy=np.asarray(geom.vertices);kept.append({'edges':edges,'length':float(geom.length_m),'vertices':xy.tolist(),'path':curve_xy(xy,np.linspace(0,1,129)).tolist(),'stop_count':len(m['stop_progress_m']),'lane_changes':geom.lane_changes});attempts.append({'attempt':attempt,'status':'valid','length':geom.length_m})
   if len(kept)==4:break
  except UnsupportedRoute as e:attempts.append({'attempt':attempt,'status':'geometry_unsupported','error':str(e)})
 kept.sort(key=lambda x:(x['length'],x['edges']))
 if kept:
  base=kept[0]['length'];retained=[v for v in kept if v['length']<=1.5*base]
 else:retained=[]
 for v in retained:v['detour_ratio']=v['length']/retained[0]['length']
 div=[]
 for i,a in enumerate(retained):
  for b in retained[i+1:]:
   x,y=set(a['edges']),set(b['edges']);div.append(1-sum(length[e] for e in x&y)/sum(length[e] for e in x|y))
 return {'id':r['id'],'label':r['label'],'routes':retained,'attempts':attempts,'detour_rejected':len(kept)-len(retained),'edge_pair_distance':div,'condition':'OD_plus_ordered_service_stops' if r['stops'] else 'OD','stops':len(r['stops'])}

def run(od,network,out):
 out.mkdir(parents=True,exist_ok=False);net=VehicleRouteNetwork(network);graphs={};rows=[];started=time.monotonic()
 try:
  for r in json.loads(od.read_text())['requests']:
   if 'preparation_failure' in r:row={'id':r['id'],'label':r['label'],'routes':[],'error':r['preparation_failure']}
   else:
    if r['vclass'] not in graphs:graphs[r['vclass']]=graph(net,r['vclass'])
    row=propose(net,*graphs[r['vclass']],r)
   rows.append(row);save(out/'progress.json',{'requests_processed':len(rows),'route_counts':[len(v['routes']) for v in rows],'seconds':time.monotonic()-started});print(json.dumps({'done':len(rows),'label':row['label'],'routes':len(row['routes'])}),flush=True)
  save(out/'routes.json',{'rows':rows,'od_sha256':digest(od),'network_sha256':net.network_sha256})
  save(out/'complete.json',{'complete':True,'denominator':20,'supported':sum(bool(v['routes']) for v in rows),'multiroute':sum(len(v['routes'])>1 for v in rows),'by_class':{c:{'requested':4,'supported':sum(bool(v['routes']) for v in rows if v['label']==i),'route_counts':[len(v['routes']) for v in rows if v['label']==i]} for i,c in enumerate(CLASSES)},'routes_sha256':digest(out/'routes.json'),'script_sha256':digest(__file__),'seconds':time.monotonic()-started,'reference_opened':False,'learned_route_choice':False})
 except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise

def check():
 g=nx.DiGraph();g.add_weighted_edges_from([('a','b',1),('b','d',1),('a','c',1.1),('c','d',1.1)])
 p=paths(g,['a','d'],{},0,'test');assert p==['a','b','d'];q=paths(g,['a','d'],{e:2 for e in p},1,'test');assert q==['a','c','d']
 assert paths(g,['a','b','d'],{},0,'test')==['a','b','d']
 try:paths(g,['d','a'],{},0,'test')
 except nx.NetworkXNoPath:pass
 else:raise AssertionError('direction ignored')
 r={'id':'x','label':0,'vclass':'bus','network_sha256':'h','start':{},'end':{},'stops':[{'lane':'a_0','pos':2,'edge_index':9,'source_stop_index':0}],'edges':['SECRET'],'duration':5}
 x=redact(r);assert set(x)==FIELDS and 'SECRET' not in str(x) and 'edge_index' not in x['stops'][0]
 try:propose(None,None,None,dict(x,edges=['secret']))
 except ValueError:pass
 else:raise AssertionError('route leakage accepted')
 print('checks passed: directed graph, alternatives, waypoint constraint, redaction, fail-closed schema')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['check','prepare','routes']);p.add_argument('--source',type=Path);p.add_argument('--network',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if a.action=='check':check()
 elif a.action=='prepare':prepare(a.source,a.output)
 else:run(a.source,a.network,a.output)
