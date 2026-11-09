"""Network-only OD catalogs and matched route-policy/GAF candidate selection."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from g33_evaluate import digest,save,curve_xy
from g33_model import MotionDenoiser,CLASSES,sample
from g37_routes import redact,propose,graph
from g32_routes import VehicleRouteNetwork,UnsupportedRoute
from g38_policy import RoadFeatures,load_policy,route_logprob
from g36_feedback import load_reader,spatial_sequence
from g23_modes import speed_series
from g28_teachers import image_inputs
from cnr.original_gaf import encode_frozen

METHODS=['Uniform','Shortest','Linear','R','S','V','SV','R+S','R+V','R+SV']


def catalog(prepared,network,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        receipt=json.loads((prepared/'prepare-complete.json').read_text())
        assert receipt['complete'] and digest(prepared/'requests.json')==receipt['requests_sha256']
        manifest=json.loads((prepared/'requests.json').read_text())
        rows=[];ods=[];net=VehicleRouteNetwork(network);graphs={}
        for request in manifest['requests']:
            try:
                r=redact(request);ods.append(r)
                if r['vclass'] not in graphs:graphs[r['vclass']]=graph(net,r['vclass'])
                row=propose(net,*graphs[r['vclass']],r)
            except (KeyError,ValueError) as e:
                row={'id':request['id'],'label':request['label'],'routes':[],'error':str(e)}
            rows.append(row)
            save(out/'progress.json',{'done':len(rows),'supported':sum(bool(r['routes']) for r in rows),'seconds':time.monotonic()-start})
        save(out/'od.json',{'requests':ods})
        save(out/'routes.json',{'rows':rows,'network_sha256':digest(network),'od_sha256':digest(out/'od.json')})
        save(out/'complete.json',{'complete':True,'requested':len(rows),'supported':sum(bool(r['routes']) for r in rows),
            'by_class':{c:{'requested':sum(r['label']==i for r in rows),'counts':[len(r['routes']) for r in rows if r['label']==i]} for i,c in enumerate(CLASSES)},
            'seconds':time.monotonic()-start,'routes_sha256':digest(out/'routes.json'),'od_sha256':digest(out/'od.json'),
            'request_manifest_sha256':digest(prepared/'requests.json'),
            'reference_opened':False,'script_sha256':digest(__file__)})
    except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise


def probability(logits):
    logits=np.asarray(logits,float);assert np.isfinite(logits).any()
    x=np.exp(logits-np.max(logits));return x/x.sum()


def selections(route_index,linear,neural,s,v,key,valid=True):
    """Return all weights plus coupled draws. Correct unequal candidates/route."""
    route_index=np.asarray(route_index);n=len(linear);counts=np.bincount(route_index,minlength=n)
    assert np.all(counts>0) and len(route_index)==len(s)==len(v)
    base=-np.log(counts[route_index]); uniform=base-mathlog(n)
    scores={'Uniform':uniform,'Shortest':base+np.where(route_index==0,0.,-np.inf),
            'Linear':np.log(np.maximum(probability(linear),1e-300))[route_index]+base,
            'R':np.log(np.maximum(probability(neural),1e-300))[route_index]+base}
    ls=np.log(np.maximum(s,1e-8));lv=np.log(np.maximum(v,1e-8))
    scores.update(S=uniform+ls,V=uniform+lv,SV=uniform+.5*(ls+lv))
    scores.update({'R+S':scores['R']+ls,'R+V':scores['R']+lv,'R+SV':scores['R']+.5*(ls+lv)})
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    gum=-np.log(-np.log(rng.uniform(1e-12,1-1e-12,len(route_index))))
    result={}
    for name in METHODS:
        score=scores[name] if valid else uniform
        weights=probability(score);mass=np.bincount(route_index,weights=weights,minlength=n)
        result[name]={'candidate':int(np.argmax(score+gum)),'weights':weights.tolist(),'route_weights':mass.tolist(),
            'effective_routes':float(np.exp(-sum(p*np.log(p) for p in mass if p>0))),'fallback':not valid}
    return result


def mathlog(n):return float(np.log(n))


def generate(cat,network,models,frozen,seed,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
    try:
        receipt=json.loads((cat/'complete.json').read_text());assert receipt['complete']
        assert digest(cat/'routes.json')==receipt['routes_sha256'] and digest(cat/'od.json')==receipt['od_sha256']
        ods={r['id']:r for r in json.loads((cat/'od.json').read_text())['requests']}
        roads=RoadFeatures(network);policies={k:load_policy(models/f'{k}-{seed}')[0] for k in ['linear','neural']}
        for k in policies:assert json.loads((models/f'{k}-{seed}'/'config.json').read_text())['network_sha256']==digest(network)
        teachers={};cache=frozen/'gaf-cache';ch=digest(cache/'complete.json');meta=json.loads((cache/'complete.json').read_text())['spatial']['frozen_parameters']
        for view in ['spatial','speed']:teachers[view],_=load_reader(frozen/'teachers'/f'{view}-42',ch,view)
        mr=frozen/'models'/f'native_0_25s-{seed}';mc=json.loads((mr/'complete.json').read_text());config=json.loads((mr/'config.json').read_text())
        assert mc['complete'] and mc['model_sha256']==digest(mr/'model.pt') and mc['stats_sha256']==digest(mr/'stats.json')
        assert mc['config_sha256']==digest(mr/'config.json') and config['model_source_sha256']==digest(inspect.getsourcefile(MotionDenoiser))
        state=torch.load(mr/'model.pt',map_location='cpu',weights_only=False)
        assert state['seed']==seed and state['arm']=='native_0_25s' and state['classes']==CLASSES
        model=MotionDenoiser(base=state['width']).cuda().eval();model.load_state_dict(state['state_dict']);stats=json.loads((mr/'stats.json').read_text());assert state['stats']==stats
        rows=[];tracks=[];chosen=[];routings=[];failures=[];calls={'n':0}
        model.register_forward_hook(lambda *args:calls.__setitem__('n',calls['n']+1))
        for r in json.loads((cat/'routes.json').read_text())['rows']:
            if not r['routes']:continue
            od=ods[r['id']];lp={};routing_failed=False
            try:
                for kind,policy in policies.items():lp[kind]=[route_logprob(roads,policy,od['vclass'],r['label'],route['edges'],od['stops']) for route in r['routes']]
            except ValueError as e:
                routing_failed=True;lp={k:[0.]*len(r['routes']) for k in policies};failures.append({'id':r['id'],'reason':str(e),'kind':'route_policy_uniform_fallback'})
            routings.append({'id':r['id'],'label':r['label'],'logprob':lp,'fallback':routing_failed})
            batch=[]
            for k in range(8):
                ri=k%len(r['routes']);route=r['routes'][ri];name=f"{r['id']}/candidate-{k:02d}"
                request={'id':name,'label':r['label'],'length':route['length'],'path':np.asarray(route['path'],np.float32)}
                with torch.no_grad():draw=sample(model,[request],stats,'native_0_25s',seed,device='cuda',ddim_steps=100)[0]
                xy=curve_xy(np.asarray(route['vertices']),draw['progress']);txy=np.c_[draw['time'],xy]
                assert np.allclose(xy[[0,-1]],np.asarray(route['vertices'])[[0,-1]],atol=1e-8,rtol=0) and np.all(np.diff(draw['progress'])>=0)
                row={'id':name,'request_id':r['id'],'label':r['label'],'candidate':k,'route_index':ri,'route_length':route['length'],'detour_ratio':route['detour_ratio'],'index':len(rows)}
                try:
                    seq,clip=spatial_sequence(xy,meta);rgb,chclip=encode_frozen(seq[None],np.asarray(meta['channel_min']),np.asarray(meta['channel_max']));v,_=speed_series(txy[:,0],xy)
                    with torch.no_grad():probs=[float(teachers[view](image_inputs(view,value)).softmax(1)[0,r['label']]) for view,value in [('spatial',rgb),('speed',v[None])]]
                    assert np.isfinite(probs).all()
                    row.update(probabilities=probs,spatial_image_sha256=hashlib.sha256(rgb[0].tobytes()).hexdigest(),xy_clipped_points=clip,channel_clipped=int(chclip))
                except ValueError as e:row.update(probabilities=[0.,0.],encoding_failed=str(e))
                batch.append(row);rows.append(row);tracks.append(txy)
            valid=all('encoding_failed' not in x for x in batch);q=np.asarray([x['probabilities'] for x in batch])
            picks=selections([x['route_index'] for x in batch],lp['linear'],lp['neural'],q[:,0],q[:,1],f'{seed}:{r["id"]}:g38',valid)
            for m,v in picks.items():chosen.append(dict(v,request_id=r['id'],label=r['label'],method=m,index=batch[v['candidate']]['index']))
            save(out/'progress.json',{'requests':len(routings),'candidates':len(rows),'seconds':time.monotonic()-start});print(json.dumps({'requests':len(routings),'seconds':time.monotonic()-start}),flush=True)
        assert tracks
        np.savez_compressed(out/'samples.npz',offsets=np.r_[0,np.cumsum([len(x) for x in tracks])],txy=np.concatenate(tracks))
        save(out/'candidates.json',{'rows':rows});save(out/'selections.json',{'rows':chosen});save(out/'routings.json',{'rows':routings,'failures':failures})
        save(out/'complete.json',{'complete':True,'requested':receipt['requested'],'generated':len(routings),'candidates':len(rows),'network_forwards':calls['n'],
            'seconds':time.monotonic()-start,'seed':seed,'reference_opened':False,'new_diffusion_training':False,
            'encoding_fallback_pools':sum(s['method']=='Uniform' and s['fallback'] for s in chosen),'policy_fallback_pools':len(failures),
            'model_sha256':digest(mr/'model.pt'),'route_model_hashes':{k:digest(models/f'{k}-{seed}'/'model.pt') for k in policies},
            'catalog_sha256':digest(cat/'routes.json'),'script_sha256':digest(__file__),
            'artifacts':{n:digest(out/n) for n in ['samples.npz','candidates.json','selections.json','routings.json']}})
    except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise


def checks():
    idx=np.arange(8)%3
    r=selections(idx,[0,0,0],[0,0,0],[.4]*8,[.4]*8,'test')
    assert np.allclose(r['Uniform']['route_weights'],1/3)
    for m in ['Linear','R','S','V','SV','R+S','R+V','R+SV']:
        assert r[m]['candidate']==r['Uniform']['candidate'] and np.allclose(r[m]['weights'],r['Uniform']['weights'])
    assert r['Shortest']['route_weights']==[1.,0.,0.]
    f=selections(idx,[0,3,6],[2,3,4],np.arange(8)/8,np.arange(8)/8,'test',False)
    assert all(v['candidate']==f['Uniform']['candidate'] and np.allclose(v['weights'],f['Uniform']['weights']) for v in f.values())
    print('G38 pool checks passed: route-multiplicity correction, constant-score control, shared random draws, common fallback')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['catalog','generate','checks']);p.add_argument('--prepared',type=Path);p.add_argument('--catalog',type=Path);p.add_argument('--network',type=Path);p.add_argument('--models',type=Path);p.add_argument('--frozen',type=Path);p.add_argument('--seed',type=int);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='catalog':catalog(a.prepared,a.network,a.output)
    elif a.action=='generate':generate(a.catalog,a.network,a.models,a.frozen,a.seed,a.output)
    else:checks()
