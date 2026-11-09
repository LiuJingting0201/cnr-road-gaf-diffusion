"""Route × train-prior duration × motion candidates, shared by all GAF rules."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from g33_model import MotionDenoiser,CLASSES
from g33_train import digest,save
from g33_evaluate import curve_xy
from g34_fixed_duration import duration_plan,sample_fixed
from g38_policy import RoadFeatures,load_policy,route_logprob
from g38_pool import probability
from g36_feedback import load_reader,spatial_sequence
from g35c_teachers import speed_profile
from g28_teachers import image_inputs
from cnr.original_gaf import encode_frozen
from g39_mechanism import preserve
import g38_evaluate as evaluation

METHODS=('R',)+tuple(f'{kind}-{view}' for kind in ('Global','FixedRoute','FixedRouteT') for view in ('S','V','SV'))


def weights(route_index,duration_index,route_probability,scores,valid=True):
    ri=np.asarray(route_index);di=np.asarray(duration_index);rp=np.asarray(route_probability)
    counts=np.bincount(ri,minlength=len(rp));assert np.all(counts==8)
    assert all(np.sum((ri==r)&(di==d))==4 for r in range(len(rp)) for d in (0,1))
    base=rp[ri]/counts[ri];out={'R':base}
    for name in METHODS[1:]:
        kind,view=name.split('-');s=scores[:,0] if view=='S' else scores[:,1] if view=='V' else scores.mean(1)
        if not valid:out[name]=base.copy()
        elif kind=='Global':out[name]=probability(np.log(np.maximum(base,1e-300))+s)
        else:out[name]=preserve(base,ri if kind=='FixedRoute' else ri*2+di,s)
    return out


def generate(cat,network,policy,frozen,seed,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
    try:
        receipt=json.loads((cat/'complete.json').read_text());assert receipt['complete']
        assert digest(cat/'routes.json')==receipt['routes_sha256'] and digest(cat/'od.json')==receipt['od_sha256']
        ods={r['id']:r for r in json.loads((cat/'od.json').read_text())['requests']}
        roads=RoadFeatures(network);rp,_=load_policy(policy)
        assert json.loads((policy/'config.json').read_text())['network_sha256']==digest(network)
        folder=frozen/'models'/f'native_0_25s-{seed}';mr=json.loads((folder/'complete.json').read_text())
        assert mr['complete'] and digest(folder/'model.pt')==mr['model_sha256']
        state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);stats=state['stats']
        assert state['classes']==CLASSES and state['arm']=='native_0_25s' and state['seed']==seed
        model=MotionDenoiser(base=state['width']).cuda().eval();model.load_state_dict(state['state_dict'],strict=True)
        cache=frozen/'gaf-cache';ch=digest(cache/'complete.json');meta=json.loads((cache/'complete.json').read_text())['spatial']['frozen_parameters']
        readers={v:load_reader(frozen/'teachers'/f'{v}-42',ch,v)[0] for v in ('spatial','speed')}
        rows=[];tracks=[];selections=[];routings=[];plans=[];failures=[];marginal_errors=[]
        for r in json.loads((cat/'routes.json').read_text())['rows']:
            if not r['routes']:continue
            od=ods[r['id']];fallback=False
            try:lp=[route_logprob(roads,rp,od['vclass'],r['label'],v['edges'],od['stops']) for v in r['routes']]
            except ValueError as e:lp=[0.]*len(r['routes']);fallback=True;failures.append({'id':r['id'],'reason':str(e)})
            routeprob=probability(lp);batch=[]
            for ri,route in enumerate(r['routes']):
                for di in (0,1):
                    request={'id':f'{r["id"]}/route-{ri}/duration-{di}','label':r['label'],'length':route['length'],'path':np.asarray(route['path'],np.float32)}
                    plan=duration_plan(request,stats['duration_prior'],seed);plans.append(plan)
                    for noise in range(4):
                        q=dict(request,id=f'{request["id"]}/candidate-{noise:02d}')
                        draw=sample_fixed(model,q,stats,plan,seed)
                        xy=curve_xy(np.asarray(route['vertices']),draw['progress']);txy=np.c_[draw['time'],xy]
                        row={'id':q['id'],'request_id':r['id'],'label':r['label'],'candidate':len(batch),'index':len(rows),'route_index':ri,
                             'duration_index':di,'duration':draw['duration'],'route_length':route['length'],'detour_ratio':route['detour_ratio']}
                        try:
                            seq,_=spatial_sequence(xy,meta);image,_=encode_frozen(seq[None],np.asarray(meta['channel_min']),np.asarray(meta['channel_max']))
                            velocity,_=speed_profile(txy[:,0],xy)
                            with torch.no_grad():prob=[float(readers[v](image_inputs(v,x)).softmax(1)[0,r['label']]) for v,x in [('spatial',image),('speed',velocity[None])]]
                            assert np.isfinite(prob).all();row['probabilities']=prob
                        except ValueError as e:row.update(probabilities=[0.,0.],encoding_failed=str(e))
                        rows.append(row);tracks.append(txy);batch.append(row)
            ri=np.array([v['route_index'] for v in batch]);di=np.array([v['duration_index'] for v in batch]);scores=np.log(np.maximum([v['probabilities'] for v in batch],1e-8))
            valid=all('encoding_failed' not in v for v in batch);ww=weights(ri,di,routeprob,scores,valid)
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'G39factorial:{seed}:{r["id"]}'.encode()).digest()[:8],'little'))
            gum=-np.log(-np.log(rng.uniform(1e-12,1-1e-12,len(batch))))
            for method,w in ww.items():
                mass=np.bincount(ri,weights=w,minlength=len(routeprob));joint=np.bincount(ri*2+di,weights=w,minlength=2*len(routeprob))
                if method.startswith('FixedRouteT'):
                    error=float(np.max(np.abs(joint-np.repeat(routeprob/2,2))));assert error<1e-12;marginal_errors.append(error)
                if method.startswith('FixedRoute'):assert np.allclose(mass,routeprob,atol=1e-12,rtol=0)
                k=int(np.argmax(np.log(np.maximum(w,1e-300))+gum))
                selections.append({'method':method,'request_id':r['id'],'label':r['label'],'index':batch[k]['index'],'candidate':k,
                    'weights':w.tolist(),'route_weights':mass.tolist(),'route_duration_weights':joint.tolist(),
                    'effective_routes':float(np.exp(-sum(p*np.log(p) for p in mass if p>0))),'fallback':not valid})
            routings.append({'id':r['id'],'label':r['label'],'logprob':{'neural':lp},'fallback':fallback})
            save(out/'progress.json',{'requests':len(routings),'candidates':len(rows),'seconds':time.monotonic()-start})
        np.savez_compressed(out/'samples.npz',offsets=np.r_[0,np.cumsum([len(t) for t in tracks])],txy=np.concatenate(tracks))
        for name,value in [('candidates',rows),('selections',selections),('routings',routings),('duration-plans',plans)]:save(out/(name+'.json'),{'rows':value})
        save(out/'complete.json',{'complete':True,'seed':seed,'reference_opened':False,'requested':receipt['requested'],'generated':len(routings),
             'candidates':len(rows),'seconds':time.monotonic()-start,'model_sha256':digest(folder/'model.pt'),'source':'frozen G35 native generator on exposed G38 OD catalog',
             'catalog_sha256':digest(cat/'routes.json'),'network_forwards':len(rows)*100,'new_diffusion_training':False,
             'joint_route_duration_max_abs_error':max(marginal_errors),'route_policy_fallbacks':failures,
             'budget':'each route2 training-prior durations x4 independent motion noises; same pool for all10policies',
             'artifacts':{n:digest(out/n) for n in ('samples.npz','candidates.json','selections.json','routings.json','duration-plans.json')}})
    except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise


def checks():
    ri=np.repeat(np.arange(3),8);di=np.tile(np.repeat([0,1],4),3);rp=np.array([.6,.3,.1]);s=np.random.default_rng(1).normal(size=(24,2))
    w=weights(ri,di,rp,s)
    for name,v in w.items():
        assert np.isclose(v.sum(),1.)
        if name.startswith('FixedRoute'):assert np.allclose(np.bincount(ri,weights=v),rp)
        if name.startswith('FixedRouteT'):assert np.allclose(np.bincount(ri*2+di,weights=v),np.repeat(rp/2,2))
    assert all(np.allclose(v,w['R']) for v in weights(ri,di,rp,np.ones((24,2))).values())
    print('G39 factorial checks: exact route and duration marginals, identical-score no-op, matched budget')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate','checks'])
    for k in ('catalog','network','policy','frozen','output','generated','reference','simulation'):p.add_argument('--'+k,type=Path)
    p.add_argument('--seed',type=int);a=p.parse_args()
    if a.action=='generate':generate(a.catalog,a.network,a.policy,a.frozen,a.seed,a.output)
    elif a.action=='evaluate':
        evaluation.METHODS=list(METHODS);evaluation.evaluate(a.generated,a.catalog,a.reference,a.simulation,a.network,a.output)
    else:checks()
