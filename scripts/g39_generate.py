"""Direct draws with shared prospective route/prior T; no GAF reranking."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from g33_train import digest,save
from g33_model import MotionDenoiser,CLASSES,sample
from g33_evaluate import curve_xy
from g38_policy import RoadFeatures,load_policy,route_logprob
from g38_pool import probability


def generate(cat,network,policy,model_folder,seed,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
    try:
        cr=json.loads((cat/'complete.json').read_text());assert cr['complete']
        assert digest(cat/'routes.json')==cr['routes_sha256'] and digest(cat/'od.json')==cr['od_sha256']
        ods={r['id']:r for r in json.loads((cat/'od.json').read_text())['requests']}
        roads=RoadFeatures(network);rp,_=load_policy(policy)
        assert json.loads((policy/'config.json').read_text())['network_sha256']==digest(network)
        mr=json.loads((model_folder/'complete.json').read_text())
        assert mr['complete'] and mr['model_sha256']==digest(model_folder/'model.pt')
        assert mr['stats_sha256']==digest(model_folder/'stats.json') and not (model_folder/'failed.json').exists()
        state=torch.load(model_folder/'model.pt',map_location='cpu',weights_only=False)
        assert state['classes']==CLASSES and state['arm']=='native_0_25s' and state['seed']==seed
        model=MotionDenoiser(base=state['width']).cuda().eval();model.load_state_dict(state['state_dict'],strict=True)
        stats=json.loads((model_folder/'stats.json').read_text());assert state['stats']==stats
        rows=[];tracks=[];picks=[];routings=[];calls=0;pairing=[];failures=[]
        for r in json.loads((cat/'routes.json').read_text())['rows']:
            if not r['routes']:continue
            od=ods[r['id']];fallback=False
            try:lp=[route_logprob(roads,rp,od['vclass'],r['label'],route['edges'],od['stops']) for route in r['routes']]
            except ValueError as e:
                fallback=True;lp=[0.]*len(r['routes']);failures.append({'id':r['id'],'reason':str(e),'kind':'policy_uniform_fallback'})
            probs=probability(lp);batch=[]
            for k in range(4):
                key=int.from_bytes(hashlib.sha256(f'G39:route:{seed}:{r["id"]}:{k}'.encode()).digest()[:8],'little')
                ri=int(np.random.default_rng(key).choice(len(probs),p=probs));route=r['routes'][ri]
                name=f'{r["id"]}/draw-{k:02d}'
                request={'id':name,'label':r['label'],'length':route['length'],'path':np.asarray(route['path'],np.float32)}
                draw=sample(model,[request],stats,'native_0_25s',seed,ddim_steps=100)[0];calls+=100
                xy=curve_xy(np.asarray(route['vertices']),draw['progress']);txy=np.c_[draw['time'],xy]
                assert np.isfinite(txy).all() and np.all(np.diff(draw['progress'])>=0)
                assert np.allclose(xy[[0,-1]],np.asarray(route['vertices'])[[0,-1]],atol=1e-8,rtol=0)
                row={'id':name,'request_id':r['id'],'label':r['label'],'candidate':k,'index':len(rows),
                     'route_index':ri,'route_length':route['length'],'detour_ratio':route['detour_ratio'],
                     'duration':draw['duration'],'motion_bins':len(draw['time'])-1,'negative_bins':draw['negative_bins'],'upper_clipped_bins':draw['upper_clipped_bins'],
                     'constant_speed_fallback':draw['constant_speed_fallback']}
                rows.append(row);tracks.append(txy);batch.append(row)
                pairing.append([name,ri,draw['duration']])
            mass=np.bincount([v['route_index'] for v in batch],minlength=len(probs))/4
            picks.append({'method':'R','request_id':r['id'],'label':r['label'],'index':batch[0]['index'],'candidate':0,
                          'weights':[.25]*4,'route_weights':mass.tolist(),
                          'effective_routes':float(np.exp(-sum(p*np.log(p) for p in mass if p>0))),'fallback':fallback})
            routings.append({'id':r['id'],'label':r['label'],'logprob':{'neural':lp},'fallback':fallback})
            save(out/'progress.json',{'requests':len(routings),'candidates':len(rows),'seconds':time.monotonic()-start})
            if len(routings)%10==0:print(json.dumps({'requests':len(routings),'seconds':time.monotonic()-start}),flush=True)
        assert tracks
        np.savez_compressed(out/'samples.npz',offsets=np.r_[0,np.cumsum([len(x) for x in tracks])],txy=np.concatenate(tracks))
        save(out/'candidates.json',{'rows':rows});save(out/'selections.json',{'rows':picks});save(out/'routings.json',{'rows':routings,'failures':failures})
        save(out/'complete.json',{'complete':True,'seed':seed,'alignment_arm':state['alignment_arm'],'requested':cr['requested'],
             'generated':len(routings),'candidates':len(rows),'seconds':time.monotonic()-start,'network_forwards':calls,
             'reference_opened':False,'GAF_used_at_sampling':False,'primary':'candidate0 K1, no selection','secondary':'equal-weight four direct draws',
             'route_duration_pairing_sha256':hashlib.sha256(json.dumps(pairing,separators=(',',':')).encode()).hexdigest(),
             'catalog_sha256':digest(cat/'routes.json'),'model_sha256':digest(model_folder/'model.pt'),
             'artifacts':{n:digest(out/n) for n in ('samples.npz','candidates.json','selections.json','routings.json')}})
    except BaseException as e:save(out/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('catalog','network','policy','model','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args();generate(a.catalog,a.network,a.policy,a.model,a.seed,a.output)
