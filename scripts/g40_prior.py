"""Training-only conditional code prior; synthesis never reads target GAF."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from g33_train import digest, save
from g33_model import clock, pack_motion, sample_duration, decode
from g40_model import CodePrior, views, predict


def condition_arrays(paths, lengths, durations, stats):
    start = paths[:,0]
    context = np.c_[(start-np.asarray(stats['start_mean']))/np.asarray(stats['start_sd']),
                    np.log(lengths)/10.,np.log(durations)/10.,np.log(lengths/durations)/5.].astype(np.float32)
    route = ((paths-start[:,None])/1500.).astype(np.float32).transpose(0,2,1)
    return context, route


def train(folder, output, steps):
    output.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    try:
        receipt=json.loads((folder/'complete.json').read_text());assert receipt['complete']
        assert digest(folder/'model.pt')==receipt['model_sha256']
        assert digest(folder/'train-codes.npz')==receipt['codes_sha256']
        assert digest(folder/'stats.json')==receipt['stats_sha256']
        stats=json.loads((folder/'stats.json').read_text());arm=receipt['arm'];seed=receipt['seed']
        with np.load(folder/'train-codes.npz',allow_pickle=False) as f:
            data={k:f[k] for k in ('ids','parent_ids','labels','paths','length','duration','code')}
        assert len(set(data['ids']))==len(data['labels'])==receipt['training_rows']
        assert np.isfinite(data['code']).all()
        if arm in ('B','C'):
            centers=np.stack([data['code'][data['labels']==c].mean(0) for c in range(5)])
            assert np.allclose(data['code'],centers[data['labels']],atol=2e-4,rtol=2e-4)
            state={'kind':'zero' if arm=='B' else 'class_centroid','centers':torch.from_numpy(centers),'arm':arm,'seed':seed}
            # Use the exact first identical code rather than an accumulated mean.
            state['centers']=torch.from_numpy(np.stack([data['code'][np.flatnonzero(data['labels']==c)[0]] for c in range(5)]))
            steps_run=0;history=[]
        else:
            torch.set_num_threads(2);torch.manual_seed(seed+401000);torch.backends.cudnn.benchmark=False
            mask=np.r_[np.full(32,'spatial' in views(arm)),np.full(32,'speed' in views(arm))].astype(np.float32)
            mean=data['code'].mean(0);sd=np.maximum(data['code'].std(0),1e-3)
            context,route=condition_arrays(data['paths'],data['length'],data['duration'],stats)
            convert=lambda x:torch.as_tensor(x,device='cuda')
            z=convert(((data['code']-mean)/sd)*mask);labels=convert(data['labels']);ct=convert(context);rt=convert(route);active=convert(mask)
            model=CodePrior().cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
            opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
            schedule=torch.cumprod(1-torch.linspace(.0001,.02,1000,device='cuda'),0)
            rng=np.random.default_rng(seed+401001);groups=[np.flatnonzero(data['labels']==c) for c in range(5)];history=[]
            with (output/'training.jsonl').open('w',buffering=1) as log:
                for step in range(1,steps+1):
                    if time.monotonic()-start>1800:raise TimeoutError('30 minute prior fit limit')
                    classes=rng.integers(0,5,256);ix=np.array([rng.choice(groups[c]) for c in classes])
                    t=torch.randint(1000,(len(ix),),device='cuda');noise=torch.randn_like(z[ix])*active
                    alpha=schedule[t,None];noisy=alpha.sqrt()*z[ix]+(1-alpha).sqrt()*noise
                    pred=model(noisy,t,labels[ix],ct[ix],rt[ix]);loss=((pred-noise).square()*active).sum()/(len(ix)*active.sum())
                    opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    values=torch.stack([loss.detach(),norm.detach()]).cpu().numpy()
                    if not np.isfinite(values).all():raise ValueError('Nonfinite prior loss/gradient')
                    opt.step()
                    with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.001)
                    if step==1 or step%100==0 or step==steps:
                        row={'step':step,'loss':float(values[0]),'seconds':time.monotonic()-start}
                        history.append(row);log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            state={'kind':'diffusion','state_dict':ema.state_dict(),'mean':torch.from_numpy(mean),'sd':torch.from_numpy(sd),
                   'mask':torch.from_numpy(mask),'arm':arm,'seed':seed,'sample_steps':50}
            steps_run=steps
        state['motion_model_sha256']=receipt['model_sha256'];torch.save(state,output/'model.pt')
        save(output/'complete.json',{'complete':True,'arm':arm,'seed':seed,'kind':state['kind'],'steps':steps_run,
            'seconds':time.monotonic()-start,'model_sha256':digest(output/'model.pt'),'motion_model_sha256':receipt['model_sha256'],
            'training_codes_sha256':receipt['codes_sha256'],'training_rows':len(data['labels']),
            'conditions':'available route geometry, class, training-prior sampled duration; no held-out code/trajectory',
            'code_dimensions':64,'training_only':True,'no_validation_selection':True,'lr':.0002,'batch':256,'ema':.001,
            'history':history,'source_sha256':digest(__file__)})
    except BaseException as error:
        save(output/'failed.json',{'error':repr(error),'seconds':time.monotonic()-start});raise


def load_prior(folder, motion_hash, device='cuda'):
    r=json.loads((folder/'complete.json').read_text())
    assert r['complete'] and r['training_only'] and not (folder/'failed.json').exists()
    assert r['motion_model_sha256']==motion_hash and digest(folder/'model.pt')==r['model_sha256']
    state=torch.load(folder/'model.pt',map_location=device,weights_only=False)
    assert state['motion_model_sha256']==motion_hash
    if state['kind']=='diffusion':
        model=CodePrior().to(device).eval();model.load_state_dict(state['state_dict'],strict=True);state['model']=model
    return state


@torch.no_grad()
def sample_code(prior,batch,request_id,seed):
    if prior['kind']=='zero':return None
    if prior['kind']=='class_centroid':return prior['centers'][batch['labels']]
    device=batch['x'].device
    key=int.from_bytes(hashlib.sha256(f'G40:style:{seed}:{request_id}'.encode()).digest()[:8],'little')
    generator=torch.Generator(device=device).manual_seed(key)
    z=torch.randn((len(batch['labels']),64),generator=generator,device=device)*prior['mask']
    schedule=torch.cumprod(1-torch.linspace(.0001,.02,1000,device=device),0)
    steps=torch.linspace(999,0,50).long().tolist()
    for j,t in enumerate(steps):
        epsilon=prior['model'](z,torch.full((len(z),),t,dtype=torch.long,device=device),batch['labels'],batch['context'],batch['route'])
        clean=(z-(1-schedule[t]).sqrt()*epsilon)/schedule[t].sqrt()
        previous=schedule[steps[j+1]] if j+1<len(steps) else 1.
        z=(previous**.5*clean+(1-previous)**.5*epsilon)*prior['mask']
    return (z*prior['sd']+prior['mean'])*prior['mask']


@torch.no_grad()
def sample_motion(model,prior,request,stats,seed,device='cuda',code_override=None):
    # This public sampling entry accepts only an already-proposed public route.
    if set(request)!={'id','label','length','path'}:raise ValueError('Future or undeclared sampling input rejected')
    key=int.from_bytes(hashlib.sha256(f"G33:{request['id']}:{seed}".encode()).digest()[:8],'little')
    duration,bounded=sample_duration(stats['duration_prior'],[request['length']],[request['label']],np.random.default_rng(key))
    times=clock(float(duration[0]),'native_0_25s');n=len(times)-1
    batch=pack_motion([np.zeros(n,np.float32)],[times],[request['label']],[request['path']],
                      [request['length']],duration,stats,device)
    if code_override is not None:raise ValueError('Known-future code is forbidden in prospective synthesis')
    code=sample_code(prior,batch,request['id'],seed)
    generator=torch.Generator(device=device).manual_seed(key)
    x=torch.randn(batch['x'].shape,generator=generator,device=device)*batch['mask']
    schedule=torch.cumprod(1-torch.linspace(.0001,.02,1000,device=device),0);steps=torch.linspace(999,0,100).long().tolist()
    for j,t in enumerate(steps):
        epsilon=predict(model,batch,x,torch.full((1,),t,dtype=torch.long,device=device),code)
        clean=(x-(1-schedule[t]).sqrt()*epsilon)/schedule[t].sqrt();previous=schedule[steps[j+1]] if j+1<len(steps) else 1.
        x=(previous**.5*clean+(1-previous)**.5*epsilon)*batch['mask']
    progress,flags=decode(x[0,0,:n].cpu().numpy(),times,stats['motion_mean'],stats['motion_sd'])
    return {'id':request['id'],'duration':float(duration[0]),'time':times,'progress':progress,
            'duration_bounded':bool(bounded[0]),'code_abs_max':float(code.abs().max()) if code is not None else 0.,**flags}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5000);a=p.parse_args();train(a.model,a.output,a.steps)
