"""Controlled relative-coordinate diffusion: map condition and road-loss ablations."""
import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from cnr.road_diffusion import RoadDenoiser,RoadFields
from cnr.representations import RepresentationClassifier


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['relative','map','map_roadloss'],required=True)
    p.add_argument('--data',default='data/processed/most64')
    p.add_argument('--fields',default='data/maps/road_fields_4m.npz')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--semantic-teacher',default=None)
    p.add_argument('--semantic-weight',type=float,default=.01)
    p.add_argument('--semantic-objective',choices=['ce','consistency'],default='ce')
    p.add_argument('--shuffle-semantic-labels',action='store_true')
    p.add_argument('--noise-seed',type=int,default=None)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    data=Path(a.data);tr=np.load(data/'train.npz');va=np.load(data/'val.npz');meta=json.loads((data/'dataset.json').read_text())
    rel=tr['xy'][:,1:].astype(np.float64)-tr['xy'][:,:1].astype(np.float64)
    rm=rel.mean(axis=(0,1));rs=rel.std(axis=(0,1)).clip(1e-6)
    ds=np.diff(tr['xy'].astype(np.float64),axis=1).std(axis=(0,1)).clip(1e-6)
    to=lambda v:torch.tensor(v,dtype=torch.float32,device='cuda')
    relmean,relstd,deltastd=to(rm),to(rs),to(ds);mean,std=to(meta['mean']),to(meta['std'])
    xy=torch.from_numpy(tr['xy']).cuda();label=torch.from_numpy(tr['labels']).cuda()
    model=RoadDenoiser(use_map=a.variant!='relative').cuda();ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    teacher=None;teacher_record=None
    if a.semantic_teacher:
        tp=Path(a.semantic_teacher);tc=json.loads((tp/'config.json').read_text())
        if tc['variant'] not in ['delta_gaf','delta_tcn']:raise ValueError('Use matched differentiable displacement inputs')
        teacher=RepresentationClassifier(tc['variant'],tc['stats']).cuda().eval()
        teacher.load_state_dict(torch.load(tp/'best.pt',map_location='cuda',weights_only=True))
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        teacher_record={'variant':tc['variant'],'sha256':hashlib.sha256((tp/'best.pt').read_bytes()).hexdigest()}
    if a.noise_seed is not None:torch.manual_seed(a.noise_seed)
    shuffle_generator=torch.Generator(device='cuda').manual_seed(a.seed+3000)
    reference_probabilities=None
    if teacher is not None and a.semantic_objective=='consistency':
        if a.shuffle_semantic_labels:raise ValueError('Consistency and label permutation are separate experiments')
        with torch.no_grad():
            reference_probabilities=torch.cat([teacher(xy[j:j+128]).softmax(-1) for j in range(0,len(xy),128)])
    roads=RoadFields(a.fields).cuda()
    patches=None
    with torch.no_grad():
        if model.use_map:
            patches=torch.cat([roads.patches(xy[i:i+256,0],label[i:i+256]) for i in range(0,len(xy),256)])
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    config=vars(a)|{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   'semantic_teacher_record':teacher_record,'semantic_timesteps':'t<100',
                   'relative_mean':rm.tolist(),'relative_std':rs.tolist(),'delta_std':ds.tolist(),
                   'ema_decay':.999,'motion_loss_weight':.02,'motion_loss_timesteps':'t<100',
                   'road_loss_weight':.05 if a.variant=='map_roadloss' else 0.,'road_loss_timesteps':'t<250',
                   'map_patch':'32x32 centered at supplied start, radius256m; class-permitted distance and proximity',
                   'dataset':meta,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
                   'parameters':sum(p.numel() for p in model.parameters()),
                   'road_field_sha256':hashlib.sha256(Path(a.fields).read_bytes()).hexdigest(),
                   'data_hashes':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['train.npz','val.npz']},
                   'scope':'single-trajectory geometric conditioning; topology not guaranteed; validation-only pilot'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    ordergen=torch.Generator(device='cuda').manual_seed(a.seed+1000)
    order=torch.randperm(len(xy),generator=ordergen,device='cuda');offset=0;started=time.monotonic()
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.steps+1):
            if offset>=len(xy):order=torch.randperm(len(xy),generator=ordergen,device='cuda');offset=0
            ix=order[offset:offset+a.batch_size];offset+=a.batch_size
            x=xy[ix];y=label[ix];start=x[:,0];batch=len(x)
            target=((x[:,1:]-start[:,None]-relmean)/relstd).transpose(1,2)
            t=torch.randint(0,1000,(batch,),device='cuda');noise=torch.randn_like(target);aa=alpha[t,None,None]
            patch=patches[ix] if patches is not None else None
            model.train();optim.zero_grad(set_to_none=True)
            eps=model(aa.sqrt()*target+(1-aa).sqrt()*noise,t,y,(start-mean)/std,patch)
            diffusion=(eps-noise).square().mean()
            clean=((aa.sqrt()*target+(1-aa).sqrt()*noise)-(1-aa).sqrt()*eps)/aa.sqrt()
            predrel=clean.transpose(1,2)*relstd+relmean
            pred=torch.cat((start[:,None],predrel+start[:,None]),1)
            motion=torch.nn.functional.smooth_l1_loss(torch.diff(pred,dim=1)/deltastd,
                         torch.diff(x,dim=1)/deltastd,reduction='none').mean((1,2))
            active=(t<100).float();motion=(motion*active).sum()/active.sum().clamp_min(1)
            road=torch.zeros((),device='cuda')
            if a.variant=='map_roadloss':
                active=t<250
                if active.any():
                    distance=roads.distances(pred[active,1:],y[active])/10
                    road=torch.nn.functional.smooth_l1_loss(distance,torch.zeros_like(distance))
            semantic=torch.zeros((),device='cuda')
            if teacher is not None:
                active=t<100
                if active.any():
                    targets=y[active]
                    if a.shuffle_semantic_labels:
                        # Permute full minibatch labels, independently of diffusion RNG.
                        targets=y[torch.randperm(batch,device='cuda',generator=shuffle_generator)][active]
                    logits=teacher(pred[active])
                    if a.semantic_objective=='consistency':
                        semantic=torch.nn.functional.kl_div(logits.log_softmax(-1),reference_probabilities[ix][active],reduction='batchmean')
                    else:semantic=torch.nn.functional.cross_entropy(logits,targets)
            loss=diffusion+.02*motion+(.05*road if a.variant=='map_roadloss' else 0)+a.semantic_weight*semantic
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.001)
            if step==1 or step%1000==0 or step==a.steps:
                torch.cuda.synchronize()
                row={'step':step,'loss':loss.item(),'noise_mse':diffusion.item(),'motion':motion.item(),
                     'road':road.item(),'semantic':semantic.item(),'elapsed_s':time.monotonic()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'step':step},out/'latest.pt')
    selected=np.concatenate([np.flatnonzero(va['labels']==c)[np.linspace(0,(va['labels']==c).sum()-1,
                                  min(32,(va['labels']==c).sum())).astype(int)] for c in range(6)])
    real=torch.from_numpy(va['xy'][selected]).cuda();labels=torch.from_numpy(va['labels'][selected]).cuda()
    start=real[:,0];patch=roads.patches(start,labels) if ema.use_map else None
    torch.manual_seed(2026);sample=torch.randn(len(real),2,meta['length']-1,device='cuda')
    times=torch.linspace(999,0,100).long().tolist()
    with torch.no_grad():
        for j,t in enumerate(times):
            eps=ema(sample,torch.full((len(real),),t,device='cuda'),labels,(start-mean)/std,patch)
            clean=(sample-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            prev=alpha[times[j+1]] if j+1<len(times) else torch.tensor(1.,device='cuda')
            sample=prev.sqrt()*clean+(1-prev).sqrt()*eps
        generated=torch.cat((start[:,None],sample.transpose(1,2)*relstd+relmean+start[:,None]),1)
        if not torch.isfinite(generated).all():raise RuntimeError('Nonfinite samples')
    np.savez_compressed(out/'validation_samples.npz',xy=generated.cpu().numpy(),labels=labels.cpu().numpy(),validation_indices=selected)
    (out/'complete.json').write_text(json.dumps({'status':'complete','variant':a.variant,'steps':a.steps,
        'elapsed_s':time.monotonic()-started,'generated_samples':len(selected),'sampler':'EMA DDIM100 unclipped'},indent=2))


if __name__=='__main__':main()
