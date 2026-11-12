"""Fit the released coordinate architecture on the paper's public Porto benchmark.

Run prepare once, then torchrun --standalone --nproc-per-node=4 porto.py train.
Evaluate the final checkpoint on the reserved 3,000 requests with porto.py score.
"""
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from utils.Traj_UNet import Guide_UNet, Model
from utils.config import args
from utils.utils import p_xt, resample_trajectory


def arc_length(track):
    lon,lat=np.radians(track).T
    a=np.sin(np.diff(lat)/2)**2+np.cos(lat[1:])*np.cos(lat[:-1])*np.sin(np.diff(lon)/2)**2
    return float((12742000*np.arcsin(np.sqrt(np.clip(a,0,1)))).sum())


def grid(trips, lo, hi):
    q=np.floor((trips-lo)/(hi-lo)*16).astype(np.int64)
    return np.clip(q[...,0],0,15)*16+np.clip(q[...,1],0,15)


def prepare():
    raw=np.load('observations.npz')
    sizes=raw['length']
    taxi=raw['taxi']
    stamps=raw['start']
    delta=raw['delta']
    xy=np.empty((len(sizes),200,2),dtype=np.float32)
    start=0
    head=np.empty((len(sizes),8),dtype=np.float32)
    durations=raw['duration'] if 'duration' in raw.files else None
    tz=ZoneInfo('Asia/Shanghai' if durations is not None else 'Europe/Lisbon')
    for i,n in enumerate(sizes):
        tr=delta[start:start+n].cumsum(0)*1e-6
        start+=n
        xy[i]=resample_trajectory(tr,200)
        stamp=datetime.fromtimestamp(int(stamps[i]),tz)
        length=arc_length(tr)
        duration=durations[i] if durations is not None else (n-1)*15.
        head[i,:6]=[stamp.hour*12+stamp.minute//5,length,duration,n,length/(n-1),length/duration]
    # One taxi belongs to one partition throughout. All transformations use training data.
    split=np.array([int(hashlib.sha256(str(t).encode()).hexdigest()[:8],16)%10 for t in taxi])
    train=split<8
    lo=xy[train].min((0,1));hi=xy[train].max((0,1))
    head[:,6]=grid(xy[:,0],lo,hi)
    head[:,7]=grid(xy[:,-1],lo,hi)
    head[:,1:6]=(head[:,1:6]-head[train,1:6].mean(0))/head[train,1:6].std(0).clip(1e-7)
    mean=xy[train].mean((0,1),dtype=np.float64)
    std=xy[train].std((0,1),dtype=np.float64)
    np.savez('prepared.npz',xy=xy,h=head,split=split,lo=lo,hi=hi,mean=mean,std=std,sizes=sizes)
    print(json.dumps(dict(train=int(train.sum()),development=int((split==8).sum()),held_out=int((split==9).sum()),coord_range=[lo.tolist(),hi.tolist()])))


def model():
    config=SimpleNamespace(**{k:SimpleNamespace(**v) for k,v in args.items()})
    if '--unconditional' in sys.argv:
        class Unconditional(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unet=Model(config)
            def forward(self,x,t,head):
                return self.unet(x,t,extra_embed=None)
        return Unconditional().cuda()
    return Guide_UNet(config).cuda()


def schedule():
    return torch.linspace(.0001,.05,500,device='cuda')


def train():
    rank=int(os.environ['LOCAL_RANK'])
    world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl')
    torch.manual_seed(42)
    fast='--large-microbatch' in sys.argv
    net=DistributedDataParallel(model(),device_ids=[rank],find_unused_parameters=not fast)
    torch.manual_seed(42+rank)
    data=np.load('prepared.npz')
    keep=data['split']<8
    x=torch.tensor(((data['xy'][keep]-data['mean'])/data['std']).astype('f')).cuda().transpose(1,2)
    h=torch.tensor(data['h'][keep]).cuda()
    optim=torch.optim.AdamW(net.parameters(),lr=2e-4)
    alpha=(1-schedule()).cumprod(0)
    from contextlib import nullcontext
    import time
    start=time.monotonic()
    epochs=2 if '--timing-run' in sys.argv else 200
    for epoch in range(1,epochs+1):
        gen=torch.Generator().manual_seed(42+epoch)
        order=torch.randperm(len(x),generator=gen)
        loss_total=0.
        for group in order.split(1024):
            # Distribute equal lengths across ranks, trimming at most world-1 examples per epoch.
            n=(len(group)//world)*world
            if n==0:
                continue
            local=group[:n].reshape(world,-1)[rank].cuda()
            opt_batches=local.split(256 if fast else 32)
            optim.zero_grad()
            for j,idx in enumerate(opt_batches):
                with net.no_sync() if j+1<len(opt_batches) else nullcontext():
                    t=torch.randint(500,(len(idx)//2+1,),device='cuda')
                    t=torch.cat((t,499-t))[:len(idx)]
                    a=alpha[t,None,None]
                    e=torch.randn_like(x[idx])
                    z=a.sqrt()*x[idx]+(1-a).sqrt()*e
                    loss=(net(z,t,h[idx])-e).square().mean()
                    (loss*(len(idx)/len(local))).backward()
                    loss_total+=float(loss.detach())*len(idx)
            optim.step()
        if rank==0:
            log=dict(epoch=epoch,rank0_mean_noise_loss=loss_total/(len(order)//world),elapsed_s=time.monotonic()-start)
            if not np.isfinite(log['rank0_mean_noise_loss']):
                raise FloatingPointError(log)
            with open('epochs.jsonl','a') as f:
                print(json.dumps(log),file=f)
            if epoch==1 or epoch%10==0:
                torch.save(dict(epoch=epoch,weights=net.module.state_dict()),'checkpoint.tmp')
                os.replace('checkpoint.tmp','checkpoint.pt')
    torch.distributed.destroy_process_group()


def score():
    from scipy.spatial.distance import jensenshannon
    data=np.load('prepared.npz')
    dev='--development' in sys.argv
    ids=np.flatnonzero(data['split']==(8 if dev else 9))
    raw=np.load('observations.npz')
    sizes=raw['length']
    ends=np.r_[0,sizes.cumsum()]
    delta=raw['delta']
    rng=np.random.default_rng(42)
    ids=rng.choice(ids,256 if dev else 3000,replace=False)
    coords=data['xy']
    norm_mean=data['mean']
    norm_std=data['std']
    h=torch.tensor(data['h'][ids]).cuda()
    ck=torch.load('checkpoint.pt',weights_only=True)
    if not dev and ck['epoch']!=200:
        raise ValueError('Final comparison uses the declared training budget')
    net=model().eval()
    net.load_state_dict(ck['weights'])
    if '--mode-check' in sys.argv:
        if not dev: raise ValueError('Use development requests for the sampling-mode diagnostic')
        net.train('--notebook-mode' in sys.argv)
    beta=schedule()
    torch.manual_seed(42)
    generated=[]
    with torch.no_grad():
        for batch_number,batch in enumerate(h.split(32)):
            if '--mode-check' in sys.argv:
                torch.manual_seed(42+batch_number)
            z=torch.randn(len(batch),2,200,device='cuda')
            for step in reversed(range(0,500,5)):
                t=torch.full((len(z),),step,device='cuda')
                z=p_xt(z,net(z,t,batch),t,torch.full_like(t,max(-1,step-5)),beta,0.)
            generated.extend(z.transpose(1,2).cpu().numpy()*norm_std+norm_mean)
    lo,hi=data['lo'],data['hi']
    ref=[delta[ends[i]:ends[i+1]].cumsum(0)*1e-6 for i in ids]
    synth=[resample_trajectory(g,int(sizes[i])) for i,g in zip(ids,generated)]
    def counts(trips):
        cells=[grid(tr,lo,hi) for tr in trips]
        return np.bincount(np.concatenate(cells),minlength=256),np.bincount([int(c[0]*256+c[-1]) for c in cells],minlength=256**2),np.array([arc_length(tr) for tr in trips])
    a,b=counts(ref),counts(synth)
    train=np.flatnonzero(data['split']<8)
    lengths=[arc_length(delta[ends[i]:ends[i+1]].cumsum(0)*1e-6) for i in train]
    bins=np.unique(np.quantile(lengths,np.linspace(0,1,21)[1:-1]))
    lh=lambda n:np.bincount(np.digitize(n,bins),minlength=len(bins)+1)
    js=lambda x,y:float(jensenshannon(x,y)**2)
    top=lambda hist:set(np.argsort(-hist,kind='stable')[:20])
    result=dict(epoch=ck['epoch'],requests=len(ids),density_js=js(a[0],b[0]),trip_js=js(a[1],b[1]),length_js=js(lh(a[2]),lh(b[2])),top20_cell_f1=len(top(a[0])&top(b[0]))/20,generated_outside_fraction=float(np.any((np.concatenate(synth)<lo)|(np.concatenate(synth)>hi),axis=1).mean()))
    result['model']='unconditional' if '--unconditional' in sys.argv else 'conditional'
    # Issue 15 adds a pseudocount to each raw count before normalization.
    result['density_js_issue15']=js(a[0]+1e-6,b[0]+1e-6)
    result['trip_js_pseudocount_sensitivity']=js(a[1]+1e-6,b[1]+1e-6)
    result['request_index_sha256']=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
    # Training-only audit found that one cell holds 92.4% of observed points
    # under full min/max bounds. Keep that primary score and report a city-core
    # sensitivity on exactly the same trajectories, including an overflow cell.
    low,high=np.quantile(data['xy'][train].reshape(-1,2),[.01,.99],axis=0)
    def core_counts(trips):
        points=np.concatenate(trips)
        c=grid(points,low,high)
        c[np.any((points<low)|(points>high),axis=1)]=256
        return np.bincount(c,minlength=257)
    result['core_density_js_with_overflow']=js(core_counts(ref),core_counts(synth))
    if '--audit-lengths' in sys.argv:
        from porto_audit_lengths import audit
        dev_ids=np.flatnonzero(data['split']==8)
        dev_lengths=[arc_length(delta[ends[i]:ends[i+1]].cumsum(0)*1e-6) for i in dev_ids]
        report=audit(ref,synth,data['xy'][ids],generated,np.asarray(lengths),data['h'][ids],arc_length,dev_lengths)
        Path('length-audit.json').write_text(json.dumps(report,indent=2))
    Path('development.json' if dev else ('geolife-final.json' if 'duration' in raw.files else 'porto-final.json')).write_text(json.dumps(result,indent=2))
    print(result)


if __name__=='__main__':
    {'prepare':prepare,'train':train,'score':score}[sys.argv[1]]()
