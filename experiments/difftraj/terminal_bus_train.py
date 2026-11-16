"""Fine-tune the released output convolution on the terminal state of free DDIM draws."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from five_class_train import make_model
from five_class_evaluate import stratify
from utils.utils import p_xt

ROOT=Path('/workspace/cnr-experiments/five-class-difftraj-200k-v1')
DATA=Path('/workspace/cnr-experiments/difftraj-five-40k-v1')
DEVICE='cuda'
def tensor(x):
    return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)

def network(arm,seed):
    model=make_model(arm).eval()
    ck=torch.load(ROOT/f'runs/bus-{arm}-{seed}/final.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['weights'])
    return model.requires_grad_(False)

def advance(model,z,h,steps):
    beta=torch.linspace(.0001,.05,500,device=DEVICE)
    y=torch.ones(len(z),dtype=torch.long,device=DEVICE)
    for step in steps:
        t=torch.full_like(y,step)
        z=p_xt(z,model(z,t,h,y),t,torch.full_like(t,max(step-5,-1)),beta,0.)
    return z

def rollout(model,z,h,steps):
    with torch.no_grad():
        return torch.cat([advance(model,z[i:i+64],h[i:i+64],steps) for i in range(0,len(z),64)])

def inputs(split):
    part=DATA/split
    with np.load(part/'index.npz') as d:
        ids,classes=d['ids'],d['classes']
    rows=np.flatnonzero(classes=='bus')
    if split=='train':
        rng=np.random.default_rng(9316)
        rows=rng.choice(rows,2048,replace=False)
        noise=rng.standard_normal((len(rows),2,200),dtype='f')
    else:
        pool,_=stratify(classes)
        previous=pool[np.isin(classes[pool],['bicycle','bus'])]
        missing=np.setdiff1d(np.flatnonzero(np.isin(classes,['bicycle','bus'])),previous)
        order=np.r_[previous,missing]
        rng=np.random.default_rng(42)
        old=rng.standard_normal((len(pool),2,200),dtype='f')[np.isin(pool,previous)]
        extra=rng.standard_normal((len(missing),2,200),dtype='f')
        noise=np.concatenate([old,extra])[np.argsort(order)][classes[np.sort(order)]=='bus']
        assert np.array_equal(np.sort(order)[classes[np.sort(order)]=='bus'],rows)
    return ids[rows],tensor(np.load(part/'head.npy')[rows]),tensor(noise),tensor(np.load(part/'traj.npy',mmap_mode='r')[rows]).transpose(1,2)

def persistent(speed,dt):
    widths=torch.ceil(20/dt).long().clamp(1,199)
    out=torch.zeros_like(dt)
    for w in widths.unique().tolist():
        ix=widths==w
        peaks=F.max_pool1d(speed[ix,None],w,stride=1)
        out[ix]=torch.sigmoid((1-peaks)/.25).mean((1,2))
    return out

def main(seed,mode):
    train_ids,h,n,target=inputs('train')
    ids,val,initial,_=inputs('validation')
    norm=np.load(DATA/'normalization.npz')
    sd=tensor(norm['std'])[None,:,None]
    dt=(h[:,2]*norm['header_std'][1]+norm['header_mean'][1])/199
    model=network('A',seed)
    tail=(10,5,0)
    prefix=range(495,10,-5)
    def save(tag,z):
        xy=(z.transpose(1,2).cpu().numpy()*norm['std']+norm['mean']).astype('f')
        assert np.isfinite(xy).all()
        np.savez_compressed(f'{tag}-{seed}.npz',ids=ids,xy_m=xy,noise_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest())
    states=rollout(model,n,h,prefix)
    original=rollout(model,states,h,tail)
    vs=rollout(model,initial,val,prefix)
    if mode=='distribution':
        save('A',rollout(model,vs,val,tail))
        save('B',rollout(network('B',seed),initial,val,range(495,-1,-5)))
    active=model.unet.conv_out.requires_grad_(True)
    opt=torch.optim.AdamW(active.parameters(),lr=.0001,weight_decay=0)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,128,eta_min=.00001)
    order=np.random.default_rng(116)
    history=[]
    for _ in range(2):
        for batch in np.array_split(order.permutation(len(h)),64):
            pred=advance(model,states[batch],h[batch],tail)
            vel=lambda z:(z.diff(dim=2)*sd).norm(dim=1)/dt[batch,None]
            vp,vt=vel(pred),vel(target[batch])
            if mode=='pointwise':
                motion=F.smooth_l1_loss(vp.log1p(),vt.log1p())
            else:
                motion=F.smooth_l1_loss(vp.sort(1).values.log1p(),vt.sort(1).values.log1p())
                motion+=F.smooth_l1_loss(persistent(vp,dt[batch]),persistent(vt,dt[batch]))
            displacement=(pred-original[batch])*sd
            anchor=0.05*((displacement/10).square().mean()+(displacement[:,:,[0,-1]]/10).square().mean())
            loss=motion+anchor
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(active.parameters(),1,error_if_nonfinite=True)
            opt.step();sched.step()
            history.append([float(motion.detach()),float(anchor.detach())])
    save(mode,rollout(model,vs,val,tail))
    torch.save(active.state_dict(),f'{mode}-{seed}.pt')
    Path(f'fit-{mode}-{seed}.json').write_text(json.dumps(dict(first=history[0],last=history[-1],updates=len(history),train_ids=hashlib.sha256('\n'.join(train_ids).encode()).hexdigest())))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('seed',type=int)
    p.add_argument('mode',choices=('pointwise','distribution'))
    a=p.parse_args();main(a.seed,a.mode)
