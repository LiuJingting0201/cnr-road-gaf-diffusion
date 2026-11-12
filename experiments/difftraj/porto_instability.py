"""Instrumented replay of the Porto fast run and a matched clipping intervention."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from porto import model, schedule

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
torch.manual_seed(42)
net=DistributedDataParallel(model(),device_ids=[rank])
opt=torch.optim.AdamW(net.parameters(),lr=2e-4)
torch.manual_seed(42+rank)
d=np.load('prepared.npz')
keep=d['split']<8
x=torch.tensor(((d['xy'][keep]-d['mean'])/d['std']).astype('f')).cuda().transpose(1,2)
h=torch.tensor(d['h'][keep]).cuda()
alpha=(1-schedule()).cumprod(0)
branch=sys.argv[1]
start=1
if branch=='clip':
    state=torch.load('epoch140.pt',map_location='cpu',weights_only=True)
    net.module.load_state_dict(state['weights'])
    opt.load_state_dict(state['optimizer'])
    torch.cuda.set_rng_state(state['rng'][rank])
    start=141
with open(f'{branch}-steps.jsonl','a',buffering=1) as fp:
    for epoch in range(start,166):
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(42+epoch))
        for batch,group in enumerate(order.split(1024)):
            n=len(group)//4*4
            if n==0: continue
            idx=group[:n].reshape(4,-1)[rank].cuda()
            opt.zero_grad()
            t=torch.randint(500,(len(idx)//2+1,),device='cuda')
            t=torch.cat((t,499-t))[:len(idx)]
            a=alpha[t,None,None]
            e=torch.randn_like(x[idx])
            z=a.sqrt()*x[idx]+(1-a).sqrt()*e
            pred=net(z,t,h[idx])
            loss=(pred-e).square().mean()
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0 if branch=='clip' else float('inf'),error_if_nonfinite=True)
            diag=torch.stack([loss.detach(),pred.detach().abs().max(),x[idx].abs().max(),h[idx,1:6].abs().max()])
            dist.all_reduce(diag,op=dist.ReduceOp.MAX)
            if rank==0:
                print(json.dumps(dict(epoch=epoch,batch=batch,max_rank_loss=diag[0].item(),grad_norm=norm.item(),prediction_peak=diag[1].item(),xy_peak=diag[2].item(),head_peak=diag[3].item())),file=fp)
            opt.step()
        if epoch==140:
            states=[None]*4
            dist.all_gather_object(states,torch.cuda.get_rng_state().cpu())
            if rank==0: torch.save(dict(weights=net.module.state_dict(),optimizer=opt.state_dict(),rng=states),'epoch140.pt')
        if rank==0 and epoch in (140,145,146,147,150,165):
            torch.save(dict(weights=net.module.state_dict(),epoch=epoch),f'{branch}-{epoch}.pt')
dist.destroy_process_group()
