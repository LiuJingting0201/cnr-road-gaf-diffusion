"""Matched FP32 global-batch DDP throughput on actual Porto training inputs."""
import os,time,json,contextlib
import numpy as np
import torch
from porto import model,schedule
from torch.nn.parallel import DistributedDataParallel as DDP
rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
torch.distributed.init_process_group('nccl')
d=np.load('prepared.npz'); sel=d['split']<8
x=torch.tensor(((d['xy'][sel]-d['mean'])/d['std']).astype('f')).cuda().transpose(1,2)
h=torch.tensor(d['h'][sel]).cuda()
alpha=(1-schedule()).cumprod(0)
for micro,unused in [(32,True),(32,False),(128,False),(256,False),(32,True)]:
    torch.manual_seed(42)
    net=DDP(model(),device_ids=[rank],find_unused_parameters=unused)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-4)
    torch.manual_seed(42+rank)
    times=[]
    torch.cuda.reset_peak_memory_stats()
    for j in range(14):
        ids=torch.randint(len(x),(256,),device='cuda')
        torch.distributed.barrier();torch.cuda.synchronize();start=time.monotonic()
        opt.zero_grad()
        for k,ix in enumerate(ids.split(micro)):
            with net.no_sync() if (k+1)*micro<256 else contextlib.nullcontext():
                t=torch.randint(500,(len(ix)//2+1,),device='cuda');t=torch.cat([t,499-t])[:len(ix)]
                a=alpha[t,None,None];e=torch.randn_like(x[ix])
                loss=(net(a.sqrt()*x[ix]+(1-a).sqrt()*e,t,h[ix])-e).square().mean()
                (loss*(len(ix)/256)).backward()
                # Original implementation reads one loss scalar per microbatch.
                value=float(loss.detach())
        opt.step();torch.cuda.synchronize();torch.distributed.barrier()
        times.append(time.monotonic()-start)
    if rank==0:
        print(json.dumps(dict(micro=micro,unused=unused,median_s=float(np.median(times[4:])),samples_per_s=float(1024/np.median(times[4:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)
    del net,opt
torch.distributed.destroy_process_group()
