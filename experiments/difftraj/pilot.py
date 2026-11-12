"""Four paired development arms using the author's full coordinate denoiser."""
import json
import os
import sys
from types import SimpleNamespace
import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn.functional as F
from utils.Traj_UNet import Guide_UNet
from utils.config import args
from utils.utils import p_xt

arm=sys.argv[1]
steps=int(sys.argv[2])
training_seed=int(os.environ.get("DIFFTRAJ_TRAINING_SEED","42"))
torch.manual_seed(training_seed)
d=np.load('pilot.npz')
def tensor(key):
    return torch.from_numpy(d[key]).cuda()
xy,v,h,dt,y,train=map(tensor,['xy','v','h','dt','y','train'])
mu=xy[train].mean((0,2))[None,:,None]
sd=xy[train].std((0,2))[None,:,None]
x=(xy-mu)/sd
ids=torch.where(train)[0]
config=SimpleNamespace(**{k:SimpleNamespace(**v) for k,v in args.items()})
net=Guide_UNet(config).cuda()
emb=torch.nn.Embedding(len(d['classes']),net.ch).cuda()
torch.nn.init.zeros_(emb.weight)
optim=torch.optim.AdamW([*net.parameters(),*emb.parameters()],lr=2e-4)
beta=torch.linspace(.0001,.05,500,device='cuda')
abar=(1-beta).cumprod(0)
def predict(z,t,i):
    cond=net.guide_emb(h[i])
    if 'C' in arm:
        cond=cond+emb(y[i])
    a=net.unet(z,t,cond)
    b=net.unet(z,t,net.place_emb(torch.zeros_like(h[i])))
    return a+3*(a-b)
if '--evaluate' in sys.argv:
    net.load_state_dict(torch.load(f'{arm}-{steps}.pt',weights_only=True))
    emb.load_state_dict(torch.load(f'{arm}-{steps}-class.pt',weights_only=True))
else:
    for step in range(steps):
        i=ids[torch.randint(len(ids),(16,),device='cuda')]
        t=torch.randint(500,(9,),device='cuda')
        t=torch.cat([t,499-t])[:16]
        a=abar[t,None,None]
        noise=torch.randn_like(x[i])
        z=a.sqrt()*x[i]+(1-a).sqrt()*noise
        pred=predict(z,t,i)
        loss=F.mse_loss(pred,noise)
        if 'V' in arm:
            clean=(z-(1-a).sqrt()*pred)/a.sqrt()
            vel=(clean.diff()*sd).norm(dim=1)/dt[i,None]
            # Log speed limits domination by high-speed errors; SNR weighting
            # suppresses unstable clean estimates near the noise endpoint.
            residual=F.smooth_l1_loss(vel.log1p(),v[i].clamp_min(0).log1p(),reduction='none')
            loss=loss+.05*(a[:,0]*residual).mean()
        if not loss.isfinite():
            raise FloatingPointError(f"nonfinite training loss at {step}")
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step%50==0: print(step,float(loss),flush=True)
    torch.save(net.state_dict(),f'{arm}-{steps}.pt')
    torch.save(emb.state_dict(),f'{arm}-{steps}-class.pt')
net.eval()
with torch.no_grad():
    torch.manual_seed(173)
    valid=torch.where(~train)[0]
    i=valid[torch.randperm(len(valid),device='cuda')[:128]]
    if '--all-validation' in sys.argv:
        i=valid
    extra='--remaining' in sys.argv
    if extra:
        # Exclude the original scored requests and the 32 reconstruction-diagnostic requests.
        used=torch.cat([i,valid[:32]])
        i=valid[~torch.isin(valid,used)]
    outs=[]
    for batch in i.split(16):
        z=torch.randn_like(x[batch])
        for t0 in reversed(range(0,500,5)):
            t=torch.full((len(batch),),t0,device='cuda')
            z=p_xt(z,predict(z,t,batch),t,torch.full_like(t,max(-1,t0-5)),beta,0.)
        outs.append(z*sd+mu)
    generated=torch.cat(outs)
    if '--save-draws' in sys.argv:
        np.savez_compressed(f'private-{arm}-{steps}.npz', generated=generated.cpu().numpy(), reference=xy[i].cpu().numpy(), ids=i.cpu().numpy(), labels=d['classes'][y[i].cpu().numpy()], dt=dt[i].cpu().numpy())
    def speeds(seq):
        return (seq.diff().norm(dim=1)/dt[i,None]).cpu().numpy()
    actual=speeds(xy[i]); synth=speeds(generated)
    score=lambda a,b:float(wasserstein_distance(a.ravel(),b.ravel()))
    summary=dict(arm=arm,training_seed=training_seed,updates=steps,train=len(ids),validation=len(valid),evaluated=len(i),speed_w1_mps=score(actual,synth),length_w1_m=score(actual.sum(1)*dt[i].cpu().numpy(),synth.sum(1)*dt[i].cpu().numpy()),classes={str(c):score(actual[m],synth[m]) for k,c in enumerate(d['classes']) if (m:=(y[i].cpu().numpy()==k)).any()})
    # Fix spatial bounds and length bins on training data, with an overflow cell.
    from scipy.spatial.distance import jensenshannon
    bounds = xy[train].amin((0,2)), xy[train].amax((0,2))
    def cells(seq):
        lo,hi=bounds
        q=((seq-lo[None,:,None])/(hi-lo)[None,:,None]*16).floor().long()
        outside=((q<0)|(q>15)).any(1)
        return (q[:,0]*16+q[:,1]).masked_fill(outside,256).cpu().numpy()
    def js(a,b,n):
        return float(jensenshannon(np.bincount(a.ravel(),minlength=n),np.bincount(b.ravel(),minlength=n))**2)
    ref,gen=cells(xy[i]),cells(generated)
    edges=np.unique(np.quantile(xy[train].diff().norm(dim=1).sum(1).cpu().numpy(),np.linspace(0,1,21)[1:-1]))
    lens=lambda seq:np.digitize(seq.diff().norm(dim=1).sum(1).cpu().numpy(),edges)
    summary.update(density_js=js(ref,gen,257),od_js=js(ref[:,0]*257+ref[:,-1],gen[:,0]*257+gen[:,-1],257**2),length_js=js(lens(xy[i]),lens(generated),len(edges)+1),outside_fraction=float((gen==256).mean()))
    summary['mean_class_speed_w1']=float(np.mean(list(summary['classes'].values())))
    summary['reference_speed_median_mps']=float(np.median(actual))
    summary['generated_speed_median_mps']=float(np.median(synth))
    summary['per_class_counts']={str(c):int((y[i]==k).sum()) for k,c in enumerate(d['classes'])}
    with open(f'{arm}-{steps}'+('-remaining' if extra else '')+'.json','w') as f: json.dump(summary,f,indent=2)
