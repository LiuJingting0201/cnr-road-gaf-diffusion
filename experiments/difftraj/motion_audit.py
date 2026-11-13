"""Compare teacher-forced clean predictions and free-running DDIM at fixed noise levels."""
import json
from pathlib import Path
import torch
import numpy as np

def audit(xy,x,mu,sd,dt,v,train,abar,beta,predict,p_xt):
    ids=torch.where(~train)[0]
    i=ids[np.random.default_rng(173).choice(len(ids),96,replace=False)]
    def speed(p,idx):
        return p.diff().norm(dim=1)/dt[idx,None]
    def stats(p,idx):
        s=speed(p,idx)
        return [float(s.median()),float(s.mean()),float((s>40).float().mean()),float(p.diff().norm(dim=1).median())]
    target=xy[ids]
    restored=x[ids]*sd+mu
    res=dict(n=len(i),roundtrip_position_max_m=float((restored-target).abs().max()),roundtrip_speed_max_mps=float((speed(restored,ids)-speed(target,ids)).abs().max()),chord_arc_median=[float(speed(target,ids).median()),float(v[ids].median())],real=stats(xy[i],i),levels={})
    torch.manual_seed(174)
    noise=torch.randn_like(x[i])
    z=noise.clone()
    for step in reversed(range(0,500,5)):
        t=torch.full((len(i),),step,device=x.device)
        a=abar[step]
        pred=predict(z,t,i)
        def clean(state,e):
            return (state-(1-a).sqrt()*e)/a.sqrt()*sd+mu
        if step in (495,400,300,200,100,50,10,0):
            noisy=a.sqrt()*x[i]+(1-a).sqrt()*noise
            teacher=predict(noisy,t,i)
            res['levels'][step]=dict(teacher=stats(clean(noisy,teacher),i),free=stats(clean(z,pred),i),oracle=stats(clean(noisy,noise),i),noise_mse=float((teacher-noise).square().mean()))
        z=p_xt(z,pred,t,torch.full_like(t,max(-1,step-5)),beta,0.)
    p=z*sd+mu
    s=speed(p,i)
    res['final']=stats(p,i)
    res['final_mean_by_dt_quartile']=[float(s[j].mean()) for j in dt[i].argsort().chunk(4)]
    res['dt_quartiles']=np.quantile(dt[i].cpu().numpy(),[0,.25,.5,.75,1]).tolist()
    res['final_at_reference_slow_median']=float(s[speed(xy[i],i)<1].median())
    res['fields']=['median speed m/s','mean speed m/s','fraction above 40 m/s','median displacement m']
    Path('motion-audit.json').write_text(json.dumps(res,indent=2))
