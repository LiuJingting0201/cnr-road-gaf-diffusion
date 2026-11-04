"""E5 training-only motion synthesis on known training paths, never validation paths."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser, decode

torch.set_num_threads(4)
root=Path('runs/augmentation-e5/pool');root.mkdir(parents=True,exist_ok=False)
run=Path('runs/long-path-e3/seed42');cfg=json.loads((run/'config.json').read_text())
data=Path(cfg['data']);tr=np.load(data/'train.npz')
rawroot=Path(cfg['dataset']['source_directory']);raw=np.load(rawroot/'train.npz');validation=np.load(rawroot/'val.npz')
assert set(raw['ids']).isdisjoint(set(validation['ids']))
assert np.array_equal(tr['labels'],raw['labels'][tr['source_index']])
assert np.allclose(tr['xy'],raw['xy'][tr['source_index']])
assert hashlib.sha256((data/'train.npz').read_bytes()).hexdigest()==cfg['data_hashes']['train']
checkpoint=run/'latest.pt';model=PathDenoiser().cuda().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['ema'])
to=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
mean=to(cfg['dataset']['mean']);std=to(cfg['dataset']['std'])
alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
times=torch.linspace(999,0,100).long().tolist();torch.manual_seed(30261104)
indices=np.repeat(np.arange(len(tr['xy'])),8);xy_all=[];progress_all=[]
(root/'config.json').write_text(json.dumps({'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'diffusion_config':cfg,'source_split':'train','K':8,'conditions':len(tr['xy']),'noise_seed':30261104,'scope':'Known training-path conditional motion augmentation; not novel-route generalization. No validation/test conditions used.'},indent=2))
with torch.no_grad():
    for offset in range(0,len(indices),32):
        ix=indices[offset:offset+32]
        route=to(tr['route'][ix]);length=to(tr['route_length'][ix]);start=to(tr['xy'][ix,0]);label=torch.as_tensor(tr['labels'][ix],device='cuda')
        condition=((route[:,::8]-start[:,None])/128).transpose(1,2)
        sample=torch.randn(len(ix),1,int(cfg['dataset']['length'])-1,device='cuda')
        for j,t in enumerate(times):
            eps=model(sample,torch.full((len(ix),),t,device='cuda'),label,(start-mean)/std,condition)
            clean=(sample-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            prev=alpha[times[j+1]] if j+1<len(times) else 1.
            sample=prev**.5*clean+(1-prev)**.5*eps
        xy,progress=decode(sample,cfg['log_increment_mean'],cfg['log_increment_std'],route,length,start)
        assert torch.isfinite(xy).all()
        xy_all.append(xy.cpu().numpy());progress_all.append(progress.cpu().numpy())
        if offset%256==0:print(json.dumps({'generated':offset+len(ix),'total':len(indices)}),flush=True)
np.savez_compressed(root/'candidates.npz',xy=np.concatenate(xy_all),progress=np.concatenate(progress_all),labels=tr['labels'][indices],source_index=tr['source_index'][indices],condition_id=indices,route_length=tr['route_length'][indices],parent_ids=raw['ids'][tr['source_index'][indices]])
(root/'complete.json').write_text(json.dumps({'complete':True,'samples':len(indices),'conditions':len(tr['xy'])}))
