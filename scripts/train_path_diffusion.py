"""Map-path conditional diffusion, followed by matched sampling guidance sweeps."""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser,decode
from cnr.representations import RepresentationClassifier

p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/route512')
p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=20000)
p.add_argument('--seed',type=int,default=42);p.add_argument('--samples-per-class',type=int,default=32)
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed)
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);d=Path(a.data)
tr=np.load(d/'train.npz');va=np.load(d/'val.npz');meta=json.loads((d/'dataset.json').read_text())
to=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
loginc=np.log1p(np.maximum(0,np.diff(tr['progress'],axis=1)))
mu=float(loginc.mean());sd=max(float(loginc.std()),1e-6)
target=to((loginc-mu)/sd)[:,None];labels=torch.as_tensor(tr['labels'],device='cuda')
starts=to(tr['xy'][:,0]);mean=to(meta['mean']);std=to(meta['std'])
routes=to(tr['route']);route_condition=((routes[:,::8]-starts[:,None])/128).transpose(1,2)
model=PathDenoiser().cuda();ema=copy.deepcopy(model).eval()
for x in ema.parameters():x.requires_grad_(False)
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
cfg=vars(a)|{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'log_increment_mean':mu,'log_increment_std':sd,'dataset':meta,'decoder':'exp(log increments clipped [0,5])-1; cumulative capped at route length; start offset fades over 10m',
 'data_hashes':{s:hashlib.sha256((d/f'{s}.npz').read_bytes()).hexdigest() for s in ['train','val']},
 'guidance':'gradient of clean-trajectory teacher logprob for t<100; per-sample gradient norm capped at 1; clean latent update weight*gradient; fixed DDIM epsilon',
 'scope':'moving nonpedestrian subset; route planner plus conditional motion diffusion; not learned end-to-end route diffusion'}
(out/'config.json').write_text(json.dumps(cfg,indent=2))
generator=torch.Generator(device='cuda').manual_seed(a.seed+1000);order=torch.randperm(len(target),device='cuda',generator=generator);offset=0
beg=time.monotonic()
with (out/'metrics.jsonl').open('w',buffering=1) as f:
 for step in range(1,a.steps+1):
  if offset>=len(target):order=torch.randperm(len(target),device='cuda',generator=generator);offset=0
  ix=order[offset:offset+128];offset+=128;x=target[ix];t=torch.randint(1000,(len(ix),),device='cuda');noise=torch.randn_like(x);aa=alpha[t,None,None]
  eps=model(aa.sqrt()*x+(1-aa).sqrt()*noise,t,labels[ix],(starts[ix]-mean)/std,route_condition[ix]);loss=(eps-noise).square().mean()
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  with torch.no_grad():
   for e,m in zip(ema.parameters(),model.parameters()):e.lerp_(m,.001)
  if step==1 or step%1000==0 or step==a.steps:
   row={'step':step,'loss':loss.item(),'elapsed_s':time.monotonic()-beg};f.write(json.dumps(row)+'\n');print(row,flush=True)
   torch.save({'model':model.state_dict(),'ema':ema.state_dict()},out/'latest.pt')
selected=np.concatenate([np.flatnonzero(va['labels']==c)[np.linspace(0,np.sum(va['labels']==c)-1,min(a.samples_per_class,np.sum(va['labels']==c))).astype(int)] for c in np.unique(va['labels'])])
route=to(va['route'][selected]);length=to(va['route_length'][selected]);start=to(va['xy'][selected,0]);lab=torch.as_tensor(va['labels'][selected],device='cuda')
condition=((route[:,::8]-start[:,None])/128).transpose(1,2)
teachers={};records={}
for kind,path in [('gaf','runs/representation-r1/delta_gaf-seed42'),('tcn','runs/representation-r1-delta/delta_tcn-seed42')]:
 tp=Path(path);c=json.loads((tp/'config.json').read_text());m=RepresentationClassifier(c['variant'],c['stats']).cuda().eval();m.load_state_dict(torch.load(tp/'best.pt',weights_only=True,map_location='cuda'))
 for q in m.parameters():q.requires_grad_(False)
 teachers[kind]=m;records[kind]={'path':path,'sha256':hashlib.sha256((tp/'best.pt').read_bytes()).hexdigest()}
(out/'teachers.json').write_text(json.dumps(records,indent=2))
sweep=[('none',None,0),('gaf_01','gaf',.1),('gaf_03','gaf',.3),('tcn_01','tcn',.1),('tcn_03','tcn',.3),('shuffled_gaf_01','gaf',.1)]
results=[];times=torch.linspace(999,0,100).long().tolist()
for name,kind,weight in sweep:
 torch.manual_seed(2026);sample=torch.randn(len(lab),1,63,device='cuda')
 g=torch.Generator(device='cuda').manual_seed(3026);goal=lab[torch.randperm(len(lab),device='cuda',generator=g)] if name.startswith('shuffled') else lab
 for j,t in enumerate(times):
  with torch.no_grad():
   eps=ema(sample,torch.full((len(lab),),t,device='cuda'),lab,(start-mean)/std,condition)
   clean=(sample-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
  if kind and t<100:
   clean=clean.detach().requires_grad_(True);xy,_=decode(clean,mu,sd,route,length,start)
   score=teachers[kind](xy).log_softmax(-1).gather(1,goal[:,None]).sum()
   grad=torch.autograd.grad(score,clean)[0];norm=grad.flatten(1).norm(dim=1)[:,None,None].clamp_min(1)
   clean=(clean+weight*grad/norm).detach()
  with torch.no_grad():
   prev=alpha[times[j+1]] if j+1<len(times) else 1.
   sample=prev**.5*clean+(1-prev)**.5*eps
 with torch.no_grad():xy,progress=decode(sample,mu,sd,route,length,start)
 if not torch.isfinite(xy).all():raise RuntimeError('Nonfinite sample')
 np.savez_compressed(out/f'{name}.npz',xy=xy.cpu().numpy(),labels=lab.cpu().numpy(),source_index=va['source_index'][selected],progress=progress.cpu().numpy(),route_length=length.cpu().numpy())
 results.append({'variant':name,'route_end_capped_fraction':float((progress[:,-1]>=length).float().mean()),'samples':len(lab)})
(out/'complete.json').write_text(json.dumps({'status':'complete','steps':a.steps,'sweeps':results,'elapsed_s':time.monotonic()-beg},indent=2))
