"""E3 variable-length map-path diffusion; save an unfiltered candidate pool."""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser,decode

p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/route512')
p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=20000)
p.add_argument('--seed',type=int,default=42);p.add_argument('--samples-per-class',type=int,default=32)
p.add_argument('--checkpoint');p.add_argument('--repeats',type=int,default=1)
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
if a.checkpoint:
 ema.load_state_dict(torch.load(a.checkpoint,map_location='cuda',weights_only=True)['ema'])
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
cfg=vars(a)|{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,
 'log_increment_mean':mu,'log_increment_std':sd,'dataset':meta,'decoder':'exp(log increments clipped [0,5])-1; cumulative capped at route length; start offset fades over 10m',
 'data_hashes':{s:hashlib.sha256((d/f'{s}.npz').read_bytes()).hexdigest() for s in ['train','val']},
 'guidance':'none; inherited GAF selection is a separate future same-pool evaluation',
 'scope':'moving nonpedestrian subset; route planner plus conditional motion diffusion; not learned end-to-end route diffusion'}
(out/'config.json').write_text(json.dumps(cfg,indent=2))
generator=torch.Generator(device='cuda').manual_seed(a.seed+1000);order=torch.randperm(len(target),device='cuda',generator=generator);offset=0
beg=time.monotonic()
with (out/'metrics.jsonl').open('w',buffering=1) as f:
 for step in range(1,1 if a.checkpoint else a.steps+1):
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

# All candidates are retained. No GAF model is loaded in the generator.
selected=np.concatenate([np.flatnonzero(va['labels']==c)[np.linspace(0,np.sum(va['labels']==c)-1,min(a.samples_per_class,np.sum(va['labels']==c))).astype(int)] for c in np.unique(va['labels'])])
condition_ids=np.repeat(np.arange(len(selected)),a.repeats)
selected=np.repeat(selected,a.repeats)
all_xy=[];all_progress=[]
times=torch.linspace(999,0,100).long().tolist()
torch.manual_seed(2026)
for batch_start in range(0,len(selected),32):
 ix=selected[batch_start:batch_start+32]
 route=to(va['route'][ix]);length=to(va['route_length'][ix]);start=to(va['xy'][ix,0]);lab=torch.as_tensor(va['labels'][ix],device='cuda')
 condition=((route[:,::8]-start[:,None])/128).transpose(1,2)
 sample=torch.randn(len(ix),1,int(meta['length'])-1,device='cuda')
 with torch.no_grad():
  for j,t in enumerate(times):
   eps=ema(sample,torch.full((len(ix),),t,device='cuda'),lab,(start-mean)/std,condition)
   clean=(sample-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
   prev=alpha[times[j+1]] if j+1<len(times) else 1.
   sample=prev**.5*clean+(1-prev)**.5*eps
  xy,progress=decode(sample,mu,sd,route,length,start)
 if not torch.isfinite(xy).all():raise RuntimeError('Nonfinite sample')
 all_xy.append(xy.cpu().numpy());all_progress.append(progress.cpu().numpy())
xy=np.concatenate(all_xy);progress=np.concatenate(all_progress)
np.savez_compressed(out/'candidates.npz',xy=xy,labels=va['labels'][selected],source_index=va['source_index'][selected],condition_id=condition_ids,progress=progress,route_length=va['route_length'][selected])
report={'complete':True,'scope':'training and unguided sampling complete; topology, dynamics and selection evaluation pending','steps':0 if a.checkpoint else a.steps,'samples':len(xy),'points':int(meta['length']),'candidates_per_condition':a.repeats,'route_end_capped_fraction':float(np.mean(progress[:,-1]>=va['route_length'][selected])),'elapsed_s':time.monotonic()-beg}
(out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
