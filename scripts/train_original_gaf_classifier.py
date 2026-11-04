"""New CNN on original Hilbert RGB; participant-classification baseline, not guidance."""
import argparse,hashlib,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from cnr.representations import ImageClassifier
p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/original-hilbert224');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(a.seed);root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
tr=np.load(root/'train.npz');va=np.load(root/'val.npz');meta=json.loads((root/'dataset.json').read_text())
model=ImageClassifier(3).cuda();tx=torch.from_numpy(tr['images']).cuda();vx=torch.from_numpy(va['images']).cuda();ty=torch.from_numpy(tr['labels']).cuda();vy=torch.from_numpy(va['labels']).cuda()
assert set(tr['ids']).isdisjoint(set(va['ids']))
counts=torch.bincount(ty,minlength=6).float();assert (counts>0).all();weights=counts.rsqrt();weights/=weights.mean()
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
config=vars(a)|{'dataset':meta,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'model':'New 3-channel CNN: Conv32/64/128 + GroupNorm/SiLU + global pooling + 6-class linear head; not a supplied group CNN','selection':'fixed final step is primary; validation best is secondary, no test evaluation','data_hashes':{s:hashlib.sha256((root/f'{s}.npz').read_bytes()).hexdigest() for s in ['train','val']}}
(out/'config.json').write_text(json.dumps(config,indent=2));gen=torch.Generator(device='cuda').manual_seed(a.seed+1000);order=torch.randperm(len(tx),device='cuda',generator=gen);offset=0;best=-1.;beg=time.monotonic()
def features(x):return x.permute(0,3,1,2).float()/127.5-1
@torch.no_grad()
def evaluate():
 model.eval();logits=torch.cat([model(features(vx[i:i+32])) for i in range(0,len(vx),32)]);pred=logits.argmax(-1);cm=torch.bincount(vy*6+pred,minlength=36).reshape(6,6).float();f1=2*cm.diag()/(cm.sum(0)+cm.sum(1)).clamp_min(1)
 return {'macro_f1':float(f1.mean()),'accuracy':float((pred==vy).float().mean()),'per_class_f1':f1.tolist(),'confusion':cm.int().tolist()},logits.softmax(-1).cpu().numpy()
with (out/'metrics.jsonl').open('w',buffering=1) as f:
 for step in range(1,a.steps+1):
  if offset>=len(tx):order=torch.randperm(len(tx),device='cuda',generator=gen);offset=0
  ix=order[offset:offset+a.batch_size];offset+=a.batch_size;model.train();opt.zero_grad(set_to_none=True)
  loss=torch.nn.functional.cross_entropy(model(features(tx[ix])),ty[ix],weight=weights)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step==1 or step%500==0 or step==a.steps:
   report,prob=evaluate();record={'step':step,'loss':float(loss),'elapsed_s':time.monotonic()-beg,**report};f.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
   if report['macro_f1']>best:best=report['macro_f1'];torch.save(model.state_dict(),out/'best.pt')
 torch.save(model.state_dict(),out/'final.pt');np.savez_compressed(out/'final_validation.npz',probabilities=prob,labels=va['labels'],ids=va['ids'])
(out/'complete.json').write_text(json.dumps({'complete':True,'steps':a.steps,'final_validation':report,'best_validation_macro_f1':best,'elapsed_s':time.monotonic()-beg},indent=2))
