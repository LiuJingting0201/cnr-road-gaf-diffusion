"""Validation-only representation study with identical minibatch schedules."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from cnr.representations import VARIANTS, RepresentationClassifier, training_stats


@torch.no_grad()
def evaluate(net, x, y):
    net.eval()
    logits=torch.cat([net(x[i:i+256]) for i in range(0,len(x),256)])
    prob=logits.softmax(-1);pred=prob.argmax(-1)
    cm=torch.bincount(y*6+pred,minlength=36).reshape(6,6).float()
    f1=2*cm.diag()/(cm.sum(0)+cm.sum(1)).clamp_min(1)
    report={'macro_f1':f1.mean().item(),'accuracy':(cm.diag().sum()/cm.sum()).item(),
            'per_class_f1':f1.tolist(),'confusion':cm.int().tolist(),
            'cross_entropy':torch.nn.functional.cross_entropy(logits,y).item()}
    return report,prob.cpu().numpy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--data',default='data/processed/most64')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--eval-every',type=int,default=500)
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.eval_every)<=0: raise ValueError('Positive budgets required')
    torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    data=Path(a.data);train=np.load(data/'train.npz');val=np.load(data/'val.npz')
    stats=training_stats(train['xy'])
    model=RepresentationClassifier(a.variant,stats).cuda()
    params=sum(p.numel() for p in model.parameters())
    cfg=vars(a)|{'stats':stats,'parameters':params,'gpu':torch.cuda.get_device_name(),
                 'torch':torch.__version__,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 'dataset':json.loads((data/'dataset.json').read_text()),
                 'hashes':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['train.npz','val.npz']},
                 'selection':'best validation macro-F1; no test-set access',
                 'protocol':'fixed windows; train-only bounds; no squeeze/bounce; Hilbert RGB fixed channel range uint8',
                 'input_scope':'XY-derived only; fixed dt; not historical classifier reproduction'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    started=time.monotonic()
    with torch.no_grad():
        features=[]
        for split in [train,val]:
            chunks=[]
            for i in range(0,len(split['xy']),512):
                chunks.append(model.encode(torch.from_numpy(split['xy'][i:i+512]).cuda()))
            features.append(torch.cat(chunks));del chunks
        x,xv=features
    y=torch.from_numpy(train['labels']).cuda();yv=torch.from_numpy(val['labels']).cuda()
    counts=torch.bincount(y,minlength=6).float()
    if (counts==0).any():raise ValueError('Missing class')
    weights=counts.rsqrt();weights/=weights.mean()
    optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    generator=torch.Generator(device='cuda').manual_seed(a.seed+1000)
    order=torch.randperm(len(x),generator=generator,device='cuda');offset=0
    best=-1.;best_step=0;best_report=None
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.steps+1):
            if offset>=len(x):
                order=torch.randperm(len(x),generator=generator,device='cuda');offset=0
            idx=order[offset:offset+a.batch_size];offset+=a.batch_size
            model.train();optim.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model.classifier(x[idx]),y[idx],weight=weights)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            if step==1 or step%a.eval_every==0 or step==a.steps:
                report,prob=evaluate(model.classifier,xv,yv)
                torch.cuda.synchronize()
                row={'step':step,'train_loss':loss.item(),'elapsed_s':time.monotonic()-started,**report}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                if report['macro_f1']>best:
                    best=report['macro_f1'];best_step=step;best_report=report
                    torch.save(model.state_dict(),out/'best.pt')
                    np.savez_compressed(out/'best_val_predictions.npz',probabilities=prob,labels=val['labels'])
    torch.save({'model':model.state_dict(),'optimizer':optim.state_dict(),'step':a.steps},out/'latest.pt')
    (out/'complete.json').write_text(json.dumps({'status':'complete','variant':a.variant,'seed':a.seed,
          'steps':a.steps,'best_step':best_step,'best_validation':best_report,'parameters':params,
          'elapsed_s':time.monotonic()-started,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2},indent=2))


if __name__=='__main__':main()
