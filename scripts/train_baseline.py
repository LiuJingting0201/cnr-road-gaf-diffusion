"""Seeded, self-contained first baselines. No road constraint is claimed here."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from cnr.models import GAFClassifier,TrajectoryDenoiser


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['gaf_classifier','raw_diffusion'],required=True)
    p.add_argument('--data',default='data/processed/most64')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-every',type=int,default=500)
    args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this training entry point')
    torch.set_num_threads(4)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((Path(args.data)/'dataset.json').read_text())
    arrays=np.load(Path(args.data)/'train.npz');val=np.load(Path(args.data)/'val.npz')
    mean=torch.tensor(meta['mean'],device='cuda',dtype=torch.float32)
    std=torch.tensor(meta['std'],device='cuda',dtype=torch.float32)
    loader=DataLoader(TensorDataset(torch.from_numpy(arrays['xy']),torch.from_numpy(arrays['labels'])),
                      batch_size=args.batch_size,shuffle=True,num_workers=0)
    classifier=args.kind=='gaf_classifier'
    model=(GAFClassifier() if classifier else TrajectoryDenoiser()).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    class_counts=np.bincount(arrays['labels'],minlength=6)
    if (class_counts==0).any():raise ValueError('Missing training class')
    weights=torch.tensor(1/np.sqrt(class_counts),device='cuda',dtype=torch.float32)
    weights/=weights.mean()
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    configuration=vars(args)|{'dataset':meta,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
                              'parameters':sum(p.numel() for p in model.parameters()),
                              'sampler':'DDIM eta=0; no x0 clipping','road_conditioned':False,
                              'classifier_representation':'four float GASF/GADF XY channels; not Hilbert/MTF reproduction'}
    (out/'config.json').write_text(json.dumps(configuration,indent=2))
    step=0;started=time.monotonic();best=-1
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        while step<args.steps:
            for xy,label in loader:
                xy,label=xy.cuda(),label.cuda()
                model.train();optim.zero_grad(set_to_none=True)
                if classifier:
                    loss=torch.nn.functional.cross_entropy(model(xy),label,weight=weights)
                else:
                    z=(xy-mean)/std
                    # The first observed position is a given condition, not a generated target.
                    target=z[:,1:,:].transpose(1,2)
                    noise=torch.randn_like(target);t=torch.randint(0,1000,(len(xy),),device='cuda')
                    a=alpha[t,None,None]
                    pred=model(a.sqrt()*target+(1-a).sqrt()*noise,t,label,z[:,0,:])
                    loss=(pred-noise).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();step+=1
                if step%50==0 or step==args.steps:
                    torch.cuda.synchronize()
                    record={'step':step,'loss':loss.item(),'elapsed_s':time.monotonic()-started}
                    if classifier and (step%args.eval_every==0 or step==args.steps):
                        record.update(evaluate_classifier(model,val))
                        if record['macro_f1']>best:
                            best=record['macro_f1'];torch.save(model.state_dict(),out/'best.pt')
                    log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                if step%args.eval_every==0 or step==args.steps:
                    torch.save({'model':model.state_dict(),'optimizer':optim.state_dict(),'step':step},out/'latest.pt')
                if step>=args.steps:break
    if not classifier:
        model.eval()
        with torch.no_grad():
            xy=torch.from_numpy(val['xy'][:96]).cuda();labels=torch.from_numpy(val['labels'][:96]).cuda()
            starts=(xy[:,0]-mean)/std
            sample=torch.randn(len(xy),2,meta['length']-1,device='cuda')
            times=torch.linspace(999,0,50).long().tolist()
            for i,t in enumerate(times):
                eps=model(sample,torch.full((len(xy),),t,device='cuda'),labels,starts)
                x0=(sample-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
                prev=alpha[times[i+1]] if i+1<len(times) else torch.tensor(1.,device='cuda')
                sample=prev.sqrt()*x0+(1-prev).sqrt()*eps
            generated=torch.cat([xy[:,:1],sample.transpose(1,2)*std+mean],1)
            if not torch.isfinite(generated).all():raise RuntimeError('Nonfinite generated sample')
            np.savez_compressed(out/'validation_samples.npz',xy=generated.cpu().numpy(),labels=labels.cpu().numpy(),starts=xy[:,0].cpu().numpy())
    (out/'complete.json').write_text(json.dumps({'steps':step,'elapsed_s':time.monotonic()-started,'status':'complete'}))


@torch.no_grad()
def evaluate_classifier(model,val):
    model.eval();cm=torch.zeros(6,6,dtype=torch.long)
    for i in range(0,len(val['xy']),256):
        x=torch.from_numpy(val['xy'][i:i+256]).cuda()
        y=torch.from_numpy(val['labels'][i:i+256])
        pred=model(x).argmax(1).cpu()
        cm+=torch.bincount(y*6+pred,minlength=36).reshape(6,6)
    c=cm.float();tp=c.diag();f1=2*tp/(c.sum(0)+c.sum(1)).clamp_min(1)
    return {'macro_f1':f1.mean().item(),'per_class_f1':f1.tolist(),'confusion':cm.tolist(),'accuracy':(tp.sum()/c.sum()).item()}


if __name__=='__main__':main()
