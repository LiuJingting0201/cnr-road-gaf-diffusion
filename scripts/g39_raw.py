"""Raw-series teacher controls; no angular images or pretrained image weights."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from g33_train import digest,save
from g33_teachers import metrics
from g35c_teachers import load_dataset


class Residual(nn.Module):
    def __init__(self, a,b,stride=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(a,b,3,stride,padding=1,bias=False),nn.BatchNorm1d(b),nn.ReLU(),
                               nn.Conv1d(b,b,3,padding=1,bias=False),nn.BatchNorm1d(b))
        self.skip=nn.Identity() if a==b and stride==1 else nn.Sequential(nn.Conv1d(a,b,1,stride,bias=False),nn.BatchNorm1d(b))
    def forward(self,x):return F.relu(self.net(x)+self.skip(x))


class RawEncoder(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(channels,64,7,2,3,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(3,2,1))
        blocks=[];a=64
        for b in (64,128,256,512):
            blocks.extend([Residual(a,b,1 if a==b else 2),Residual(b,b)]);a=b
        self.blocks=nn.Sequential(*blocks);self.fc=nn.Linear(512,5)
    def features(self,x):return self.blocks(self.stem(x)).mean(-1)
    def forward(self,x):return self.fc(self.features(x))


def prepare(data,cache,out):
    out.mkdir(parents=True,exist_ok=False)
    try:
        receipt,raw=load_dataset(data)
        cr=json.loads((cache/'complete.json').read_text())
        assert cr['complete'] and cr['sources']['data_sha256']==digest(data/'dataset.npz')
        assert digest(cache/'index.npz')==cr['artifacts']['index.npz']['sha256']
        with np.load(cache/'index.npz',allow_pickle=False) as f:
            for k in ('ids','parent_ids','labels','partition'):assert np.array_equal(f[k],raw[k])
            speed=f['velocity'].copy()[:,None,:]/35.
        xy=np.empty((len(raw['ids']),2,224),np.float32)
        profiles=np.empty((len(raw['ids']),257),np.float32)
        for i,(a,b) in enumerate(zip(raw['offset'][:-1],raw['offset'][1:])):
            t=raw['time'][a:b];q=np.linspace(0,t[-1],224)
            xy[i]=np.stack([np.interp(q,t,raw['xy'][a:b,j]) for j in range(2)])
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(raw['xy'][a:b],axis=0),axis=1))]
            profiles[i]=np.interp(np.linspace(0,t[-1],257),t,arc/max(arc[-1],1e-12))
        train=raw['partition']=='train';mean=xy[train].mean((0,2));sd=np.maximum(xy[train].std((0,2)),1.)
        xy=(xy-mean[None,:,None])/sd[None,:,None]
        records={k:raw[k] for k in ('ids','parent_ids','labels','partition')}
        np.savez_compressed(out/'raw.npz',**records,spatial=xy,speed=speed,profiles=profiles)
        save(out/'complete.json',{'complete':True,'rows':len(xy),'train':int(train.sum()),
             'raw_sha256':digest(out/'raw.npz'),'source_dataset_sha256':digest(data/'dataset.npz'),
             'cache_index_sha256':digest(cache/'index.npz'),'xy_mean':mean.tolist(),'xy_sd':sd.tolist(),
             'normalizer_train_only':True,'spatial':'224 linear relative-time XY samples; train global coordinate normalization',
             'speed':'identical128 cached profiles /35, no GAF transform'})
    except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise


def train(cache,out,view):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
        torch.backends.cudnn.benchmark=False
        receipt=json.loads((cache/'complete.json').read_text())
        assert receipt['complete'] and digest(cache/'raw.npz')==receipt['raw_sha256']
        with np.load(cache/'raw.npz',allow_pickle=False) as f:
            rows={k:f[k].copy() for k in ('ids','parent_ids','labels','partition')};values=f[view].copy()
        ii={s:np.flatnonzero(rows['partition']==s) for s in ('train','val')}
        assert not set(rows['parent_ids'][ii['train']]) & set(rows['parent_ids'][ii['val']])
        model=RawEncoder(values.shape[1]).cuda()
        x=torch.as_tensor(values,device='cuda');y=torch.as_tensor(rows['labels'],device='cuda')
        weights=np.bincount(rows['labels'][ii['train']],minlength=5).astype(float)**-.5;weights/=weights.mean()
        w=torch.as_tensor(weights,dtype=torch.float32,device='cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);rng=np.random.default_rng(42)
        save(out/'config.json',{'view':view,'epochs':50,'batch':64,'seed':42,'parameters':sum(p.numel() for p in model.parameters()),
            'pretraining':None,'cache_sha256':digest(cache/'complete.json'),'source_sha256':digest(__file__),
            'class_weights':weights.tolist(),'label_budget_and_epochs_match_GAF':True,'architecture_and_pretraining_match_GAF':False})
        updates=0
        with (out/'training.jsonl').open('w',buffering=1) as f:
            for epoch in range(1,51):
                model.train();order=rng.permutation(ii['train']);losses=[]
                for k in range(0,len(order),64):
                    ix=order[k:k+64];loss=F.cross_entropy(model(x[ix]),y[ix],weight=w)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite RAW loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();updates+=1
                    losses.append((float(loss.detach()),len(ix)))
                row={'epoch':epoch,'updates':updates,'loss':sum(v*n for v,n in losses)/sum(n for _,n in losses),'seconds':time.monotonic()-start}
                f.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        torch.save(model.state_dict(),out/'model.pt');model.eval();scores={}
        for split,ix in ii.items():
            features=[];logits=[]
            with torch.no_grad():
                for k in range(0,len(ix),64):
                    h=model.features(x[ix[k:k+64]]);features.append(h.cpu().numpy());logits.append(model.fc(h).cpu().numpy())
            z=np.concatenate(logits);features=np.concatenate(features);scores[split]=metrics(rows['labels'][ix],z)
            np.savez_compressed(out/f'{split}-predictions.npz',**{k:rows[k][ix] for k in ('ids','parent_ids','labels')},features=features,logits=z)
        save(out/'complete.json',{'complete':True,'view':view,'seconds':time.monotonic()-start,'updates':updates,**scores,
             'artifacts':{n:{'sha256':digest(out/n),'bytes':(out/n).stat().st_size} for n in ('model.pt','train-predictions.npz','val-predictions.npz')},
             'limitation':'RAW scratch 1D teachers vs ImageNet initialized 2D GAF; not encoding-only isolation'})
    except BaseException as e:save(out/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--data',type=Path);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--view',choices=['spatial','speed']);a=p.parse_args()
    if a.action=='prepare':prepare(a.data,a.cache,a.output)
    else:train(a.cache,a.output,a.view)
