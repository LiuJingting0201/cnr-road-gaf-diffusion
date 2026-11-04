"""Fixed-budget downstream utility: train from scratch, no selector checkpoint reuse."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from cnr.representations import SequenceClassifier

class GRUClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.GRU(4,64,num_layers=2,batch_first=True,bidirectional=True)
        self.head=torch.nn.Linear(128,6)
    def forward(self,x):
        h,_=self.rnn(x.transpose(1,2).contiguous())
        return self.head(h.mean(1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--arch',choices=['tcn','gru'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=1150);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    root=Path('data/processed/augmentation-e5');meta=json.loads((root/'dataset.json').read_text())
    out=Path('runs/augmentation-e5')/f'{a.arch}-{a.group}-seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    tr=np.load(root/a.group/'train.npz');va=np.load(root/'val.npz');stats=meta['stats']
    def encode(x):
        x=torch.from_numpy(x).cuda();delta=torch.diff(x,dim=1,prepend=x[:,:1])
        def z(v,name):
            mean=torch.tensor(stats[name+'_mean'],device='cuda');std=torch.tensor(stats[name+'_std'],device='cuda')
            return (v-mean)/std
        return torch.cat((z(x,'xy'),z(delta,'delta')),-1).transpose(1,2).contiguous()
    x=encode(tr['xy']);v=encode(va['xy']);y=torch.from_numpy(tr['labels']).cuda();vy=torch.from_numpy(va['labels']).cuda()
    model=(SequenceClassifier(4) if a.arch=='tcn' else GRUClassifier()).cuda()
    weights=torch.tensor(meta['real_class_counts'],dtype=torch.float32,device='cuda').rsqrt();weights/=weights.mean()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    config=vars(a)|{'dataset':meta,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'optimizer':{'name':'AdamW','lr':2e-4,'weight_decay':1e-4},'batch_size':64,'draws':a.steps*64,
        'sampling':'Uniform with replacement, exactly 64 examples every step for all groups',
        'scope':'Fixed final-step real validation utility, no checkpoint selection. Frozen real-only normalization and class weights. Independent initializations 100/101/102; no pretrained selector reuse.'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    gen=torch.Generator(device='cuda').manual_seed(a.seed+1000);beg=time.monotonic()
    @torch.no_grad()
    def evaluate():
        model.eval();probs=torch.cat([model(v[i:i+64]).softmax(-1) for i in range(0,len(v),64)])
        pred=probs.argmax(-1);cm=torch.bincount(vy*6+pred,minlength=36).reshape(6,6).float()
        f1=2*cm.diag()/(cm.sum(0)+cm.sum(1)).clamp_min(1)
        return {'accuracy':float((pred==vy).float().mean()),'macro_f1':float(f1.mean()),'per_class_f1':f1.tolist(),'per_class_recall':(cm.diag()/cm.sum(1).clamp_min(1)).tolist(),'confusion':cm.int().tolist()},probs.cpu().numpy()
    with (out/'metrics.jsonl').open('x',buffering=1) as log:
        for step in range(1,a.steps+1):
            ix=torch.randint(len(x),(64,),device='cuda',generator=gen);model.train();opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[ix]),y[ix],weight=weights)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite downstream loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step==1 or step%250==0 or step==a.steps:
                metrics,probs=evaluate();record={'step':step,'loss':float(loss.detach()),'elapsed_s':time.monotonic()-beg,**metrics}
                log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
    torch.save(model.state_dict(),out/'final.pt')
    np.savez_compressed(out/'validation.npz',probabilities=probs,labels=va['labels'],ids=va['ids'])
    (out/'complete.json').write_text(json.dumps({'complete':True,'steps':a.steps,'final_validation':metrics,'elapsed_s':time.monotonic()-beg},indent=2))

if __name__=='__main__':main()
