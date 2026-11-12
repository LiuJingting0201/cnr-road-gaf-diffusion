"""Fixed development noise-loss and atomic checkpoints for the stabilized Porto fits."""
import os
import numpy as np
import torch

def save(state,path):
    tmp=path+'.tmp'
    torch.save(state,tmp)
    os.replace(tmp,path)

class Validation:
    def __init__(self,data,alpha):
        ids=np.flatnonzero(data['split']==8)
        if not len(ids):
            raise ValueError('Development trips are required for checkpoint selection')
        ids=np.random.default_rng(142).permutation(ids)[:512]
        self.x=torch.tensor(((data['xy'][ids]-data['mean'])/data['std']).astype('f')).cuda().transpose(1,2)
        self.h=torch.tensor(data['h'][ids]).cuda()
        generator=torch.Generator(device='cuda').manual_seed(142)
        self.t=(torch.arange(len(ids),device='cuda')*500//len(ids)).long()
        self.e=torch.randn(self.x.shape,device='cuda',generator=generator)
        a=alpha[self.t,None,None]
        self.z=a.sqrt()*self.x+(1-a).sqrt()*self.e
    @torch.no_grad()
    def measure(self,model):
        model.eval()
        errors=[]
        for z,t,h,e in zip(self.z.split(32),self.t.split(32),self.h.split(32),self.e.split(32)):
            errors.append((model(z,t,h)-e).square().mean((1,2)))
        value=torch.cat(errors).mean().item()
        model.train()
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite development loss')
        return value
