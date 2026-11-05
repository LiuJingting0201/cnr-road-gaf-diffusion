"""Frozen development assets. Validation futures are held outside the sampler."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from cnr.original_gaf import trajectory_sequence, encode_frozen
from cnr.representations import RepresentationClassifier

def tensor(x):
    return torch.as_tensor(x, device='cuda', dtype=torch.float32)

def assets():
    cfg=json.loads(Path('runs/long-path-e3/seed42/config.json').read_text())
    meta=cfg['dataset']['source_protocol']
    pool=np.load('runs/route-diverse-e4r/seed42/candidates.npz')
    _,ix=np.unique(pool['condition_id'],return_index=True)
    return cfg,meta,{k:pool[k][ix] for k in pool.files}

class Scores:
    def __init__(self,meta):
        self.meta=meta
        self.cnn=resnet18(weights=None)
        self.cnn.fc=torch.nn.Linear(512,6)
        self.cnn.load_state_dict(torch.load('runs/gaf-duration-e2/length512-seed42/final.pt',weights_only=True))
        self.cnn.cuda().eval()
        self.auditors=[]
        for seed in [43,44]:
            p=Path(f'runs/long-selection-e4/tcn-seed{seed}')
            m=RepresentationClassifier('xy_motion_tcn',json.loads((p/'config.json').read_text())['stats'])
            m.load_state_dict(torch.load(p/'latest.pt',weights_only=True)['model'])
            self.auditors.append(m.cuda().eval())
    @torch.no_grad()
    def image(self,xy):
        m=self.meta
        seq=[];ok=[]
        for x in xy:
            try:
                h,_=trajectory_sequence(x,np.array(m['xy_min']),np.array(m['xy_max']),eps=m['hilbert_eps'])
                ok.append(True)
            except ValueError:
                h=np.zeros(224);ok.append(False)
            seq.append(h)
        im,_=encode_frozen(seq,np.array(m['channel_min']),np.array(m['channel_max']),batch=16)
        pred=[]
        for block in np.array_split(im,max(1,(len(im)+31)//32)):
            rgb=tensor(block).permute(0,3,1,2)/255
            rgb=(rgb-tensor([.485,.456,.406])[None,:,None,None])/tensor([.229,.224,.225])[None,:,None,None]
            pred.extend(self.cnn(rgb).softmax(-1).cpu().numpy())
        return np.asarray(pred),np.array(ok),im
    @torch.no_grad()
    def raw(self,xy):
        return np.mean([m(tensor(xy)).softmax(-1).cpu().numpy() for m in self.auditors],0)

def reference(cfg,data):
    windows=np.load(Path(cfg['dataset']['source_directory'])/'val.npz')
    assert np.array_equal(windows['labels'][data['source_index']],data['labels'])
    return windows['xy'][data['source_index']]
