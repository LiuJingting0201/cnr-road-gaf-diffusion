"""Start-centered road conditions and a differentiable approximate distance field.

Raster fields are model inputs/penalties only. Exact vector-lane evaluation remains
separate; no directional, connectivity or traffic-signal guarantee is implied.
"""
import numpy as np
import torch
from torch import nn
from cnr.models import TrajectoryDenoiser


class RoadFields(nn.Module):
    def __init__(self,path):
        super().__init__()
        a=np.load(path)
        self.register_buffer('fields',torch.from_numpy(a['distance'])[:,None])
        self.register_buffer('origin',torch.tensor(a['origin'],dtype=torch.float32))
        h,w=a['distance'].shape[-2:]
        self.register_buffer('extent',torch.tensor([(w-1)*float(a['resolution']),(h-1)*float(a['resolution'])]))

    def distances(self,xy,labels):
        grid=(xy-self.origin)/self.extent*2-1
        result=torch.zeros(xy.shape[:-1],device=xy.device,dtype=xy.dtype)
        for c in range(6):
            mask=labels==c
            if mask.any():
                query=grid[mask].clamp(-1,1)[None]
                value=torch.nn.functional.grid_sample(self.fields[c:c+1],query,mode='bilinear',
                                                       padding_mode='border',align_corners=True)[0,0]
                result[mask]=value
        outside=torch.maximum(self.origin-xy,xy-(self.origin+self.extent)).clamp_min(0)
        return result+torch.linalg.vector_norm(outside,dim=-1)

    def patches(self,starts,labels,size=32,radius=256.):
        offset=torch.linspace(-radius,radius,size,device=starts.device)
        yy,xx=torch.meshgrid(offset,offset,indexing='ij')
        grid=starts[:,None,:]+torch.stack((xx.flatten(),yy.flatten()),-1)[None]
        distance=self.distances(grid,labels).reshape(-1,size,size)
        return torch.stack((distance.clamp(max=128)/128,(distance<=5).float()),1)


class RoadDenoiser(nn.Module):
    def __init__(self,use_map=True):
        super().__init__()
        self.use_map=use_map
        self.backbone=TrajectoryDenoiser()
        self.map_encoder=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),
                                       nn.Conv2d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
                                       nn.Conv2d(64,128,3,stride=2,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
                                       nn.AdaptiveAvgPool2d(1),nn.Flatten())
    def forward(self,x,t,label,start,patch):
        embedding=self.map_encoder(patch) if self.use_map else None
        return self.backbone(x,t,label,start,embedding)
