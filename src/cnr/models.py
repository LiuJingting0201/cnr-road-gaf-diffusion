"""Initial executable baselines; not a claimed reproduction of the group classifier."""
import math
import torch
from torch import nn


def gaf_channels(xy):
    # Four float channels: GASF(x), GASF(y), GADF(x), GADF(y).
    # Per-window scaling intentionally removes absolute location; no uint8 quantization.
    lo, hi = xy.amin(1, keepdim=True), xy.amax(1, keepdim=True)
    u = ((xy-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1).transpose(1,2)
    v = (1-u.square()).clamp_min(1e-7).sqrt()
    gasf = u.unsqueeze(-1)*u.unsqueeze(-2)-v.unsqueeze(-1)*v.unsqueeze(-2)
    gadf = v.unsqueeze(-1)*u.unsqueeze(-2)-u.unsqueeze(-1)*v.unsqueeze(-2)
    return torch.cat([gasf,gadf],1)


class GAFClassifier(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),
                                 nn.Conv2d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
                                 nn.Conv2d(64,128,3,stride=2,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
                                 nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,classes))
    def forward(self,xy):
        return self.net(gaf_channels(xy))


class Residual(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.norm=nn.GroupNorm(8,width)
        self.conv=nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.cond=nn.Linear(width,width)
        self.out=nn.Conv1d(width,width,1)
    def forward(self,x,c):
        y=self.conv(torch.nn.functional.silu(self.norm(x)))+self.cond(c)[:,:,None]
        return x+self.out(torch.nn.functional.silu(y))


class TrajectoryDenoiser(nn.Module):
    def __init__(self,width=128,classes=6):
        super().__init__()
        self.width=width
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.label=nn.Embedding(classes,width)
        self.start=nn.Linear(2,width)
        self.input=nn.Conv1d(3,width,1)
        self.blocks=nn.ModuleList([Residual(width,2**(i%5)) for i in range(10)])
        self.out=nn.Conv1d(width,2,1)
    def forward(self,x,t,label,start,map_embedding=None):
        freq=torch.exp(torch.arange(self.width//2,device=x.device)*(-math.log(10000)/(self.width//2-1)))
        angles=t[:,None]*freq[None,:]
        c=self.time(torch.cat([angles.sin(),angles.cos()],1))+self.label(label)+self.start(start)
        if map_embedding is not None:c=c+map_embedding
        position=torch.linspace(-1,1,x.shape[-1],device=x.device)[None,None,:].expand(len(x),1,-1)
        h=self.input(torch.cat([x,position],1))
        for block in self.blocks:h=block(h,c)
        return self.out(h)
