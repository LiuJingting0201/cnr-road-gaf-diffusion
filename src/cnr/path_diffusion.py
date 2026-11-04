"""Diffusion over log nonnegative arc increments; route is an external map condition."""
import torch
from torch import nn
from cnr.models import TrajectoryDenoiser

class PathDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=TrajectoryDenoiser()
        self.backbone.input=nn.Conv1d(2,128,1)
        self.backbone.out=nn.Conv1d(128,1,1)
        self.route_encoder=nn.Sequential(nn.Conv1d(2,64,5,padding=2),nn.SiLU(),
            nn.Conv1d(64,128,5,stride=2,padding=2),nn.SiLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
    def forward(self,x,t,label,start,route):
        embedding=self.route_encoder(route)
        return self.backbone(x,t,label,start,embedding)

def decode(latent,mean,std,route,length,start):
    # Clamp defines the decoder, not a hidden post-generation rejection step.
    increments=torch.expm1((latent[:,0]*std+mean).clamp(0,5))
    progress=torch.cat((torch.zeros_like(increments[:,:1]),increments.cumsum(-1)),1)
    capped=torch.minimum(progress,length[:,None])
    u=capped/length[:,None]*(route.shape[1]-1)
    lo=u.long().clamp(0,route.shape[1]-2);w=(u-lo)[...,None]
    left=route.gather(1,lo[...,None].expand(-1,-1,2))
    right=route.gather(1,(lo+1)[...,None].expand(-1,-1,2))
    xy=left*(1-w)+right*w
    xy=xy+(1-capped/10).clamp_min(0)[...,None]*(start-route[:,0])[:,None]
    return xy,progress
