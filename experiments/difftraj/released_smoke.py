"""CPU checkpoint diagnostic with the notebook's 100 deterministic DDIM transitions."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
root=Path(sys.argv[1])
sys.path.insert(0,str(root))
from utils.Traj_UNet import Guide_UNet
from utils.config import args
from utils.utils import resample_trajectory
torch.manual_seed(42)
torch.set_num_threads(4)
cfg=SimpleNamespace(**{k:SimpleNamespace(**v) for k,v in args.items()})
net=Guide_UNet(cfg)
net.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True))
net.eval()
heads=torch.tensor(np.load(root/'heads.npy')[:16],dtype=torch.float32)
x=torch.randn(len(heads),2,200)
abar=(1-torch.linspace(.0001,.05,500)).cumprod(0)
with torch.no_grad():
    for step in reversed(range(0,500,5)):
        eps=net(x,torch.full((len(x),),step),heads)
        clean=(x-(1-abar[step]).sqrt()*eps)/abar[step].sqrt()
        prev=abar[step-5] if step else torch.tensor(1.)
        x=prev.sqrt()*clean+(1-prev).sqrt()*eps
lengths=(heads[:,3]*107.2487482839+292.30299616).int()
trips=[resample_trajectory(p.T,int(n))*[.0215106194,.0189193207]+[104.07596303,30.68085491] for p,n in zip(x.numpy(),lengths)]
points=np.concatenate(trips)
assert np.isfinite(points).all()
print(json.dumps(dict(trips=len(trips),points=len(points),mean_lon_lat=points.mean(0).tolist(),seed=42,steps=100,mode='eval',request_selection='first 16 public headers'),indent=2))
