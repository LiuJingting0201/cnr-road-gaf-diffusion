"""Regression checks for the new hidden-state supervision path."""
import copy
import json
import torch
from cnr.path_diffusion import PathDenoiser
from g28_train import Denoiser

torch.set_num_threads(1)
torch.manual_seed(42)
base = PathDenoiser(classes=8).cuda()
base_rng = torch.get_rng_state()
torch.manual_seed(42)
model = Denoiser().cuda()
assert torch.equal(base_rng, torch.get_rng_state())
for key,value in base.state_dict().items():
    assert torch.equal(value, model.state_dict()[key]), key
averaged = copy.deepcopy(model)
x = torch.randn(2,1,257,device='cuda')
t = torch.tensor([200,700],device='cuda')
y = torch.tensor([1,3],device='cuda')
s = torch.zeros(2,2,device='cuda')
r = torch.randn(2,2,129,device='cuda')
out = model(x,t,y,s,r)
assert torch.equal(out,base(x,t,y,s,r))
feature = model.hidden
averaged(x+1,t,y,s,r)
assert model.hidden is feature and averaged.hidden is not feature
target = torch.nn.functional.normalize(torch.randn(2,512,device='cuda'),dim=1)
loss = 1 - (model.projected('speed') * target).sum(1).mean()
loss.backward()
assert model.backbone.blocks[4].conv.weight.grad is not None
assert model.backbone.blocks[4].conv.weight.grad.norm() > 0
print(json.dumps({'same_backbone_and_rng':True,'same_forward':True,'ema_hook_independent':True,'gradient_path':True}))
