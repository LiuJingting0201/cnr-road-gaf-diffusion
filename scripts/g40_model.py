"""Sampleable, generation-trained GAF conditions; no discrete-encoder gradient."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from g33_model import MotionDenoiser
from g39_raw import RawEncoder

ARMS = ('B', 'F-SV', 'J-S', 'J-V', 'J-SV', 'J-RAW', 'J-SH', 'C')


def views(arm):
    return {'B': (), 'J-S': ('spatial',), 'J-V': ('speed',)}.get(arm, ('spatial', 'speed'))


def reader(kind, view):
    if kind == 'raw':
        return RawEncoder(2 if view == 'spatial' else 1)
    from torchvision.models import resnet18
    model = resnet18(weights=None); model.fc = nn.Linear(512, 5)
    return model


def split_reader(model, kind):
    if kind == 'raw':
        prefix = nn.Sequential(model.stem, *list(model.blocks.children())[:6])
        tail = nn.Sequential(*list(model.blocks.children())[6:])
    else:
        prefix = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                               model.layer1, model.layer2, model.layer3)
        tail = model.layer4
    return prefix, tail, model.fc


class CodeEncoders(nn.Module):
    def __init__(self, arm, source_states=None):
        super().__init__(); self.arm = arm
        self.kind = 'raw' if arm == 'J-RAW' else 'gaf'
        self.tails = nn.ModuleDict(); self.heads = nn.ModuleDict(); self.codes = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            for view in ('spatial', 'speed'):
                model = reader(self.kind, view)
                if source_states is not None: model.load_state_dict(source_states[view], strict=True)
                _, tail, fc = split_reader(model, self.kind)
                self.tails[view] = tail; self.heads[view] = fc
                self.codes[view] = nn.Sequential(nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 32), nn.LayerNorm(32))
        for v in ('spatial', 'speed'):
            active = v in views(arm); joint = arm.startswith('J-') and active
            self.tails[v].requires_grad_(joint); self.heads[v].requires_grad_(joint)
            self.codes[v].requires_grad_(active)
        # Frozen running statistics even when the last residual stage is updated.
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)): module.eval()
        return self

    def forward(self, inputs, labels, centers=None):
        code = []; logits = {}
        reference = next(self.parameters())
        for v in ('spatial', 'speed'):
            if v not in views(self.arm):
                code.append(reference.new_zeros((len(labels), 32))); continue
            if self.arm == 'C': h = centers[v][labels]
            else:
                h = self.tails[v](inputs[v]); h = h.mean(tuple(range(2, h.ndim)))
            logits[v] = self.heads[v](h); code.append(self.codes[v](h))
        return torch.cat(code, dim=1), logits


class ConditionedMotion(MotionDenoiser):
    def __init__(self, base=64):
        super().__init__(base=base)
        with torch.random.fork_rng(devices=[]):
            self.style = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
            nn.init.zeros_(self.style[-1].weight); nn.init.zeros_(self.style[-1].bias)

    def load_base(self, state):
        result = self.load_state_dict(state, strict=False)
        assert set(result.missing_keys) == {'style.0.weight', 'style.0.bias', 'style.2.weight', 'style.2.bias'}
        assert not result.unexpected_keys

    def forward(self, x, t, labels, context, route, clock_features, mask, code=None):
        if x.shape[-1] % 4: raise ValueError('Pad sequence to a multiple of four')
        frequency = torch.exp(torch.arange(64, device=x.device)*(-math.log(10000)/63))
        angle = t[:, None]*frequency[None, :]
        condition = (self.time(torch.cat([angle.sin(), angle.cos()], 1))+
                     self.label(labels)+self.context(context)+self.route(route))
        if code is not None: condition = condition+self.style(code)
        h0 = self.blocks(self.enc0, self.input(torch.cat([x*mask, clock_features*mask], 1))*mask, condition, mask)
        h1, m1 = self.pool(h0, mask)
        h1 = self.blocks(self.enc1, self.down1(h1)*m1, condition, m1)
        h2, m2 = self.pool(h1, m1)
        h2 = self.blocks(self.middle, self.down2(h2)*m2, condition, m2)
        h1 = self.blocks(self.dec1, self.up1(torch.cat([F.interpolate(h2, scale_factor=2, mode='nearest'), h1], 1))*m1, condition, m1)
        h0 = self.blocks(self.dec0, self.up0(torch.cat([F.interpolate(h1, scale_factor=2, mode='nearest'), h0], 1))*mask, condition, mask)
        return self.out(h0)*mask


def predict(model, batch, x, t, code):
    return model(x, t, batch['labels'], batch['context'], batch['route'], batch['clock_features'], batch['mask'], code=code)


class PriorBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.norm = nn.LayerNorm(256)
        self.net = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
    def forward(self, x): return x+self.net(self.norm(x))


class CodePrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(64, 256); self.label = nn.Embedding(5, 256)
        self.context = nn.Sequential(nn.Linear(5, 256), nn.SiLU(), nn.Linear(256, 256))
        self.route = nn.Sequential(nn.Linear(258, 256), nn.SiLU(), nn.Linear(256, 256))
        self.time = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        self.blocks = nn.Sequential(*[PriorBlock() for _ in range(4)])
        self.out = nn.Linear(256, 64)
    def forward(self, z, t, labels, context, route):
        freq = torch.exp(torch.arange(64, device=z.device)*(-math.log(10000)/63))
        angle = t[:, None]*freq[None]
        h = self.input(z)+self.label(labels)+self.context(context)+self.route(route.flatten(1))
        h = h+self.time(torch.cat([angle.sin(), angle.cos()], 1))
        return self.out(self.blocks(h))


def checks():
    from g33_model import clock, pack_motion, predict as base_predict
    import numpy as np
    torch.set_num_threads(1); torch.manual_seed(42); base = MotionDenoiser(base=16)
    state = torch.get_rng_state().clone(); torch.manual_seed(42); model = ConditionedMotion(base=16)
    assert torch.equal(state, torch.get_rng_state())
    model.load_base(base.state_dict())
    t = clock(4.125, 'native_0_25s'); path = np.c_[np.linspace(0, 100, 129), np.zeros(129)]
    batch = pack_motion([np.ones(len(t)-1)], [t], [1], [path], [100.], [t[-1]],
                        {'start_mean': [0,0], 'start_sd': [1,1]}, 'cpu')
    code = torch.ones(1, 64); stamp = torch.tensor([400])
    expected = base_predict(base, batch, batch['x'], stamp)
    assert torch.equal(expected, predict(model, batch, batch['x'], stamp, code))
    ema = copy.deepcopy(model); predict(model,batch,batch['x'],stamp,code).sum().backward()
    assert model.style[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in ema.parameters())
    with torch.no_grad(): model.style[-1].weight.fill_(.01)
    a = predict(model,batch,batch['x'],stamp,code)
    b = predict(model,batch,batch['x'],stamp,torch.zeros_like(code)); assert not torch.equal(a,b)
    expanded = dict(batch)
    for k in ('x','mask','clock_features'): expanded[k] = F.pad(batch[k], (0,12))
    padded = predict(model,expanded,expanded['x'],stamp,code)
    assert torch.allclose(a,padded[...,:a.shape[-1]],atol=4e-6,rtol=4e-6)
    print('G40: exact zero-adapter backbone/RNG, style gradient/use, padding and EMA ownership')


if __name__ == '__main__': checks()
