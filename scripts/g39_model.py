"""Train-only frozen-feature heads; exported generator is unchanged G33."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from g33_model import MotionDenoiser, clock, pack_motion, predict

ARMS = ('B', 'S', 'V', 'SV', 'P', 'SH', 'RAW')


class AlignedDenoiser(MotionDenoiser):
    def __init__(self, base=64):
        super().__init__(base=base)
        # All arms have identical unused/used heads; preserve backbone/noise RNG.
        with torch.random.fork_rng(devices=[]):
            self.projections = nn.ModuleDict({v: nn.Sequential(nn.Linear(base*2, 128),
                nn.SiLU(), nn.Linear(128, 512)) for v in ('spatial', 'speed')})
        self.hidden = None
        self._weights = None
        self.middle[-1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        weights = self._weights
        self.hidden = (output*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12)

    def forward(self, x, t, labels, context, route, clock_features, mask):
        self._weights = F.avg_pool1d(clock_features[:, 2:3]*mask, 4)*4
        return super().forward(x, t, labels, context, route, clock_features, mask)

    def projected(self, view):
        return F.normalize(self.projections[view](self.hidden), dim=1)

    def generator_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('projections.')}


def alignment(model, arm, ix, labels, vectors, centers, permutation, alpha):
    views = {'S': ('spatial',), 'V': ('speed',), 'SV': ('spatial', 'speed'),
             'P': ('spatial', 'speed'), 'SH': ('spatial', 'speed'),
             'RAW': ('spatial', 'speed')}[arm]
    losses = []
    for v in views:
        target = centers[v][labels] if arm == 'P' else vectors[v][permutation[ix] if arm == 'SH' else ix]
        losses.append(alpha*(1-(model.projected(v)*target).sum(1)))
    return torch.stack(losses).mean(0)


def checks():
    torch.set_num_threads(1)
    torch.manual_seed(42); baseline = MotionDenoiser(base=16); rng = torch.get_rng_state()
    torch.manual_seed(42); m = AlignedDenoiser(base=16)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, m.generator_state()[k]) for k, v in baseline.state_dict().items())
    ema = copy.deepcopy(m).eval().requires_grad_(False)
    path = np.c_[np.linspace(0, 100, 129), np.zeros(129)]
    times = clock(2.125, 'native_0_25s')
    b = pack_motion([np.ones(len(times)-1)], [times], [0], [path], [100.], [times[-1]],
                    {'start_mean': [0,0], 'start_sd': [1,1]}, 'cpu')
    t = torch.tensor([400]); y = predict(m,b,b['x'],t); h = m.hidden.clone()
    z = predict(baseline,b,b['x'],t); assert torch.equal(y,z)
    bb = dict(b)
    for k in ('x','mask','clock_features'): bb[k] = F.pad(b[k],(0,12))
    yy = predict(m,bb,bb['x'],t)
    assert torch.allclose(y,yy[...,:y.shape[-1]],atol=3e-6,rtol=3e-6)
    assert torch.allclose(h,m.hidden,atol=3e-6,rtol=3e-6)
    before=m.hidden; predict(ema,b,b['x'],t)
    assert m.hidden is before and ema.hidden is not m.hidden
    target=F.normalize(torch.arange(512,dtype=torch.float32)[None],dim=1)
    loss=1-(m.projected('spatial')*target).sum();loss.backward()
    assert m.middle[-1].conv.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in ema.parameters())
    plain=MotionDenoiser(base=16);plain.load_state_dict(m.generator_state(),strict=True)
    print('G39 checks: exact backbone/RNG, padding, physical pooling, EMA ownership, auxiliary gradient, strict export')


if __name__ == '__main__': checks()
