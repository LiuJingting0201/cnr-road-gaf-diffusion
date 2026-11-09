"""Constructed follow-up contracts without reading any reference outcomes."""
import copy
import numpy as np
import torch
from g39_utility import series,Evaluator,REGIMES
from g39_factorial import checks as factorial_checks,weights
from g39_mechanism import checks as marginal_checks
from g34_fixed_duration import self_test as fixed_checks


def checks():
    torch.set_num_threads(1)
    t=np.array([5.,5.25,5.5,5.75,6.]);xy=np.c_[2*(t-5),np.zeros(5)]
    s=series(t,xy)
    assert s.shape==(3,257) and np.allclose(s[2],2.) and np.allclose(s[0],np.linspace(0,2,257))
    for tt in [np.array([1.,1.,2.]),np.array([2.,1.,0.])]:
        try:series(tt,np.zeros((3,2)))
        except ValueError:pass
        else:raise AssertionError('Bad clock accepted')
    torch.manual_seed(3901);model=Evaluator();initial=copy.deepcopy(model.state_dict())
    x=torch.tensor(np.repeat(s[None],5,axis=0));y=torch.arange(5)
    loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward()
    assert torch.isfinite(loss) and model.input.weight.grad.abs().sum()>0
    for _ in REGIMES:
        torch.manual_seed(3901);other=Evaluator()
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in initial.items())
    assert len(REGIMES)==13 and len(set(REGIMES))==13
    marginal_checks();factorial_checks();fixed_checks()
    ri=np.repeat([0,1],8);di=np.tile(np.repeat([0,1],4),2)
    rr=weights(ri,di,[1.,0.],np.zeros((16,2)))
    assert np.isfinite(rr['FixedRouteT-S']).all() and np.isclose(rr['FixedRouteT-S'].sum(),1.)
    print('G39 follow-up checks: raw physical-speed input, rejected clocks, nonzero raw gradient, 13regime initialization pairing, zero-mass support, future-input rejection')


if __name__=='__main__':checks()
