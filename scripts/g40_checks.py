"""Mechanism and sampling contract tests before any prospective outcomes."""
import copy
import numpy as np
import torch
from g33_model import MotionDenoiser, clock, pack_motion, sample
from g40_model import ConditionedMotion, CodeEncoders, predict, checks
from g40_prior import condition_arrays, sample_motion, sample_code


def main():
    checks();torch.set_num_threads(1);torch.manual_seed(40000)
    path=np.c_[np.linspace(0,10,129),np.zeros(129)].astype(np.float32)
    times=clock(4.125,'native_0_25s')
    stats={'start_mean':[0,0],'start_sd':[1,1],'motion_mean':.5,'motion_sd':.2,
           'duration_prior':{'minimum_duration_s':.25,'maximum_mean_speed_mps':55.,
               'classes':[{'intercept':np.log(4.125),'slope':0.,'center_log_length':0.,'residuals':[0.]}]*5}}
    batch=pack_motion([np.ones(len(times)-1)],[times],[1],[path],[10.],[times[-1]],stats,'cpu')
    context,route=condition_arrays(path[None],np.array([10.]),np.array([times[-1]]),stats)
    assert np.array_equal(context,batch['context'].numpy()) and np.array_equal(route,batch['route'].numpy())
    for arm in ('J-SV','F-SV','J-RAW'):
        encoder=CodeEncoders(arm).train();ema=copy.deepcopy(encoder).eval()
        shape=(1,256,8) if arm=='J-RAW' else (1,256,14,14)
        inputs={v:torch.randn(shape) for v in ('spatial','speed')}
        before={n:b.clone() for n,b in encoder.named_buffers()}
        code,_=encoder(inputs,batch['labels']);model=ConditionedMotion(base=16)
        with torch.no_grad():model.style[-1].weight.normal_(0,.01)
        # No classifier CE here: the generator alone must reach active encoders.
        prediction=predict(model,batch,batch['x'],torch.tensor([400]),code)
        prediction.square().sum().backward()
        assert all(torch.equal(b,before[n]) for n,b in encoder.named_buffers()),'BN statistics changed'
        for v in ('spatial','speed'):
            assert sum(p.grad.abs().sum().item() for p in encoder.codes[v].parameters() if p.grad is not None)>0
            grads=[p.grad for p in encoder.tails[v].parameters() if p.grad is not None]
            assert (sum(g.abs().sum().item() for g in grads)>0) if arm.startswith('J-') else not grads
        assert all(p.grad is None for p in ema.parameters())
        del encoder,ema,model
    torch.manual_seed(40002);base=MotionDenoiser(base=16).eval();model=ConditionedMotion(base=16).eval();model.load_base(base.state_dict())
    request={'id':'fixture','label':1,'length':10.,'path':path}
    expected=sample(base,[request],stats,'native_0_25s',42,device='cpu')[0]
    state=torch.get_rng_state().clone()
    actual=sample_motion(model,{'kind':'zero'},request,stats,42,device='cpu')
    assert torch.equal(state,torch.get_rng_state())
    for k in ('time','progress'):assert np.array_equal(expected[k],actual[k]),k
    assert expected['duration']==actual['duration']
    for field in ('duration','true_duration','code','target_xy','progress','gaf'):
        try:sample_motion(model,{'kind':'zero'},{**request,field:0},stats,42,device='cpu')
        except ValueError:pass
        else:raise AssertionError(field)
    try:sample_motion(model,{'kind':'zero'},request,stats,42,device='cpu',code_override=torch.ones(1,64))
    except ValueError:pass
    else:raise AssertionError('Future code override accepted')
    class ZeroPrior(torch.nn.Module):
        def forward(self,z,*args):return torch.zeros_like(z)
    prior={'kind':'diffusion','model':ZeroPrior(),'mask':torch.ones(64),'sd':torch.ones(64),'mean':torch.zeros(64)}
    state=torch.get_rng_state().clone();a=sample_code(prior,batch,'fixture',42);b=sample_code(prior,batch,'fixture',42)
    assert torch.equal(a,b) and not torch.equal(a,sample_code(prior,batch,'fixture-other',42))
    assert torch.equal(state,torch.get_rng_state()) and torch.isfinite(a).all()
    print('G40: generator-only encoder gradients; frozen BN; matched base sampler; prior RNG; future-input contract PASS')


if __name__=='__main__':main()
