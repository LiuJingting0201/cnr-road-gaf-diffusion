"""Compare held-out terminal training objectives with trajectory statistics on saved bus draws."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,wasserstein_distance
import torch
from torch.nn import functional as F
from terminal_bus_train import persistent,DATA
from five_class_evaluate import read_observations
from score_terminal_speed import entered_spells,describe_spells

root=Path('/workspace/cnr-experiments/terminal-bus-v1')
index=np.load(DATA/'validation/index.npz')
rows=np.flatnonzero(index['classes']=='bus')
ids=index['ids'][rows]
n=np.load(DATA/'normalization.npz')
ref=np.load(DATA/'validation/traj.npy')[rows]*n['std']+n['mean']
raw=read_observations(DATA,ids)
dur=np.array([len(p)-1 for p in raw])
dt=dur/199
def speed(x):
    return np.linalg.norm(np.diff(x,axis=1),axis=2)/dt[:,None]
def stats(v):
    r=speed(ref)
    e=entered_spells(v,dur)
    low=v<1
    return dict(w1=wasserstein_distance(v.ravel(),r.ravel()),slow_share=100*low.mean(),
                entry=np.mean(((~low[:,:-2])&low[:,1:-1]&low[:,2:]).sum(1)*600/dur),
                spells=describe_spells(e))
rng=np.random.default_rng(1116)
rv=speed(ref)
shuffled=rng.permuted(rv,axis=1)
def ft(x):return torch.as_tensor(x,dtype=torch.float64)
tw=persistent(ft(rv),ft(dt))
def objectives(v):
    t=ft(v)
    sl=F.smooth_l1_loss(t.sort(1).values.log1p(),ft(rv).sort(1).values.log1p(),reduction='none').mean(1)
    wl=F.smooth_l1_loss(persistent(t,ft(dt)),tw,reduction='none')
    return sl,wl
out=dict(ref=stats(rv),shuffle=stats(shuffled))
out['shuffle']['speed_term'],out['shuffle']['window_term']=[z.mean().item() for z in objectives(shuffled)]
native_spells=[]
for p in raw:
    v=np.linalg.norm(np.diff(p,axis=0),axis=1)
    native_spells.extend(entered_spells(v[None],np.array([len(v)])))
out['native_1s_coordinate_spells']=describe_spells(np.array(native_spells))
for seed in (42,43):
    terms={}
    per_arm={}
    for arm in ('A','pointwise','distribution'):
        d=np.load(root/f'{arm}-{seed}.npz')
        assert np.array_equal(ids,d['ids'])
        xy=d['xy_m'].astype('d')
        if arm=='A':base=xy
        dist=np.linalg.norm(xy-base,axis=2)
        v=speed(xy)
        lv,lw=objectives(v)
        loss=(lv+lw).numpy()
        tripw=np.abs(np.sort(v,axis=1)-np.sort(rv,axis=1)).mean(1)
        slowerr=np.abs((v<1).mean(1)-(rv<1).mean(1))*100
        spell_error=np.full(len(v),np.nan)
        for i in range(len(v)):
            actual=entered_spells(rv[i:i+1],dur[i:i+1])
            generated=entered_spells(v[i:i+1],dur[i:i+1])
            if len(actual) and len(generated):
                spell_error[i]=wasserstein_distance(actual,generated)
        terms[arm]=[loss,tripw,slowerr,spell_error]
        # Compare direct objective components in metres on the same clean predictions.
        coord=ft(xy).requires_grad_()
        vt=coord.diff(dim=1).norm(dim=2)/ft(dt)[:,None]
        gl,gw=objectives(vt)
        grads=[torch.autograd.grad(l.sum(),coord,retain_graph=True)[0].norm().item() for l in (gl,gw)]
        per_arm[arm]=dict(sorted_speed_term=lv.mean().item(),window_term=lw.mean().item(),
                          window_to_sorted_output_gradient_norm=grads[1]/grads[0],
                          mean_within_trip_W1=tripw.mean(),pooled_W1=stats(v)['w1'],
                          shift_median_m=np.median(dist),shift_p90_m=np.quantile(dist,.9))
    for a in ('A','pointwise'):
        x,y,z,w=np.array(terms[a])-np.array(terms['distribution'])
        per_arm['delta_'+a+'_minus_distribution']=dict(
            training_objective_improvement=float(x.mean()),
            evaluable_spell_pairs=int(np.isfinite(w).sum()),mean_spell_W1_improvement_s=float(np.nanmean(w)),
            rank_corr_loss_vs_spell_W1=float(spearmanr(x[np.isfinite(w)],w[np.isfinite(w)]).statistic),within_trip_W1_improvement=float(y.mean()),
            slow_share_abs_error_improvement_pp=float(z.mean()),fraction_trips_speed_improved=float((y>0).mean()),
            rank_corr_loss_vs_speed=float(spearmanr(x,y).statistic),
            rank_corr_loss_vs_slow_share_error=float(spearmanr(x,z).statistic))
    out[seed]=per_arm
print(json.dumps(out,indent=2,allow_nan=False))
