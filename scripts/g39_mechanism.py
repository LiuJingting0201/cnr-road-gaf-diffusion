"""Exposed G38 diagnostic: preserve route marginal while reweighting motion."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from g33_train import digest,save
import g38_evaluate as evaluation

METHODS=('R','Global-S','Global-V','Global-SV','FixedRoute-S','FixedRoute-V','FixedRoute-SV')


def preserve(base,groups,score):
    base=np.asarray(base,float);groups=np.asarray(groups);score=np.asarray(score,float)
    assert len(base)==len(groups)==len(score) and np.isclose(base.sum(),1.) and np.all(base>=0)
    out=np.zeros_like(base)
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g)
        if base[ix].sum()==0:continue
        w=base[ix]*np.exp(score[ix]-score[ix].max())
        out[ix]=base[ix].sum()*w/w.sum()
    assert np.isclose(out.sum(),1.)
    return out


def build(source,out):
    out.mkdir(parents=True,exist_ok=False)
    gen=json.loads((source/'complete.json').read_text());assert gen['complete']
    for n,h in gen['artifacts'].items():assert digest(source/n)==h
    rows=json.loads((source/'candidates.json').read_text())['rows']
    old=json.loads((source/'selections.json').read_text())['rows'];picks=[];largest=0.
    for rcontrol in [s for s in old if s['method']=='R']:
        rid=rcontrol['request_id'];rr=[r for r in rows if r['request_id']==rid]
        assert [r['candidate'] for r in rr]==list(range(len(rr)))
        oldmethods={s['method']:s for s in old if s['request_id']==rid}
        idx=np.array([r['route_index'] for r in rr]);prob=np.array([r['probabilities'] for r in rr]);scores=np.log(np.maximum(prob,1e-8))
        key=f'{gen["seed"]}:{rid}:g38';rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
        gum=-np.log(-np.log(rng.uniform(1e-12,1-1e-12,len(rr))))
        for method in METHODS:
            if method=='R':v=copy.deepcopy(rcontrol)
            elif method.startswith('Global-'):v=copy.deepcopy(oldmethods['R+'+method.split('-')[1]])
            else:
                view=method.split('-')[1];score=scores[:,0] if view=='S' else scores[:,1] if view=='V' else scores.mean(1)
                w=preserve(rcontrol['weights'],idx,score) if not rcontrol['fallback'] else np.asarray(rcontrol['weights'])
                mass=np.bincount(idx,weights=w,minlength=len(rcontrol['route_weights']))
                largest=max(largest,float(np.max(np.abs(mass-np.array(rcontrol['route_weights'])))))
                k=int(np.argmax(np.log(np.maximum(w,1e-300))+gum))
                v=dict(rcontrol,candidate=k,index=rr[k]['index'],weights=w.tolist(),route_weights=mass.tolist(),
                       effective_routes=float(np.exp(-sum(p*np.log(p) for p in mass if p>0))))
            v['method']=method;picks.append(v)
    for n in ('samples.npz','candidates.json','routings.json'):(out/n).symlink_to((source/n).resolve())
    save(out/'selections.json',{'rows':picks})
    gen.update(script_sha256=digest(__file__),exposed_data_diagnostic=True,
        source_generation_sha256=digest(source/'complete.json'),route_marginal_max_abs_error=largest,
        artifacts={n:digest(out/n) for n in ('samples.npz','candidates.json','selections.json','routings.json')})
    assert largest<1e-12;save(out/'complete.json',gen)


def checks():
    base=np.array([.2,.2,.1,.25,.25]);groups=np.array([0,0,0,1,1]);score=np.array([1,4,2,-3,8])
    w=preserve(base,groups,score)
    assert np.allclose(np.bincount(groups,weights=w),[.5,.5]) and not np.allclose(w,base)
    assert np.allclose(preserve(base,groups,np.ones(5)),base)
    assert np.allclose(preserve(base,np.arange(5),score),base)
    assert np.allclose(preserve([.5,.5,0.],[0,0,1],[0.,0.,2.]),[.5,.5,0.])
    print('G39 mechanism checks: constant scores, exact marginal preservation, singleton no-op')


def main(root,source,network):
    root.mkdir(parents=True,exist_ok=False);checks();receipts={};evaluation.METHODS=list(METHODS)
    for episode in (380201,380202):
        for seed in (42,43):
            b=source/str(episode);g=root/str(episode)/'generated'/str(seed);out=root/str(episode)/'evaluation'/str(seed)
            build(b/'generated'/str(seed),g)
            evaluation.evaluate(g,b/'catalog',b/'reference',b/'simulation',network,out)
            receipts[f'{episode}/{seed}']=json.loads((out/'complete.json').read_text())
    save(root/'complete.json',{'complete':True,'results':receipts,'diagnostic_only':True,
         'interpretation':'Same exposed K8 pools; equal expected route marginal. Duration may still shift; fixed-route-and-T stage is separate.',
         'candidate_budget_unchanged':True,'no_new_training_or_generation':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--network',type=Path);p.add_argument('--checks',action='store_true');a=p.parse_args()
    if a.checks:checks()
    else:main(a.root,a.source,a.network)
