"""Use the same disjoint original Porto development taxis for both training populations."""
from pathlib import Path
import numpy as np
from porto import prepare

def selected(raw,ids):
    lengths=raw['length'];ends=np.r_[0,lengths.cumsum()];delta=raw['delta']
    return dict(delta=np.concatenate([delta[ends[i]:ends[i+1]] for i in ids]),**{k:raw[k][ids] for k in ('length','taxi','start')})

def make_train_dev(arm):
    b=Path('/workspace/cnr-experiments/porto-cutoff-v1')
    base=b/'long'
    split=np.load(base/'prepared.npz')['split']
    raw=np.load(base/'observations.npz')
    dev=selected(raw,np.random.default_rng(142).choice(np.flatnonzero(split==8),512,replace=False))
    tr=selected(raw,np.flatnonzero(split<8)) if arm=='long' else dict(np.load(b/'mixed-train.npz'))
    assert not set(tr['taxi']) & set(dev['taxi'])
    np.savez('observations.npz',**{k:np.concatenate([tr[k],dev[k]]) for k in tr})
    prepare()
