"""Build paired evaluation cohorts with identical request order and common training bins."""
import os
from pathlib import Path
import sys
import numpy as np
from porto import prepare, arc_length
import hashlib
root=Path.cwd()
old=np.load('long/observations.npz')
def subset(raw, ids):
    n=raw['length']; ends=np.r_[0,n.cumsum()]; delta=raw['delta']
    return dict(delta=np.concatenate([delta[ends[i]:ends[i+1]] for i in ids]),**{k:raw[k][ids] for k in ['length','taxi','start']})
taxi=old['taxi']
mapping={t:int(hashlib.sha256(str(t).encode()).hexdigest()[:8],16)%10 for t in set(taxi)}
split=np.array([mapping[t] for t in taxi])
long_train=subset(old,np.flatnonzero(split<8))
long_test=subset(old,np.random.default_rng(42).choice(np.flatnonzero(split==9),3000,replace=False))
mixed_train=dict(np.load('mixed-train.npz')); mixed_test=dict(np.load('mixed-test.npz'))
def bins(raw):
    ends=np.r_[0,raw['length'].cumsum()]
    lengths=[arc_length(raw['delta'][a:b].cumsum(0)*1e-6) for a,b in zip(ends[:-1],ends[1:])]
    return np.unique(np.quantile(lengths,np.arange(1,20)/20))
np.savez(root/'shared-bins.npz',long=bins(long_train),mixed=bins(mixed_train))
for arm,train in [('long',long_train),('mixed',mixed_train)]:
    for cohort,test in [('long',long_test),('mixed',mixed_test)]:
        assert not set(train['taxi']) & set(test['taxi'])
        path=root/(arm+'-on-'+cohort);path.mkdir(exist_ok=True);os.chdir(path)
        np.savez('observations.npz',**{k:np.concatenate([train[k],test[k]]) for k in train})
        prepare()
        os.chdir(root)
