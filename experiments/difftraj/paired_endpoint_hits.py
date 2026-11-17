"""Summarize destination-cell gains and losses for identical validation requests."""
import argparse
import json
from pathlib import Path

import numpy as np


def hits(run,seed,labels,rows,ids,head,lo,width):
    with np.load(run/f'velocity-{seed}/eval/draw-cache.npz') as z:
        n=len(rows)
        np.testing.assert_array_equal(z['ids'],ids)
        np.testing.assert_array_equal(z['validation_rows'],rows)
        np.testing.assert_array_equal(z['draw_rows'][:n],np.arange(n))
        np.testing.assert_array_equal(z['draw_labels'][:n],labels)
        end=z['xy_m'][:n,-1].astype('d')
    cell=np.floor((end-lo)/width).astype(int)
    return (cell>=0).all(1)&(cell<16).all(1)&(cell[:,0]*16+cell[:,1]==head[:,7])


def main(data,old,new):
    with np.load(data/'normalization.npz') as z:
        lo,width=z['lo'],(z['hi']-z['lo'])/16
    with np.load(old/'velocity-42/eval/draw-cache.npz') as z:
        rows,ids,labels=z['validation_rows'],z['ids'],z['true_labels']
    head=np.load(data/'validation/head.npy')[rows]
    episodes,groups=np.unique([key.split('_')[0] for key in ids],return_inverse=True)
    names,codes=np.unique(labels,return_inverse=True)
    shape=(len(episodes),len(names))
    counts=np.zeros(shape)
    np.add.at(counts,(groups,codes),1)
    # Preserve episode-level dependence and equal weight of the five classes.
    sampled=np.random.default_rng(1711).integers(len(episodes),size=(3000,len(episodes)))
    report={}
    for seed in (42,43):
        before,after=[hits(run,seed,labels,rows,ids,head,lo,width) for run in (old,new)]
        differences=np.zeros(shape)
        np.add.at(differences,(groups,codes),after.astype(int)-before.astype(int))
        rep=(differences[sampled].sum(1)/counts[sampled].sum(1)).mean(1)
        report[str(seed)]=dict(n=len(ids),episodes=len(episodes),both_hit=int((before&after).sum()),
            both_miss=int((~before&~after).sum()),gained=int((~before&after).sum()),lost=int((before&~after).sum()),
            change_pp=100*float(after.mean()-before.mean()),episode_bootstrap_95_pp=(100*np.quantile(rep,[.025,.975])).tolist())
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','old','new'): p.add_argument(name,type=Path)
    a=p.parse_args()
    main(a.data,a.old,a.new)
