"""Frozen-model development ablation of the fabricated departure-time condition."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from porto import score


def run(fit):
    out={}
    stage=Path.cwd()
    with tempfile.TemporaryDirectory(dir=stage) as tmp:
        work=Path(tmp)
        original=fit/'prepared.npz'
        with np.load(original) as f:
            arrays=dict(f)
        head=arrays['h'];part=arrays['split']; train=part==0; dev=part==8
        counts=np.bincount(head[train,0].astype(int),minlength=288);modal=int(counts.argmax())
        unseen=np.logical_and(dev,counts[head[:,0].astype(int)]==0)
        assert np.all(unseen[dev])
        head=head.copy()
        head[unseen,0]=modal
        patched=work/'modal.npz'
        np.savez(patched,**dict(arrays,h=head))
        for seed in ('42','43'):
            out[seed]={}
            for cond,inp in [('original',original),('modal',patched)]:
                folder=work/f'{seed}-{cond}'
                folder.mkdir()
                (folder/'prepared.npz').symlink_to(inp)
                (folder/'checkpoint.pt').symlink_to(fit/seed/'checkpoint.pt')
                (folder/'observations.npz').symlink_to(fit/'observations.npz')
                os.chdir(folder)
                sys.argv=['porto.py','score','--development','--all-holdout','--save-draws']
                score()
                metrics=json.loads(Path('development.json').read_text())
                ref_xy=arrays['xy']; lo=arrays['lo']; hi=arrays['hi']
                draws=np.load('draws.npz')
                ids=draws['ids']
                assert set(ids)==set(np.flatnonzero(dev))
                actual=ref_xy[ids][:,[0,-1],:]
                generated=draws['xy'][:,[0,-1],:]
                err=np.linalg.norm(actual-generated,axis=2)
                q=np.clip(np.floor((generated-lo)/(hi-lo)*16),0,15).astype(int)
                correct=(16*q[:,:,0]+q[:,:,1])==head[ids,6:]
                metrics.update(endpoint_median_m=np.median(err,axis=0).tolist(),
                               endpoint_hit_fraction=correct.mean(0).tolist(),
                               both_endpoints_hit=float(correct.all(1).mean()))
                for k in ('real_real','real_real_n'):
                    metrics.pop(k,None)
                out[seed][cond]=metrics
                os.chdir(stage)
            assert out[seed]['original']['request_index_sha256']==out[seed]['modal']['request_index_sha256']
    Path('clock-ablation.json').write_text(json.dumps(dict(modal_train_bin=modal,replaced=int(unseen.sum()),results=out),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('fit',type=Path)
    run(p.parse_args().fit.resolve())
