"""Export internal matched request arrays, verifying coordinate identity and clocks."""
from pathlib import Path
import numpy as np
from utils.utils import resample_trajectory

root=Path('/workspace/cnr-experiments/porto-metric-review-01')
def pack(name, tracks, dt, mode):
    np.savez('tracks/'+name+'.npz',points=np.concatenate(tracks),ends=np.r_[0,np.cumsum([len(t) for t in tracks])],dt=dt,mode=mode)
Path('tracks').mkdir(exist_ok=True)
for seed in [42,43]:
    run=root/f'{seed}-mixed-on-mixed'
    draws=np.load(run/'draws.npz')
    raw=np.load(run/'observations.npz')
    ids=draws['ids']; sizes=raw['length']; ends=np.r_[0,sizes.cumsum()]
    delta=raw['delta']
    real=[delta[ends[i]:ends[i+1]].cumsum(0)*1e-6 for i in ids]
    synth=[resample_trajectory(x,int(sizes[i])) for x,i in zip(draws['xy'],ids)]
    if seed==42:
        porto_ref=np.concatenate(real)
        pack('Porto-real',real,np.full(len(ids),15.),np.full(len(ids),'taxi'))
    else:
        assert np.array_equal(porto_ref,np.concatenate(real))
    pack(f'Porto-gen-{seed}',synth,np.full(len(ids),15.),np.full(len(ids),'taxi'))
for seed in [42,43]:
    for arm in ['B','V','C','VC']:
        d=np.load(f'bj-{arm}-{seed}/private-{arm}-8000.npz')
        def gps(key):
            return d[key].transpose(0,2,1)/[111195*np.cos(np.deg2rad(40)),111195]+[116,40]
        if seed==42 and arm=='B':
            bj_ref=gps('reference'); bj_ids=d['ids']; bj_dt=d['dt']; bj_labels=d['labels']
            pack('Beijing-real',bj_ref,bj_dt,bj_labels)
        else:
            assert np.array_equal(bj_ref,gps('reference')) and np.array_equal(bj_ids,d['ids']) and np.array_equal(bj_dt,d['dt']) and np.array_equal(bj_labels,d['labels'])
        pack(f'Beijing-{arm}-{seed}',gps('generated'),d['dt'],d['labels'])
