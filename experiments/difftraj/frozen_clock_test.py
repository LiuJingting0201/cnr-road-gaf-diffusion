"""Apply the prespecified two-generator comparison to every frozen test request."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from five_class_train import make_model
from five_class_evaluate import MODES, features, occupancy, read_observations, sample, string_digest, array_digest
from pooled_velocity_evaluate import scores
from retime_difftraj_paths import retime
from route_clock_eval import create_maps
from velocity_difftraj import decode, sha
from velocity_label_balance import coverage
from velocity_label_contrasts import SIGNATURE, motion_signature


def run(args):
    data, base, seed=args.data,args.base,args.seed
    out=args.output
    out.mkdir(exist_ok=True,parents=True)
    lock=json.loads(args.lock.read_text())
    assert sha(data/'manifest.json')==lock['source_manifest'] and sha(base/'prepared/manifest.json')==lock['prepared_manifest']
    manifest=json.loads((data/'manifest.json').read_text())
    test=data/'test'
    for name,hsh in manifest['splits']['test']['sha256'].items():
        assert sha(test/name)==hsh
    norm=dict(np.load(base/'prepared/normalization.npz'))
    z=np.load(test/'index.npz'); ids=z['ids']; labels=np.searchsorted(MODES,z['classes'])
    assert len(np.unique(ids))==5000 and np.all(np.bincount(labels)==1000)
    episodes=set(i.split('_')[0] for i in ids)
    for split in ('train','validation'):
        with np.load(data/split/'index.npz') as other:
            assert episodes.isdisjoint(i.split('_')[0] for i in other['ids'])
    h=np.load(test/'head.npy')
    n=len(ids)
    noise=np.random.default_rng(lock['test_noise_seed']).standard_normal((n,2,200),dtype='f4')
    pairs,coverage_stats=coverage(h,h[:,4].astype('d')*norm['header_std'][3]+norm['header_mean'][3],labels,ids)
    bike,bus=MODES.index('bicycle'),MODES.index('bus')
    left,right=pairs[bike,bus]
    m=len(left)
    slots=np.r_[np.arange(n),left,right]
    targets=np.r_[labels,np.full(m,bus),np.full(m,bike)]
    for rep in ('coordinate','velocity'):
        path=base/f'{rep}-{seed}'/'final.pt'
        assert sha(path)==lock['checkpoints'][f'{rep}-{seed}']
        net=make_model('C').eval()
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['weights'])
        if rep=='coordinate':
            route=sample(net,h,labels,noise,norm,64).astype('f4').astype('d')
        else:
            tokens=sample(net,h[slots],targets,noise[slots],dict(mean=np.zeros(2),std=np.ones(2)),64)
            velocity=decode(tokens,h[slots],norm,norm['velocity_token_mean'],norm['velocity_token_std']).astype('f4').astype('d')
    xy=np.stack([retime(route[r],v)[0] for r,v in zip(slots,velocity)])
    np.testing.assert_array_equal(xy[:,[0,-1]],route[slots][:,[0,-1]])
    raw=read_observations(data,ids)
    duration=np.array([len(r)-1 for r in raw])
    ref=np.load(test/'traj.npy').astype('d')*norm['std']+norm['mean']
    reference=features(ref,duration)
    occ=occupancy(raw,norm['lo'],norm['hi'])
    tr=np.load(data/'train/head.npy',mmap_mode='r')[:,1].astype('d')*norm['header_std'][0]+norm['header_mean'][0]
    bins=np.unique(np.quantile(tr,np.arange(1,20)/20))
    rec=dict(seed=seed,episodes=len(episodes),n=n,classes=list(MODES),requests_sha256=string_digest(ids),
             noise_sha256=array_digest(noise),route_sha256=array_digest(route),endpoint_changes=0,
             pairs=coverage_stats['bus_minus_bicycle'],arms={})
    for key,p in [('coordinate',route),('combined',xy[:n])]:
        each,macro=scores(raw,reference,occ,bins,p,h,labels,duration,norm)
        rec['arms'][key]=dict(each=each,macro=macro)
    sig=motion_signature(xy,duration[slots]); obs=motion_signature(ref,duration)
    delta=((sig[n:n+m]-sig[left])+(sig[right]-sig[n+m:]))/2
    rec['observed']=dict(zip(SIGNATURE,(obs[right]-obs[left]).mean(0)))
    rec['generated']=dict(zip(SIGNATURE,delta.mean(0)))
    create_maps(out,seed,[np.flatnonzero(labels==i)[0] for i in range(5)],labels,ids,raw,duration,[route,xy[:n]],
                names=('Coordinate diffusion','Combined'),legend='coordinate (orange), combined (blue)')
    j=np.argsort(ids[left])[(m-1)//2]; r=left[j]
    t=np.linspace(0,duration[r],200)
    fig,ax=plt.subplots(figsize=(8.5,3.1))
    for path,name,color in [(ref[r],'Observed bicycle','#50545a'),(xy[r],'Bicycle label','#007b97'),(xy[n+j],'Bus label','#bb501a')]:
        ax.plot(t[1:],np.linalg.norm(np.diff(path,axis=0),axis=1)/(t[1]-t[0]),label=name,color=color,lw=1.3)
    ax.set(title=f'Same generated route and noise · {ids[r]} · seed {seed}',xlabel='Time (s)',ylabel='Speed (m/s)')
    ax.legend(frameon=False,ncol=3); fig.tight_layout(); fig.savefig(out/f'swap-{seed}.png',dpi=165); plt.close(fig)
    (out/f'metrics-{seed}.json').write_text(json.dumps(rec,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','data','output','lock'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True,choices=(42,43))
    run(p.parse_args())
