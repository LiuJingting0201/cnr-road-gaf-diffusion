"""Align immutable train-only features by identity, never validation order."""
import json
from pathlib import Path
import numpy as np
import torch
from g33_train import digest, save


def derangement(labels, lengths, durations):
    rng=np.random.default_rng(39042); permutation=np.arange(len(labels)); groups=[]
    for c in range(5):
        eligible=np.flatnonzero(labels==c)
        order=eligible[np.argsort(lengths[eligible],kind='stable')]
        coarse=[order[k:k+128] for k in range(0,len(order),128)]
        if len(coarse)>1 and len(coarse[-1])==1:
            coarse[-2]=np.r_[coarse[-2],coarse[-1]];coarse.pop()
        for chunk in coarse:
            chunk=chunk[np.argsort(durations[chunk],kind='stable')]
            bands=[chunk[k:k+16] for k in range(0,len(chunk),16)]
            if len(bands)>1 and len(bands[-1])==1:
                bands[-2]=np.r_[bands[-2],bands[-1]];bands.pop()
            for band in bands:
                if len(band)<2: raise ValueError('Insufficient shuffle support')
                shuffled=rng.permutation(band);permutation[shuffled]=np.roll(shuffled,1)
                groups.append({'class':c,'n':len(band)})
    assert not np.any(permutation==np.arange(len(labels)))
    assert np.array_equal(labels,labels[permutation]) and len(set(permutation))==len(labels)
    return permutation,groups


def targets(rows, teachers, output, device):
    vectors,centers,means,sources={},{},{},{}
    for v in ('spatial','speed'):
        folder=Path(teachers)/f'{v}-42'
        receipt=json.loads((folder/'complete.json').read_text())
        assert receipt['complete'] and not (folder/'failed.json').exists()
        for name in ('model.pt','train-predictions.npz'):
            assert digest(folder/name)==receipt['artifacts'][name]['sha256']
        with np.load(folder/'train-predictions.npz',allow_pickle=False) as f:
            assert len(f['ids'])==len(set(f['ids']))==len(rows['ids'])
            lookup={str(s):i for i,s in enumerate(f['ids'])}
            ix=np.asarray([lookup[str(s)] for s in rows['ids']])
            assert np.array_equal(f['labels'][ix],rows['labels'])
            assert np.array_equal(f['parent_ids'][ix],rows['parent_ids'])
            raw=f['features'][ix].astype(np.float32)
            assert raw.shape==(len(ix),512) and np.isfinite(raw).all()
            state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            w,b=state['fc.weight'].numpy(),state['fc.bias'].numpy()
            assert np.allclose(raw@w.T+b,f['logits'][ix],atol=2e-4,rtol=2e-4)
        means[v]=raw.mean(0);unit=raw-means[v]
        norm=np.linalg.norm(unit,axis=1,keepdims=True)
        if np.any(norm<1e-8):raise ValueError('Constant teacher feature')
        unit/=norm
        cent=np.stack([unit[rows['labels']==c].mean(0) for c in range(5)])
        cent/=np.maximum(np.linalg.norm(cent,axis=1,keepdims=True),1e-8)
        vectors[v]=torch.as_tensor(unit,device=device)
        centers[v]=torch.as_tensor(cent,device=device)
        sources[v]={'receipt_sha256':digest(folder/'complete.json'),
                    'model_sha256':digest(folder/'model.pt'),'features_sha256':digest(folder/'train-predictions.npz')}
    perm,groups=derangement(rows['labels'],rows['length'],rows['duration'])
    np.savez_compressed(output/'target-normalizer.npz',**means)
    np.savez_compressed(output/'shuffle.npz',ids=rows['ids'],paired_ids=rows['ids'][perm])
    save(output/'targets.json',{'complete':True,'rows':len(perm),'sources':sources,'groups':groups,
         'normalization':'train global mean then row unit norm', 'validation_features_opened':False,
         'self_pairs':int(np.sum(perm==np.arange(len(perm)))),
         'same_parent_pairs':int(np.sum(rows['parent_ids']==rows['parent_ids'][perm])),
         'relative_length_gap_quantiles':np.quantile(np.abs(rows['length'][perm]/rows['length']-1),[0,.5,.9,1]).tolist(),
         'relative_duration_gap_quantiles':np.quantile(np.abs(rows['duration'][perm]/rows['duration']-1),[0,.5,.9,1]).tolist()})
    return vectors,centers,torch.as_tensor(perm,device=device)


def checks():
    labels=np.repeat(np.arange(5),257);r=np.arange(len(labels))+1.
    p,_=derangement(labels,r,r**.5)
    assert len(set(p))==len(p) and np.array_equal(labels,labels[p]) and np.all(p!=np.arange(len(p)))
    print('G39 shuffle checks: singleton tails, bijection, no self pair, class preservation')


if __name__=='__main__':checks()
