"""Independent integrity audit of a nested five-class SUMO release, with no model scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

PARTS = ('train', 'validation', 'test')
MODES = ('bicycle', 'bus', 'motorcycle', 'passenger', 'taxi')


def fingerprint(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def membership(base, part):
    with np.load(base / part / 'index.npz') as index:
        return index['ids'].tolist(), index['classes'].tolist()


def check_fresh(samples, coords, info, encoded, standardized, norm, departed, arrived):
    t, x, y, v = samples.T
    assert len(t) >= 120 and np.isfinite(samples).all() and (v >= 0).all()
    assert t[0] == departed and t[-1] + 1 == arrived and np.all(np.diff(t) == 1.)
    step = np.hypot(np.diff(x), np.diff(y))
    length = np.linalg.norm(np.diff(samples[:,1:3],axis=0),axis=1).sum()
    assert length >= 100 and step.max() <= 65.0001
    n = len(t)
    hdr = np.array([int(departed // 300) % 288, length, n-1, n, length/(n-1), length/(n-1)])
    np.testing.assert_allclose(hdr, info, atol=1.e-4, rtol=2.e-6)
    times = np.linspace(0., n - 1, 200)
    replay = np.column_stack([np.interp(times,np.arange(n),axis) for axis in (x,y)]).astype('f4')
    np.testing.assert_array_equal(replay, coords)
    expected = ((replay-norm['mean'])/norm['std']).astype('f4')
    np.testing.assert_allclose(standardized, expected, atol=2.e-6, rtol=1.e-6)
    cell = np.floor((replay[[0,-1]]-norm['lo'])/(norm['hi']-norm['lo'])*16).clip(0,15).astype(int)
    assert encoded[0] == hdr[0]
    np.testing.assert_array_equal(encoded[6:8],cell[:,0]*16+cell[:,1])
    continuous = (hdr[1:]-norm['header_mean']) / norm['header_std']
    np.testing.assert_allclose(encoded[1:6], continuous, atol=2.e-5, rtol=2.e-5)


def audit(base, parent, quota):
    manifest=json.loads((base/'manifest.json').read_text())
    assert (Path(manifest['parent']).resolve() == parent.resolve()
            and fingerprint(base/'normalization.npz') == manifest['normalization_sha256'])
    with np.load(base/'normalization.npz') as pack:
        norm = dict(pack)
    all_ids, all_geometry, all_episodes = set(), set(), set()
    result = {}
    for part in PARTS:
        old_ids,old_labels = membership(parent,part)
        ids,labels = membership(base,part)
        assert ids[:len(old_ids)] == old_ids and labels[:len(old_ids)] == old_labels
        counts = Counter(labels)
        expected = quota if part=='train' else 1000
        assert counts == Counter({m:expected for m in MODES})
        if part!='train':
            assert ids == old_ids and labels == old_labels
        assert len(ids) == len(set(ids)) and not all_ids.intersection(ids)
        all_ids.update(ids)
        parent_episodes = {k.split('_')[0] for k in old_ids}
        groups=defaultdict(dict)
        for j,(key,mode) in enumerate(zip(ids,labels)):
            seed,prefix,suffix=key.split('_')
            assert mode==prefix
            if j>=len(old_ids):
                assert seed not in parent_episodes and int(seed)>=5100000
            groups[seed][key]=j
        assert not all_episodes.intersection(groups)
        all_episodes.update(groups)
        data=base/part; rec=manifest['splits'][part]
        assert rec['n']==len(ids) and rec['classes']==dict(counts) and rec['episodes']==len(groups)
        for k in ('traj.npy','head.npy','index.npz'):
            assert fingerprint(data/k) == rec['sha256'][k]
        coordinates=np.load(data/'traj.npy',mmap_mode='r')
        header=np.load(data/'head.npy',mmap_mode='r')
        assert coordinates.shape==(len(ids),200,2) and header.shape==(len(ids),8)
        n_verified=0
        source_rows=0
        for seed,indices in groups.items():
            directory=base/'raw_shards'/seed
            fresh=seed not in parent_episodes
            if not fresh:
                assert directory.samefile(parent/'raw_shards'/seed)
            with np.load(directory/'trips.npz') as shard:
                keys=shard['ids'].tolist()
                classes=shard['classes'].tolist()
                geometry=shard['xy']
                if fresh:
                    assert part=='train' and set(indices)==set(keys)
                    receipts={item.get('id'):(float(item.get('depart')),float(item.get('arrival')))
                              for item in ET.parse(directory/'arrivals.xml').getroot().iterfind('tripinfo')
                              if item.get('vaporized','') in ('','0','false')}
                    assert set(receipts)==set(keys)
                    native=shard['raw']; original=shard['h']
                    ends=np.r_[0,original[:,3].astype(int).cumsum()]
                    assert ends[-1]==len(native)
                for j,key in enumerate(keys):
                    if key not in indices:
                        continue
                    i=indices[key]
                    assert labels[i]==classes[j]
                    digest=hashlib.sha256(geometry[j].tobytes()).digest()
                    assert digest not in all_geometry
                    all_geometry.add(digest)
                    source_rows+=1
                    if fresh:
                        check_fresh(native[ends[j]:ends[j+1]],geometry[j],original[j],header[i],
                                    coordinates[i],norm,*receipts[key])
                        n_verified+=1
        assert source_rows==len(ids) and n_verified==len(ids)-len(old_ids)
        result[part]=dict(total=len(ids),classes=dict(counts),episodes=len(groups),inherited=len(old_ids),
                          new_rows_checked_against_native_and_arrival=n_verified)
    assert len(all_geometry)==len(all_ids)
    return dict(release=base.name,parent=parent.name,geometry_unique=len(all_geometry),partitions=result,
                geometry_definition='SHA-256 of 200 two-dimensional native-metre float32 positions',
                frozen_check='Unchanged membership, labels, shared original shard identity, uniqueness and file digests only. No holdout quality scores.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('parent',type=Path)
    parser.add_argument('--quota',type=int,default=40000)
    args=parser.parse_args()
    report=audit(args.root,args.parent,args.quota)
    out=args.root/'independent-verification.json'
    temp=out.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2)+'\n')
    temp.replace(out)
