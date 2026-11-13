"""Publish corrected five-class packs without modifying their source releases."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

PARTITIONS = ('train', 'validation', 'test')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def observed_departures(directory):
    with np.load(directory/'trips.npz') as data:
        ids = data['ids']
        lengths = data['h'][:, 3].astype(int)
        starts = np.r_[0, lengths.cumsum()[:-1]]
        raw = data['raw']
        assert len(raw) == lengths.sum() and np.all(lengths >= 120)
        times = raw[starts, 0]
        ends = raw[starts+lengths-1, 0]
    native = {i: (t, end) for i, t, end in zip(ids, times, ends)}
    confirmed = {}
    for row in ET.parse(directory/'arrivals.xml').getroot().iterfind('tripinfo'):
        key = row.get('id')
        if key in native:
            first, last = native[key]
            assert float(row.get('depart')) == first and float(row.get('arrival')) == last+1
            confirmed[key] = int(first//300)%288
    assert len(confirmed) == len(native) == len(ids)
    return confirmed


def make(src, shard_roots, memo, release):
    dest = src.with_name(src.name+'-localclock-v1')
    with (src.parent/(dest.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.exists():
            if not (dest/'COMPLETE').is_file():
                raise ValueError(f'Existing incomplete destination {dest}')
            return json.loads((dest/'manifest.json').read_text())
        with tempfile.TemporaryDirectory(dir=src.parent,prefix=dest.name+'-build.') as tmp:
            staging = Path(tmp)/'release'
            staging.mkdir()
            (staging/'normalization.npz').symlink_to(src/'normalization.npz')
            manifest = {'release':dest.name, 'parent':src.name, 'clock':'floor(native SUMO departure seconds / 300) % 288',
                        'coordinate_units':'standardized planar x/y, with normalization.npz storing metre-scale statistics',
                        'split':{},'normalization_sha256':digest(src/'normalization.npz')}
            wanted_seeds=set()
            seen_ids=set()
            seen_seeds=set()
            for part in PARTITIONS:
                orig = src/part; out = staging/part; out.mkdir()
                for name in ('traj.npy','index.npz','counts.json'):
                    (out/name).symlink_to(orig/name)
                with np.load(out/'index.npz') as index:
                    keys = index['ids']; classes = index['classes']
                expected = (release*1000 if part=='train' else 1000)*5
                assert len(keys)==len(set(keys))==expected and not seen_ids.intersection(keys)
                seen_ids.update(keys)
                seeds={i.split('_')[0] for i in keys}
                assert not seen_seeds & seeds
                seen_seeds.update(seeds)
                wanted_seeds.update(seeds)
                for seed in sorted(seeds):
                    if seed not in memo:
                        matches=[r/seed for r in shard_roots if (r/seed/'trips.npz').is_file()]
                        if len(matches)!=1: raise ValueError(f'Cannot resolve shard {seed}: {matches}')
                        memo[seed]=(matches[0], observed_departures(matches[0]))
                times=np.array([memo[i.split('_')[0]][1][i] for i in keys],dtype='int32')
                h0=np.load(orig/'head.npy')
                assert h0.shape==(expected,8) and np.isfinite(h0).all()
                h=h0.copy(); h[:,0]=times
                np.save(out/'head.npy',h)
                actual=np.load(out/'head.npy')
                assert actual[:,1:].tobytes()==h0[:,1:].tobytes() and np.array_equal(actual[:,0],times)
                assert np.load(out/'traj.npy', mmap_mode='r').shape==(expected,200,2)
                pop=Counter(classes.tolist())
                assert len(pop)==5 and set(pop.values())=={expected//5}
                if part=='train': training_bins=set(times.tolist())
                manifest['split'][part]={'n':expected, 'episodes':len(seeds),'classes':dict(pop),
                                         'changed_time_rows':int(np.sum(h0[:,0]!=actual[:,0])),
                                         'old_time_bins':sorted(set(h0[:,0].astype(int).tolist())),
                                         'native_time_bin_counts':dict(sorted(Counter(times.tolist()).items())),
                                         'time_seen_in_training_fraction':float(np.isin(times,list(training_bins)).mean()),
                                         'checksum':{f:digest(out/f) for f in ('head.npy','traj.npy','index.npz')},
                                         'old_header_sha256':digest(orig/'head.npy')}
            (staging/'raw_shards').mkdir()
            for seed in sorted(wanted_seeds):
                (staging/'raw_shards'/seed).symlink_to(memo[seed][0],target_is_directory=True)
            (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (staging/'COMPLETE').touch()
            staging.rename(dest)
        return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve()
    parents=[root/f'difftraj-five-{n}k' for n in (10,20)]
    memo={}
    summary={}
    for n,src in zip((10,20),parents):
        assert (src/'COMPLETE').is_file()
        summary[str(n)]=make(src, parents, memo,n)
    print(json.dumps(summary,indent=2))
