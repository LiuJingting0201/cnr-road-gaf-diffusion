"""Freeze an independent 1,000-trip test set from the original mixed traffic recipe."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from make_balanced_trips import episode, src

ROOT = Path('/workspace/cnr-experiments')
OLD = ROOT/'difftraj-five-10k'
EXPANDED = ROOT/'difftraj-five-20k'
# Match the original holdout's synthetic local clock, preserving its seed modulo 1440.
FIRST_SEED = 430000 + 1440 * 2000
QUOTA = 1000


def digest(x):
    return hashlib.sha256(x.tobytes()).digest()


def existing_track_hashes(first_seed):
    """Match the release's interpolation digest on physical pre-normalization arrays."""
    seen = set()
    ids = set()
    for ds in [OLD, EXPANDED]:
        for split in ['train', 'validation', 'test']:
            with np.load(ds/split/'index.npz') as f:
                ids.update(f['ids'].tolist())
    old_seeds = {int(ident.split('_')[0]) for ident in ids}
    if old_seeds.intersection(range(first_seed, first_seed + 128)):
        raise ValueError('Fresh seed collision')
    for seed in sorted(old_seeds):
        shard = OLD/str(seed)/'trips.npz'
        if not shard.exists():
            shard = EXPANDED/str(seed)/'trips.npz'
        with np.load(shard) as f:
            seen.update(digest(x) for ident, x in zip(f['ids'], f['xy']) if ident in ids)
    return seen, ids, old_seeds


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--reference',action='store_true',help='Generate a separate 3,000-trip reference after the sealed 1,000-trip pack')
    args=cli.parse_args()
    first_seed=FIRST_SEED + (144000 if args.reference else 0)
    quota=3000 if args.reference else QUOTA
    seen, old_ids, old_seeds = existing_track_hashes(first_seed)
    if args.reference:
        test_folder=ROOT/'taxi-debug-fresh-01'
        assert (test_folder/'COMPLETE').is_file()
        with np.load(test_folder/'observations.npz') as pack:
            held_ids=set(pack['taxi'])
        held_seeds={int(k.split('_')[0]) for k in held_ids}
        for seed in held_seeds:
            with np.load(test_folder/str(seed)/'trips.npz') as data:
                seen.update(digest(xy) for ident,xy in zip(data['ids'],data['xy']) if ident in held_ids)
        old_ids.update(held_ids);old_seeds.update(held_seeds)
    comparison_hashes=len(seen)
    native_blocks, model_blocks, times, lengths, ids, hashes = [], [], [], [], [], []
    episodes_used = []
    duplicates = 0
    with ProcessPoolExecutor(8) as pool:
        for first in range(first_seed, first_seed + 128, 8):
            seeds = range(first, first+8)
            list(pool.map(episode,seeds))
            for seed in seeds:
                episodes_used.append(seed)
                with np.load(Path(str(seed))/'trips.npz') as f:
                    ns = f['h'][:, 3].astype(int)
                    ends = np.r_[0, ns.cumsum()]
                    bounds = list(zip(ends[:-1],ends[1:]))
                    raw = f['raw']
                    for j,(ident,label,xy) in enumerate(zip(f['ids'],f['classes'],f['xy'])):
                        if label != 'taxi':
                            continue
                        h = digest(xy)
                        if h in seen:
                            duplicates += 1
                            continue
                        a,b = bounds[j]
                        v = raw[a:b]
                        assert ident not in old_ids and v.shape[0]>=120 and np.all(np.diff(v[:,0])==1)
                        assert np.isfinite(v).all() and np.allclose(v[[0,-1],1:3],xy[[0,-1]])
                        seen.add(h)
                        hashes.append(h.hex())
                        ids.append(ident)
                        lengths.append(len(v))
                        times.append((seed%1440)*60+int(v[0,0]))
                        native_blocks.append(v)
                        pos=np.rint(v[:,1:3]*100).astype('int32')
                        model_blocks.append(np.diff(pos,axis=0,prepend=np.zeros((1,2),dtype='int32')))
                        if len(ids)==quota:
                            break
                if len(ids)==quota:
                    break
            if len(ids)==quota:
                break
    assert len(set(ids))==len(set(hashes))==quota
    assert not set(episodes_used).intersection(old_seeds)
    np.savez_compressed('native.npz',raw=np.concatenate(native_blocks),length=lengths,taxi=ids)
    model_fields=dict(delta=np.concatenate(model_blocks),length=lengths,
                      taxi=ids,start=times,duration=np.asarray(lengths)-1,timezone='UTC',
                      planar_unit_m=.01,episode_split=np.full(quota,9))
    if args.reference:
        model_fields['simulation_start']=[v[0,0] for v in native_blocks]
    np.savez_compressed('observations.npz',**model_fields)
    Path('manifest.json').write_text(json.dumps(dict(trips=quota,episodes_used=episodes_used,
        old_ids_checked=len(old_ids),old_distinct_geometry_hashes_checked=comparison_hashes,
        dropped_exact_duplicates=duplicates,source_network_sha256=hashlib.sha256((src/'most.net.xml').read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256(Path('native.npz').read_bytes()).hexdigest(),
        observations_sha256=hashlib.sha256(Path('observations.npz').read_bytes()).hexdigest(),
        ordered_id_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
        same_synthetic_clock_offset=True),indent=2)+'\n')
    Path('COMPLETE').touch()


if __name__=='__main__':
    main()
