"""Compare observation cadence on the same complete SUMO journeys and evaluation requests."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from porto import grid

root = Path.cwd()
one = root / 'one'
fifteen = root / 'fifteen'


def write(name, data):
    dest = root / name
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(dest)


def pack():
    with np.load(one / 'observations.npz', allow_pickle=False) as src:
        lengths = src['length']
        duration = src['duration']
        np.testing.assert_array_equal(lengths - 1, duration)
        delta = src['delta']
        offsets = np.r_[0, lengths.cumsum()]
        trains = src['episode_split'] < 8
        records = {k: src[k] for k in src.files if k not in ('delta', 'length')}
        packed, ratios = [], []
        for start, end in zip(offsets[:-1], offsets[1:]):
            points = delta[start:end].cumsum(axis=0)
            indices = np.r_[np.arange(0, len(points) - 1, 15), len(points) - 1]
            sampled = points[indices]
            np.testing.assert_array_equal(sampled[[0, -1]], points[[0, -1]])
            diffs = np.diff(sampled, axis=0)
            packed.append(np.vstack([sampled[0], diffs]).astype(delta.dtype))
            original = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            ratios.append(np.linalg.norm(diffs, axis=1).sum() / original if original else 1.)
        sizes = np.asarray([len(t) for t in packed], dtype=lengths.dtype)
        records.update(length=sizes, delta=np.concatenate(packed))
        np.savez_compressed(fifteen / 'observations.npz', **records)
        write('cadence-input.json', dict(train=int(trains.sum()), development=int((src['episode_split']==8).sum()),
              rule='0,15,30,... and original terminal point; final interval can be shorter than 15 s',
              train_median_native_points=float(np.median(lengths[trains])),
              train_median_thinned_points=float(np.median(sizes[trains])),
              train_median_duration_s=float(np.median(duration[trains])),
              train_median_length_retained=float(np.median(np.asarray(ratios)[trains])),
              train_full_15s_intervals_ge119=int((duration[trains] >= 1785).sum())))


def align():
    with np.load(one / 'prepared.npz') as base, np.load(fifteen / 'prepared.npz') as thin:
        data = dict(thin)
        data['lo'], data['hi'] = base['lo'], base['hi']
        for col, endpoint in [(6, 0), (7, -1)]:
            data['h'][:, col] = grid(data['xy'][:, endpoint], data['lo'], data['hi'])
        np.testing.assert_array_equal(base['split'], data['split'])
        np.testing.assert_array_equal(data['h'][:, [0, 2, 6, 7]], base['h'][:, [0, 2, 6, 7]])
    np.savez(fifteen / 'prepared.npz', **data)
    split = json.loads((fifteen / 'split.json').read_text())
    split['coord_range'] = [data['lo'].tolist(), data['hi'].tolist()]
    write('fifteen/split.json', split)


def summary():
    from taxi_od_diagnostic import audit
    result = {'preprocessing': json.loads((root / 'cadence-input.json').read_text()),
              'native_clock': {}, 'shared_15s_clock': {}}
    ref_ids = None
    for arm in ['one', 'fifteen']:
        for clock in ['native_clock', 'shared_15s_clock']:
            result[clock][arm] = {}
        for seed in ['42', '43']:
            run = root / arm / seed
            with np.load(run / 'draws.npz') as draw:
                if ref_ids is None:
                    ref_ids = draw['ids']
                np.testing.assert_array_equal(ref_ids, draw['ids'])
            for clock in ['native_clock', 'shared_15s_clock']:
                if clock == 'shared_15s_clock' and arm == 'fifteen':
                    item = result['native_clock'][arm][seed].copy()
                else:
                    folder = run
                    check = run / 'development.json'
                    if clock == 'shared_15s_clock':
                        folder = root / 'cross-eval' / seed
                        folder.mkdir(parents=True, exist_ok=True)
                        for filename in ['observations.npz','prepared.npz','draws.npz']:
                            link = folder / filename
                            if not link.exists():
                                link.symlink_to((run if filename == 'draws.npz' else fifteen) / filename)
                        check = None
                    stats = audit(folder, one, check)
                    full = stats['common']['full']
                    item = {key: full[key] for key in ('density_js','od_js','top20_cell_f1','joint_hit')}
                    item.update(length_js=stats['common']['length_js'],length_w1_km=stats['length_w1_km'],
                                n=stats['n'],reference_sha256=stats['reference_sha256'])
                result[clock][arm][seed] = item
    assert len({d['reference_sha256'] for v in result['shared_15s_clock'].values() for d in v.values()}) == 1
    write('cadence-common.json', result)


if __name__ == '__main__':
    {'pack': pack, 'align': align, 'summary': summary}[sys.argv[1]]()
