"""Score complete generated bus journeys in the release's validation partition."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from five_class_evaluate import (array_digest, features, motion_scores, occupancy,
                                 paper_scores, read_observations, string_digest)
from utils.utils import resample_trajectory


def entered_spells(speeds, duration):
    """Entered spells use the same two-interval rule as the low-speed entry metric."""
    lengths = []
    for velocity, seconds in zip(speeds, duration):
        low = velocity < 1.0
        edges = np.diff(np.r_[False, low, False].astype('i1'))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        included = (starts > 0) & (ends - starts >= 2)
        lengths.extend((ends[included] - starts[included]) * (seconds / len(velocity)))
    return np.array(lengths, dtype='f8')


def describe_spells(values):
    if not len(values):
        return dict(entered=0, median_s=None, p90_s=None, at_least_30s_percent=None)
    return dict(entered=len(values), median_s=float(np.median(values)),
                p90_s=float(np.quantile(values, .9)),
                at_least_30s_percent=float(100 * np.mean(values >= 30.)))


def align_rows(identifiers, data):
    with np.load(data/'validation/index.npz') as manifest:
        ids = manifest['ids'].astype(str)
        expected = np.flatnonzero(manifest['classes'] == 'bus')
    requested = np.asarray(identifiers)
    if requested.ndim != 1 or len(requested) != 1000:
        raise ValueError('Require one ID/validation index per trip for all 1,000 buses.')
    if requested.dtype.kind in 'ui':
        rows = requested.astype('i8')
    else:
        by_id = {ids[i]: i for i in expected}
        rows = np.array([by_id[key] for key in requested.astype(str)])
    if not np.array_equal(np.sort(rows), expected):
        raise ValueError('Requests differ from complete bus validation population.')
    return rows, ids[rows]


def score(rows, paths, data, noise_sha256=None):
    data = Path(data)
    rows, ids = align_rows(rows, data)
    xy = np.asarray(paths, dtype='f8')
    if xy.shape != (len(rows), 200, 2) or not np.isfinite(xy).all():
        raise ValueError('Expected finite metre coordinates [1,000, 200, 2].')
    with np.load(data/'normalization.npz') as pack:
        norm = dict(pack)
    full = read_observations(data, ids)
    duration = np.array([len(track)-1 for track in full])
    heads = np.load(data/'validation/head.npy', mmap_mode='r')[rows]
    duration_h = heads[:,2].astype('f8')*norm['header_std'][1]+norm['header_mean'][1]
    np.testing.assert_allclose(duration, duration_h, atol=.002, rtol=1.e-6)
    x0 = np.load(data/'validation/traj.npy', mmap_mode='r')[rows]*norm['std']+norm['mean']
    observed_features, observed_speed = features(x0, duration)
    generated_features, generated_speed = features(xy, duration)
    each = np.arange(len(rows))
    motion = motion_scores(observed_features, generated_features, observed_speed, generated_speed, each)
    observed_spells = entered_spells(observed_speed, duration)
    generated_spells = entered_spells(generated_speed, duration)
    spans = dict(observed=describe_spells(observed_spells), generated=describe_spells(generated_spells),
                 duration_w1_s=float(wasserstein_distance(observed_spells, generated_spells))
                 if len(generated_spells) else None)
    original = occupancy(full, norm['lo'], norm['hi'])
    restored = [resample_trajectory(track, int(seconds)+1) for track, seconds in zip(xy, duration)]
    synthetic = occupancy(restored, norm['lo'], norm['hi'])
    train_length = np.load(data/'train/head.npy', mmap_mode='r')[:,1].astype('f8')
    train_length = train_length*norm['header_std'][0]+norm['header_mean'][0]
    bins = np.unique(np.quantile(train_length, np.arange(1,20)/20))
    return dict(requests=len(rows), request_sha256=string_digest(ids),
                head_sha256=array_digest(heads), noise_sha256=noise_sha256,
                paper=paper_scores(original, synthetic, each, each, bins),
                motion=motion, spells=spans)


def load_draws(file):
    with np.load(file, allow_pickle=False) as z:
        identifiers = z['ids'] if 'ids' in z else z['rows']
        coordinates = z['xy_m'] if 'xy_m' in z else z['xy']
        digest = str(z['noise_sha256'].item()) if 'noise_sha256' in z else None
        if 'noise' in z:
            actual = array_digest(z['noise'])
            if digest is not None and digest != actual:
                raise ValueError(f'Noise array differs from supplied hash in {file}.')
            digest = actual
    return identifiers, coordinates, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--draws', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    ids, paths, digest = load_draws(args.draws)
    result = score(ids, paths, args.data, digest)
    if args.baseline:
        ids0, _, digest0 = load_draws(args.baseline)
        _, canonical = align_rows(ids0, args.data)
        if result['request_sha256'] != string_digest(canonical):
            raise ValueError('Initial request identifiers or order differ from reference.')
        if not digest or digest != digest0:
            raise ValueError('Matching initial-noise fingerprints required for paired comparison.')
        result['paired_baseline'] = args.baseline.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(args.output.suffix+'.tmp')
    pending.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    pending.replace(args.output)


if __name__=='__main__':
    main()
