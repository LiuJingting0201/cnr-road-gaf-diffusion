"""Evaluate a five-mode conditional DiffTraj checkpoint on shared validation requests."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from five_class_evaluate import (MODES, PAPER, array_digest, features, motion_scores,
                                 occupancy, paper_scores, read_observations, sample,
                                 stratify, string_digest)
from score_velocity_pilot import endpoint_scores
from utils.utils import resample_trajectory
from velocity_difftraj import decode, sha
from velocity_label_contrasts import (contrast_report, make_pairs, motion_signature,
                                      required_alternatives)


def request_features(headers):
    """Same eight-header distance embedding as general_speed_class.request_features."""
    angle = headers[:, 0] * 2 * np.pi / 288
    cells = headers[:, 6:8].astype(int)
    return np.column_stack((headers[:, 1:6], np.sin(angle), np.cos(angle),
                            (cells // 16) / 4, (cells % 16) / 4))


def infer_labels(train_head, train_classes, head):
    neighbors = cKDTree(request_features(train_head)).query(request_features(head), k=32)[1]
    rank = np.random.default_rng(11161).integers(32, size=len(head))
    labels = train_classes[neighbors[np.arange(len(head)), rank]]
    if not np.isin(labels, MODES).all():
        raise ValueError('Unexpected class in the training set')
    return np.searchsorted(MODES, labels)


def draw_plan(actual, inferred, alternatives):
    count = len(MODES)
    row_base = np.arange(len(actual)) * count
    supplied = row_base + actual
    inferred_keys = row_base + inferred
    swapped = np.asarray([r * count + c for r, c in alternatives], dtype='i8')
    keys = np.r_[supplied, np.setdiff1d(np.r_[inferred_keys, swapped], supplied)]
    lookup = np.full(count * len(actual), -1, dtype='i8')
    lookup[keys] = np.arange(len(keys))
    return keys, lookup, lookup[inferred_keys]


def emit_draws(args, headers, noise, keys, norm, checkpoint):
    from five_class_train import make_model
    model = make_model('C').eval()
    model.load_state_dict(checkpoint['weights'])
    # Retain the original validation order for the full block of true labels.
    chunks = (keys[:len(headers)], keys[len(headers):])
    xy = []
    for group in chunks:
        if not len(group):
            continue
        rows, labels = np.divmod(group, len(MODES))
        head = headers[rows]
        print(f'Drawing {len(rows)} unique request-label pairs', flush=True)
        tokens = sample(model, head, labels, noise[rows],
                        {'mean': np.zeros(2), 'std': np.ones(2)}, args.batch)
        if args.representation == 'velocity':
            paths = decode(tokens, head, norm, norm['velocity_token_mean'], norm['velocity_token_std'])
        else:
            paths = tokens * norm['std'] + norm['mean']
        xy.append(paths.astype('f4'))
    paths = np.concatenate(xy)
    if paths.shape != (len(keys), 200, 2) or not np.isfinite(paths).all():
        raise FloatingPointError('Invalid sampled trajectories')
    return paths


def scores(raw, ref_motion, obs_occupancy, bins, xy, head, actual, duration, norm):
    synthetic = occupancy([resample_trajectory(p, int(d) + 1) for p, d in zip(xy, duration)],
                          norm['lo'], norm['hi'])
    f0, v0 = ref_motion
    f1, v1 = features(xy, duration)
    a0, a1 = [np.diff(v, axis=1) / (duration[:, None] / 199) for v in (v0, v1)]
    results = {}
    for code, mode in enumerate(MODES):
        ix = np.flatnonzero(actual == code)
        motion = motion_scores(f0, f1, v0, v1, ix)
        motion['accel_w1_mps2'] = float(wasserstein_distance(a0[ix].ravel(), a1[ix].ravel()))
        motion['paper'] = paper_scores(obs_occupancy, synthetic, ix, ix, bins)
        motion['endpoints'] = endpoint_scores([raw[i] for i in ix], xy[ix], head[ix], norm)[0]
        results[mode] = motion

    macro = {k: float(np.mean([results[m][k] for m in MODES]))
             for k in ('speed_w1_mps', 'accel_w1_mps2')}
    macro['paper'] = {k: float(np.mean([results[m]['paper'][k] for m in MODES])) for k in PAPER}
    macro['endpoints'] = {k: float(np.mean([results[m]['endpoints'][k] for m in MODES]))
                          for k in ('start_cell_hit', 'end_cell_hit', 'joint_cell_hit')}
    return results, macro


def evaluate(args):
    import torch
    data, prepared = args.data, args.prepared
    source_hash, prepared_hash = [sha(p / 'manifest.json') for p in (data, prepared)]
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if (manifest['source_manifest_sha256'] != source_hash or manifest['mode'] != 'all'
            or manifest['classes'] != list(MODES)):
        raise ValueError('Source and prepared release differ')
    cfg = json.loads((args.checkpoint.parent / 'config.json').read_text())
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    expected_hash = prepared_hash if args.representation == 'velocity' else source_hash
    if (cfg['release_manifest_sha256'] != expected_hash or ck['epoch'] != cfg['epochs']
            or (ck['mode'], ck['arm'], tuple(ck['classes'])) != ('all', 'C', MODES)
            or ck['seed'] != cfg['seed'] or cfg['speed_loss_weight'] != 0
            or cfg['train_n'] != 200_000 or cfg['epochs'] != 10):
        raise ValueError('Checkpoint incompatible with requested experiment')
    with np.load(prepared / 'normalization.npz') as z:
        norm = dict(z)
    # The prepared release keeps the coordinate and header normalization of the source.
    with np.load(data / 'normalization.npz') as z:
        for field in ('mean', 'std', 'header_mean', 'header_std', 'lo', 'hi'):
            np.testing.assert_array_equal(z[field], norm[field])
    val = data / 'validation'
    with np.load(val / 'index.npz') as z:
        rows, _ = stratify(z['classes'])
        ids, actual = z['ids'][rows], np.searchsorted(MODES, z['classes'][rows])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Duplicate validation trip IDs')
    head = np.load(val / 'head.npy', mmap_mode='r')[rows]
    train_head = np.load(data / 'train/head.npy', mmap_mode='r')
    with np.load(data / 'train/index.npz') as z:
        inferred = infer_labels(train_head, z['classes'], head)
    noise = np.random.default_rng(42).standard_normal((len(rows), 2, 200), dtype='f4')
    pairs = make_pairs(head, actual)
    alternatives = required_alternatives(pairs)
    keys, lookup, header_draws = draw_plan(actual, inferred, alternatives)
    draw_rows, draw_labels = np.divmod(keys, len(MODES))
    args.output.mkdir(exist_ok=True, parents=True)
    cache = args.output / 'draw-cache.npz'
    fingerprints = dict(request_sha256=string_digest(ids), head_sha256=array_digest(head),
                        noise_sha256=array_digest(noise), checkpoint_sha256=sha(args.checkpoint),
                        source_manifest_sha256=source_hash, prepared_manifest_sha256=prepared_hash,
                        representation=args.representation, sample_batch=args.batch)
    if cache.exists():
        with np.load(cache) as z:
            for k, v in fingerprints.items():
                if z[k].item() != v:
                    raise ValueError(f'Draw cache differs at {k}')
            np.testing.assert_array_equal(z['keys'], keys)
            np.testing.assert_array_equal(z['ids'], ids)
            xy = z['xy_m']
        if xy.shape != (len(keys), 200, 2) or not np.isfinite(xy).all():
            raise ValueError('Invalid cached trajectories')
    else:
        xy = emit_draws(args, head, noise, keys, norm, ck)
        pending = cache.with_suffix('.pending')
        with pending.open('wb') as stream:
            np.savez_compressed(stream, keys=keys, draw_rows=draw_rows, draw_labels=draw_labels,
                                validation_rows=rows, ids=ids, true_labels=actual,
                                inferred_labels=inferred, xy_m=xy, **fingerprints)
        pending.replace(cache)
    print(f'Scoring {len(keys)} draws for {len(ids)} validation requests', flush=True)
    raw = read_observations(data, ids)
    duration = np.asarray([len(p) - 1 for p in raw], dtype='f8')
    header_duration = head[:, 2].astype('f8') * norm['header_std'][1] + norm['header_mean'][1]
    np.testing.assert_allclose(duration, header_duration, atol=.002, rtol=1.e-6)
    reference = np.load(val / 'traj.npy', mmap_mode='r')[rows].astype('f8') * norm['std'] + norm['mean']
    obs_occupancy = occupancy(raw, norm['lo'], norm['hi'])
    train_length = train_head[:, 1].astype('f8') * norm['header_std'][0] + norm['header_mean'][0]
    bins = np.unique(np.quantile(train_length, np.arange(1, 20) / 20))
    ref_motion = features(reference, duration)
    counts = np.zeros((len(MODES), len(MODES)), dtype='i8')
    np.add.at(counts, (actual, inferred), 1)
    report = dict(**fingerprints, split='validation', classes=list(MODES), seed=ck['seed'], epoch=ck['epoch'],
                  updates=cfg['updates_per_epoch'] * ck['epoch'], requests=len(ids), unique_draws=len(keys),
                  inferred_labels_changed=int((actual != inferred).sum()), class_confusion=counts.tolist(),
                  confusion_axes=['true class', 'header-only sampled class'], results={}, macro={})
    for name, select in [('provided', np.arange(len(ids))), ('header_only', header_draws)]:
        report['results'][name], report['macro'][name] = scores(
            raw, ref_motion, obs_occupancy, bins, xy[select].astype('f8'), head, actual, duration, norm)
    signatures = motion_signature(xy.astype('f8'), duration[draw_rows])
    alternative_signatures = {(r, c): signatures[lookup[r * len(MODES) + c]] for r, c in alternatives}
    report['label_contrasts'] = contrast_report(pairs, ids, motion_signature(reference, duration),
                                                signatures[:len(ids)], alternative_signatures)
    metrics = args.output / 'metrics.json'
    pending = metrics.with_suffix('.pending')
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pending.replace(metrics)
    print(f'Completed {metrics}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'prepared', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--representation', choices=('coordinate', 'velocity'), required=True)
    p.add_argument('--batch', type=int, default=64)
    evaluate(p.parse_args())
