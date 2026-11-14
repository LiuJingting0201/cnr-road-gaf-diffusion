"""Score final DiffTraj draws against fixed validation journeys in the five-class release."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

MODES = ('bicycle', 'bus', 'motorcycle', 'passenger', 'taxi')
PAPER = ('density_js', 'od_js', 'length_js', 'hotspot_f1')
FEATURES = ('mean_speed_mps', 'p90_speed_mps', 'slow_time_percent', 'slow_entries_per_10min')


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def string_digest(values):
    return hashlib.sha256(('\n'.join(values) + '\n').encode()).hexdigest()


def stratify(labels):
    rng = np.random.default_rng(42)
    picks, other = [], []
    for mode in MODES:
        rows = np.flatnonzero(labels == mode)
        if len(rows) != 1000:
            raise ValueError(f'Expected 1000 validation rows for {mode}, got {len(rows)}')
        order = rng.permutation(rows)
        picks.extend(order[:600])
        other.append(order)
    return np.asarray(picks), other


def read_observations(data, ids):
    """Actual native positions for paper metrics; do not use the 200-point proxy."""
    grouped = defaultdict(dict)
    for row, key in enumerate(ids):
        grouped[key.split('_')[0]][key] = row
    paths = [None] * len(ids)
    for episode, lookup in grouped.items():
        with np.load(data / 'raw_shards' / episode / 'trips.npz') as pack:
            size = pack['h'][:, 3].astype(int)
            end = np.r_[0, size.cumsum()]
            raw = pack['raw']
            if len(raw) != end[-1]:
                raise ValueError(f'Invalid native shard {episode}')
            for j, key in enumerate(pack['ids'].tolist()):
                if key in lookup:
                    track = raw[end[j]:end[j + 1]]
                    if not (np.diff(track[:, 0]) == 1).all():
                        raise ValueError(f'Non-uniform native clock for {key}')
                    paths[lookup[key]] = track[:, 1:3].astype('f8')
    if any(t is None for t in paths):
        raise ValueError('Missing validation source rows')
    return paths


def occupancy(tracks, low, high):
    n = len(tracks)
    points = np.zeros((n, 256), dtype='i4')
    od = np.empty(n, dtype='i8')
    length = np.empty(n)
    outside = np.empty(n, dtype='i8')
    for k, track in enumerate(tracks):
        if not np.isfinite(track).all():
            raise ValueError('Nonfinite coordinate')
        cell = np.clip(np.floor((track - low) / (high - low) * 16), 0, 15).astype('i8')
        c = cell[:, 0] * 16 + cell[:, 1]
        points[k] = np.bincount(c, minlength=256)
        od[k] = c[0] * 256 + c[-1]
        length[k] = np.linalg.norm(np.diff(track, axis=0), axis=1).sum()
        outside[k] = np.any((track < low) | (track > high), axis=1).sum()
    return points, od, length, outside


def paper_scores(obs, gen, rows_o, rows_g, bins):
    def counts(items, selection):
        p, od, length, outside = (x[selection] for x in items)
        return (p.sum(0), np.bincount(od, minlength=65536), length,
                np.bincount(np.digitize(length, bins), minlength=len(bins) + 1),
                float(outside.sum() / p.sum()))
    p, od, length, hist, _ = counts(obs, rows_o)
    q, gd, generated, ghist, goutside = counts(gen, rows_g)
    def js(x, y):
        return float(jensenshannon(x, y) ** 2)
    def top(x):
        return set(np.argsort(-x, kind='stable')[:20])
    return dict(density_js=js(p, q), od_js=js(od, gd), length_js=js(hist, ghist),
                hotspot_f1=len(top(p) & top(q)) / 20, length_w1_m=float(wasserstein_distance(length, generated)),
                reference_median_length_m=float(np.median(length)), generated_median_length_m=float(np.median(generated)),
                outside_fraction=goutside, reference_occupied_od=int((od > 0).sum()),
                generated_occupied_od=int((gd > 0).sum()), requests=len(rows_g))


def features(paths, duration):
    dt = duration / (paths.shape[1] - 1)
    speed = np.linalg.norm(np.diff(paths, axis=1), axis=2) / dt[:, None]
    slow = speed < 1.0
    entries = ((~slow[:, :-2]) & slow[:, 1:-1] & slow[:, 2:]).sum(1)
    stats = np.column_stack((speed.mean(1), np.quantile(speed, .9, axis=1),
                             100 * slow.mean(1), entries * 600 / duration))
    return stats, speed


def motion_scores(obs, gen, real_v, gen_v, rows):
    a = obs[rows].mean(0)
    b = gen[rows].mean(0)
    return dict(requests=len(rows),
                speed_w1_mps=float(wasserstein_distance(real_v[rows].ravel(), gen_v[rows].ravel())),
                observed=dict(zip(FEATURES, a.tolist())),
                generated=dict(zip(FEATURES, b.tolist())),
                signed_error=dict(zip(FEATURES, (b-a).tolist())))


def match_bus_bicycle(headers, labels):
    bike, bus = [np.flatnonzero(labels == mode) for mode in ('bicycle', 'bus')]
    x, y = headers[bike, None], headers[None, bus]
    num = np.abs(x[:, :, 1:6] - y[:, :, 1:6])
    clock = np.abs(x[:, :, 0] - y[:, :, 0])
    clock = np.minimum(clock, 288-clock)
    origin_dest = np.concatenate([np.abs(x[:, :, 6:] // 16 - y[:, :, 6:] // 16),
                                  np.abs(x[:, :, 6:] % 16 - y[:, :, 6:] % 16)], axis=2)
    ok = (num.max(2) <= .75) & (clock <= 36) & (origin_dest.max(2) <= 2)
    cost = num.sum(2) + clock / 36 + origin_dest.sum(2) / 2
    i, j = linear_sum_assignment(np.where(ok, cost, 1.e7))
    keep = ok[i, j]
    return bike[i[keep]], bus[j[keep]]


def sample(net, head, target, noise, scale, batch):
    import torch
    from utils.utils import p_xt
    beta = torch.linspace(.0001, .05, 500, device='cuda')
    vals = []
    with torch.inference_mode():
        for start in range(0, len(noise), batch):
            sl = slice(start, start + batch)
            z = torch.from_numpy(noise[sl]).cuda()
            h = torch.from_numpy(head[sl]).cuda()
            y = torch.from_numpy(target[sl]).cuda()
            for step in reversed(range(0, 500, 5)):
                t = torch.full((len(z),), step, device='cuda')
                z = p_xt(z, net(z, t, h, y), t, torch.full_like(t, max(-1, step-5)), beta, 0.)
            vals.append(z.transpose(1, 2).cpu().numpy())
    paths = np.concatenate(vals) * scale['std'] + scale['mean']
    if not np.isfinite(paths).all():
        raise ValueError('Nonfinite generated path')
    return paths


def evaluate(args):
    import torch
    from five_class_train import CLASSES, make_model
    from utils.utils import resample_trajectory
    started = time.monotonic()
    if args.batch < 1:
        raise ValueError('Batch must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert tuple(CLASSES) == tuple(ck['classes']) == MODES
    arm, mode = ck['arm'], ck['mode']
    assert (arm in ('C', 'D') and mode == 'all') or (arm in ('A', 'B') and mode in MODES)
    net = make_model(arm, device='cuda')
    net.load_state_dict(ck['weights'])
    net.eval()
    epoch, train_seed = int(ck['epoch']), int(ck['seed'])
    del ck
    with np.load(args.data / 'normalization.npz') as pack:
        norm = dict(pack)
    valdir = args.data / args.split
    with np.load(valdir / 'index.npz') as idx:
        ids, labels = idx['ids'], idx['classes']
    head = np.load(valdir / 'head.npy')
    standardized = np.load(valdir / 'traj.npy', mmap_mode='r')
    train_head = np.load(args.data / 'train/head.npy', mmap_mode='r')
    length_bins = np.unique(np.quantile(train_head[:, 1].astype('f8') * norm['header_std'][0]
                                        + norm['header_mean'][0], np.arange(1, 20) / 20))
    pool, controls = stratify(labels)
    # Generate the same noise table on CPU independent of GPU, arm, mode, and minibatch.
    noise_all = np.random.default_rng(42).standard_normal((len(pool), 2, 200), dtype=np.float32)
    slots = np.arange(len(pool))
    if mode != 'all':
        slots = slots[labels[pool[slots]] == mode]
    if args.smoke:
        slots = np.concatenate([slots[labels[pool[slots]] == k][:2] for k in MODES])
    picked = pool[slots]
    labs = labels[picked]
    active = [k for k in MODES if k in labs]
    target = np.array([MODES.index(k) for k in labs], dtype='i8')
    noise = noise_all[slots]
    xy = sample(net, head[picked], target, noise, norm, args.batch)
    val_keep = np.unique(np.r_[picked, np.concatenate(
        [order[:4] if args.smoke else order for k, order in zip(MODES, controls) if k in active])])
    actual = read_observations(args.data, ids[val_keep])
    back = {row: j for j, row in enumerate(val_keep)}
    native_rows = np.array([back[j] for j in picked])
    sizes = np.array([len(actual[j]) for j in native_rows])
    durations = sizes.astype('f8') - 1
    obs200 = standardized[picked].astype('f8') * norm['std'] + norm['mean']
    header_duration = head[picked, 2].astype('f8') * norm['header_std'][1] + norm['header_mean'][1]
    np.testing.assert_allclose(header_duration, durations, atol=.002, rtol=1.e-6)
    np.testing.assert_allclose(np.stack([resample_trajectory(actual[j], 200) for j in native_rows]),
                               obs200, rtol=1.e-6, atol=.002)
    real_occ = occupancy(actual, norm['lo'], norm['hi'])
    gen_occ = occupancy(
        [resample_trajectory(t, int(s)) for t, s in zip(xy, sizes)], norm['lo'], norm['hi'])
    real_f, real_v = features(obs200, durations)
    gen_f, gen_v = features(xy, durations)
    every = np.arange(len(xy))
    scores = paper_scores(real_occ, gen_occ, native_rows, every, length_bins)
    by_class, control_scores, motion = {}, {}, {}
    for k, order in zip(MODES, controls):
        ix = np.flatnonzero(labs == k)
        if not len(ix):
            continue
        by_class[k] = paper_scores(real_occ, gen_occ, native_rows[ix], ix, length_bins)
        by_class[k].update(request_sha256=string_digest(ids[picked[ix]]),
                           noise_sha256=array_digest(noise[ix]))
        motion[k] = motion_scores(real_f, gen_f, real_v, gen_v, ix)
        used = order[:4] if args.smoke else order
        control_ix = np.array([back[j] for j in used])
        a, b = np.split(control_ix, 2)
        control_scores[k] = paper_scores(real_occ, real_occ, a, b, length_bins)
    macro = {k: float(np.mean([v[k] for v in by_class.values()])) for k in PAPER}
    swap = dict(applicable=mode == 'all', pairs=0)
    if mode == 'all':
        bike, bus = match_bus_bicycle(head[picked], labs)
        swap.update(pairs=len(bike), eligible_requests_per_side={k: int((labs == k).sum()) for k in ('bus','bicycle')},
                    matching_rule='Disjoint minimum-cost pairs. All five standardized continuous attributes within 0.75 SD, departure within 180 minutes, each OD cell coordinate within 2 grid steps.',
                    covered_fraction=len(bike) / int((labs == 'bicycle').sum()),
                    observed_pair_ids_sha256=string_digest([ids[picked[i]]+'|'+ids[picked[j]] for i,j in zip(bike,bus)]))
        if len(bike):
            pairs = np.r_[bike, bus]
            switched = np.where(labs[pairs] == 'bus', MODES.index('bicycle'), MODES.index('bus')).astype('i8')
            changed_xy = sample(net, head[picked[pairs]], switched, noise[pairs], norm, args.batch)
            changed, _ = features(changed_xy, durations[pairs])
            delta = (changed - gen_f[pairs]) * np.where(labs[pairs, None] == 'bicycle', 1, -1)
            contrast = (real_f[bus] - real_f[bike]).mean(0)
            swap.update(observed_bus_minus_bicycle=dict(zip(FEATURES, contrast.tolist())),
                        generated_bus_minus_bicycle=dict(zip(FEATURES, delta.mean(0).tolist())),
                        absolute_contrast_error=dict(zip(FEATURES, np.abs(delta.mean(0)-contrast).tolist())))
            np.savez_compressed(out / 'swap-draws.npz', ids=ids[picked[pairs]], swapped=switched,
                                xy_m=changed_xy.astype('f4'), pair_observed_ids=ids[picked[np.column_stack([bike, bus])]])
        else:
            swap['coverage_note'] = 'No eligible pairs; no contrast estimated.'
    phase = 'smoke' if args.smoke else 'validation'
    report = dict(arm=arm, seed=train_seed, epoch=epoch, mode=mode, split=args.split, phase=phase,
                  **scores, by_class=by_class, macro_average=macro, motion_by_class=motion,
                  motion=motion_scores(real_f, gen_f, real_v, gen_v, every), label_swap=swap,
                  disjoint_observed_vs_observed=control_scores,
                  metric_protocol='Paper metrics use native observed metre coordinates and generated coordinates interpolated to the corresponding native count; 16 x 16 training bounds, 20 training length-quantile bins, natural-log JS, top 20 occupied-cell F1.',
                  motion_protocol='Both paths on the same 200-point elapsed-time grid. Equal weight per trip for pooled 199-interval speed W1. Slow means below 1 m/s. An entry requires a preceding fast interval and two consecutive slow intervals.',
                  length_bins_m=length_bins.tolist(), low_m=norm['lo'].tolist(), high_m=norm['hi'].tolist(),
                  request_sha256=string_digest(ids[picked]), noise_sha256=array_digest(noise), pool_noise_sha256=array_digest(noise_all),
                  eps_seed=42, sampling=dict(steps=100, stride=5, eta=0, batch=args.batch),
                  seconds=time.monotonic()-started)
    np.savez_compressed(out / 'draws.npz', ids=ids[picked], validation_rows=picked, classes=labs, xy_m=xy.astype('f4'))
    dest = out / 'metrics.json'
    temp = out / 'metrics.json.tmp'
    temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temp.replace(dest)
    print(json.dumps(dict(phase=phase, arm=arm, seed=train_seed, mode=mode, requests=len(xy),
                          **{k: report[k] for k in PAPER}, matched_pairs=swap['pairs'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation'], default='validation')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--smoke', action='store_true', help='Two requests per eligible class and four observed control trips per class.')
    evaluate(parser.parse_args())
