"""Sensitivity analysis on the original validation matches, with a 0.50 m/s speed caliper."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from five_class_evaluate import MODES, array_digest, stratify, string_digest
from velocity_difftraj import sha
from velocity_label_contrasts import SIGNATURE, make_pairs, motion_signature


# Fixed before inspection of model outcomes. All four fits use the same pairs and bootstrap draws.
MAX_SPEED_GAP = 0.50
MIN_PAIRS = 30
MIN_EFFECT = 0.20
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 11167
FITS = tuple(product(('coordinate', 'velocity'), (42, 43)))
FEATURES = ('slow_time_percent', 'slow_entries_per_10min', 'p90_speed_mps',
            'mean_abs_acceleration_mps2')
UNITS = ('percentage points', 'entries/10 min', 'm/s', 'm/s²')


def speed_balance(speed, left, right):
    diff = speed[right] - speed[left]
    return dict(pairs=len(left), mean_signed_gap_mps=float(diff.mean()) if len(diff) else None,
                mean_absolute_gap_mps=float(np.abs(diff).mean()) if len(diff) else None)


def coverage(head, speed, labels, ids):
    selected, report = {}, {}
    for (a, b), (left, right) in make_pairs(head, labels).items():
        keep = np.abs(speed[right] - speed[left]) <= MAX_SPEED_GAP
        l, r = left[keep], right[keep]
        selected[a, b] = l, r
        report[f'{MODES[b]}_minus_{MODES[a]}'] = dict(first=MODES[a], second=MODES[b],
            before=speed_balance(speed, left, right), after=speed_balance(speed, l, r),
            matched_ids_sha256=string_digest([f'{ids[i]}|{ids[j]}' for i, j in zip(l, r)]))
    return selected, report


def generated_signatures(run, rep, seed, rows, actual, ids, duration, hashes):
    directory = run / f'{rep}-{seed}'
    cfg = json.loads((directory / 'config.json').read_text())
    if cfg['seed'] != seed or (cfg['arm'], cfg['mode']) != ('C', 'all'):
        raise ValueError(f'Unexpected training configuration in {directory}')
    with np.load(directory / 'eval/draw-cache.npz') as cache:
        expected = dict(**hashes, representation=rep, checkpoint_sha256=sha(directory / 'final.pt'))
        for key, value in expected.items():
            if cache[key].item() != value:
                raise ValueError(f'{rep}-{seed} draw cache mismatch at {key}')
        for name, value in [('validation_rows', rows), ('ids', ids), ('true_labels', actual)]:
            np.testing.assert_array_equal(cache[name], value)
        requests, targets = cache['draw_rows'], cache['draw_labels']
        keys = requests * len(MODES) + targets
        np.testing.assert_array_equal(cache['keys'], keys)
        if (len(np.unique(keys)) != len(keys) or np.any((requests < 0) | (requests >= len(rows)))
                or np.any((targets < 0) | (targets >= len(MODES)))):
            raise ValueError('Duplicate or invalid cached request/target')
        paths = cache['xy_m'].astype('f8')
    grid = np.full((len(rows), len(MODES), len(SIGNATURE)), np.nan)
    grid[requests, targets] = motion_signature(paths, duration[requests])
    return grid, expected['checkpoint_sha256']


def estimate(values, resamples):
    means = values[resamples].mean(axis=1)
    return dict(mean=float(values.mean()), percentile_95=np.quantile(means, [.025, .975]).tolist())


def effects(selected, pairs, observed, grids):
    for (a, b), (left, right) in selected.items():
        pair = pairs[f'{MODES[b]}_minus_{MODES[a]}']
        pair['features'] = {}
        n = len(left)
        if not n:
            continue
        obs_l, obs_r = observed[left], observed[right]
        pooled_sd = np.sqrt((obs_l.var(axis=0) + obs_r.var(axis=0)) / 2)
        contrasts = {}
        for fit, grid in grids.items():
            # Target b minus a at each of the two fixed headers; average within matched pair.
            contrasts[fit] = ((grid[left, b] - grid[left, a]) + (grid[right, b] - grid[right, a])) / 2
            if not np.isfinite(contrasts[fit]).all():
                raise ValueError(f'Missing label-swap draw for {fit}, {MODES[a]} / {MODES[b]}')
        rng = np.random.default_rng([BOOTSTRAP_SEED, a, b])
        resamples = rng.integers(n, size=(BOOTSTRAPS, n)) if n >= MIN_PAIRS else None
        for feature in FEATURES:
            i = SIGNATURE.index(feature)
            obs_delta = obs_r[:, i] - obs_l[:, i]
            d = float(obs_delta.mean() / pooled_sd[i]) if pooled_sd[i] > 0 else None
            eligible = n >= MIN_PAIRS and d is not None and abs(d) >= MIN_EFFECT
            item = dict(observed_standardized_gap=d, eligible=eligible)
            pair['features'][feature] = item
            if eligible:
                item['observed'] = estimate(obs_delta, resamples)
                item['generated'] = {fit: estimate(delta[:, i], resamples) for fit, delta in contrasts.items()}


def format_estimate(value):
    low, high = value['percentile_95']
    return f"{value['mean']:.3g} [{low:.3g}, {high:.3g}]"


def markdown(report):
    lines = ['# Vehicle-label contrasts at similar requested average speeds', '',
             'The original matched pairs were retained when their requested average speeds differed by at most '
             '0.50 m/s. Each class contributed 600 validation trips. Positive differences refer to the second '
             'class minus the first.', '',
             '| First class | Second class | Pairs before | Pairs retained | Mean signed speed gap before → after (m/s) | Mean absolute speed gap before → after (m/s) |',
             '|---|---|---:|---:|---:|---:|']
    for p in report['pairs'].values():
        x, y = p['before'], p['after']
        def gap(field):
            return ' → '.join('—' if s[field] is None else f'{s[field]:.3f}' for s in (x, y))
        lines.append(f"| {p['first']} | {p['second']} | {x['pairs']} | {y['pairs']} | "
                     f"{gap('mean_signed_gap_mps')} | {gap('mean_absolute_gap_mps')} |")
    if not report['coverage_only']:
        lines += ['', 'Means and percentile 95% intervals appear for pairs with at least 30 matches and '
                  'an absolute standardized observed difference of at least 0.20. Each generated difference '
                  'comes from changing the label on the same request and initial noise. The observed reference '
                  'compares two matched trips.', '',
                  '| Second minus first | Feature (unit) | Observed | Coordinate 42 | Coordinate 43 | Velocity 42 | Velocity 43 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for name, p in report['pairs'].items():
            for feat, unit in zip(FEATURES, UNITS):
                item = p['features'].get(feat)
                if item and item['eligible']:
                    vals = [item['observed']] + [item['generated'][f'{rep}-{seed}'] for rep, seed in FITS]
                    lines.append('| ' + ' | '.join([name.replace('_', ' '), f'{feat} ({unit})']
                                                   + [format_estimate(v) for v in vals]) + ' |')
        lines += ['', 'Intervals resample matched pairs within this fixed validation sample, holding the '
                  'models, latent noise, matches and eligibility screen fixed. They ignore possible dependence '
                  'between trips from the same simulation episode and variation between fits.']
    return '\n'.join(lines) + '\n'


def run(args):
    val = args.data / 'validation'
    with np.load(val / 'index.npz') as idx:
        rows, _ = stratify(idx['classes'])
        ids = idx['ids'][rows]
        actual = np.searchsorted(MODES, idx['classes'][rows])
    head = np.load(val / 'head.npy', mmap_mode='r')[rows]
    with np.load(args.data / 'normalization.npz') as z:
        norm = dict(z)
    speed = head[:, 4].astype('f8') * norm['header_std'][3] + norm['header_mean'][3]
    selected, pairs = coverage(head, speed, actual, ids)
    report = dict(coverage_only=args.coverage_only, split='validation', requests=len(rows),
                  request_sha256=string_digest(ids), pairs=pairs, max_requested_speed_gap_mps=MAX_SPEED_GAP,
                  direction='second class minus first class', min_pairs=MIN_PAIRS, min_abs_observed_d=MIN_EFFECT,
                  bootstrap_repeats=BOOTSTRAPS, bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_unit='matched pair; conditions on fixed sample and ignores episode dependence')
    if not args.coverage_only:
        if args.run is None:
            raise ValueError('--run is required for four-fit comparison')
        noise = np.random.default_rng(42).standard_normal((len(rows), 2, 200), dtype='f4')
        expected = dict(request_sha256=string_digest(ids), head_sha256=array_digest(head), noise_sha256=array_digest(noise),
                        source_manifest_sha256=sha(args.data / 'manifest.json'),
                        prepared_manifest_sha256=sha(args.run / 'prepared/manifest.json'))
        duration = head[:, 2].astype('f8') * norm['header_std'][1] + norm['header_mean'][1]
        np.testing.assert_allclose(duration, np.rint(duration), atol=.002, rtol=1.e-6)
        duration = np.rint(duration)
        xy = np.load(val / 'traj.npy', mmap_mode='r')[rows].astype('f8') * norm['std'] + norm['mean']
        observed = motion_signature(xy, duration)
        grids, checkpoints = {}, {}
        for rep, seed in FITS:
            fit = f'{rep}-{seed}'
            grids[fit], checkpoints[fit] = generated_signatures(args.run, rep, seed, rows, actual, ids, duration, expected)
        effects(selected, pairs, observed, grids)
        report.update(expected, checkpoints=checkpoints)
    md = markdown(report)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for filename, content in [('balance.json', json.dumps(report, indent=2, allow_nan=False) + '\n'), ('BALANCE.md', md)]:
            dest = args.output / filename
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(content)
            tmp.replace(dest)
    print(md, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--coverage-only', action='store_true')
    run(parser.parse_args())
