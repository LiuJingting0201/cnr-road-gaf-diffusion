"""Summarize four paired vehicle-label evaluations from their aggregate JSON files."""
import argparse
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
from statistics import fmean


MODES = ('bicycle', 'bus', 'motorcycle', 'passenger', 'taxi')
PAIRS = tuple((a, b, f'{b}_minus_{a}') for a, b in combinations(MODES, 2))
RUNS = tuple(product(('coordinate', 'velocity'), (42, 43)))
RUN_LABELS = [f'{representation.title()} {seed}' for representation, seed in RUNS]
FEATURES = {
    'mean_speed_mps': 'Mean speed (m/s)',
    'p90_speed_mps': '90th-percentile speed (m/s)',
    'slow_time_percent': 'Time below 1 m/s (pp)',
    'slow_entries_per_10min': 'Low-speed entries per 10 min',
    'mean_abs_acceleration_mps2': 'Mean absolute acceleration (m/s²)',
}
# Multiplier gives display units. F1 stays on the 0–1 scale.
METRICS = [
    ('Density JS ↓', ('paper', 'density_js'), 1),
    ('OD JS ↓', ('paper', 'od_js'), 1),
    ('Length JS ↓', ('paper', 'length_js'), 1),
    ('Hotspot F1 ↑', ('paper', 'hotspot_f1'), 1),
    ('Speed W1 (m/s) ↓', ('speed_w1_mps',), 1),
    ('Acceleration W1 (m/s²) ↓', ('accel_w1_mps2',), 1),
    ('Destination cell hit (%) ↑', ('endpoints', 'end_cell_hit'), 100),
]


def field(record, path):
    for key in path:
        record = record[key]
    if not math.isfinite(record):
        raise ValueError(f'Nonfinite {path}')
    return record


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def observations(run):
    evidence = {}
    direction = {}
    for a, b, key in PAIRS:
        pair = run['label_contrasts'][key]
        n = pair['matched_pairs']
        if (pair['first'], pair['second']) != (a, b) or not 0 <= n <= 600:
            raise ValueError(f'Invalid class pair {key}')
        features = pair.get('features', {})
        if set(features) != (set(FEATURES) if n else set()):
            raise ValueError(f'Missing or unexpected signatures in {key}')
        evidence[key] = {}
        for name, signature in features.items():
            obs, pred = [field(signature, (k,)) for k in ('observed', 'generated')]
            sd = field(signature, ('observed_pooled_sd',))
            if sd < 0 or not math.isclose(abs(pred - obs), signature['absolute_error']):
                raise ValueError(f'Invalid contrast in {key}/{name}')
            valid = n >= 20 and sd > 0 and abs(obs / sd) >= .2
            if signature['direction_evaluable'] != valid:
                raise ValueError(f'Direction screen disagrees in {key}/{name}')
            correct = obs * pred > 0 if valid else None
            if signature['same_direction'] != correct:
                raise ValueError(f'Invalid direction in {key}/{name}')
            direction[key, name] = correct
            evidence[key][name] = [obs, sd]
    for mode in MODES:
        direct = run['results']['provided'][mode]
        controls = run['results']['header_only'][mode]
        if direct['requests'] != 600 or controls['requests'] != 600:
            raise ValueError(f'Expected 600 balanced requests for {mode}')
        if (direct['observed'] != controls['observed'] or
                direct['paper']['reference_median_length_m'] != controls['paper']['reference_median_length_m']):
            raise ValueError(f'Observation differs between label policies in {mode}')
        evidence[mode] = [direct['observed'], direct['paper']['reference_median_length_m']]
    return digest(evidence), direction


def load_runs(root):
    runs, directions = {}, {}
    check = None
    for rep, seed in RUNS:
        rec = json.loads((root / f'{rep}-{seed}.json').read_text())
        if (rec['representation'], rec['seed'], rec['split'], rec['epoch'], rec['updates'],
                tuple(rec['classes']), rec['requests']) != (rep, seed, 'validation', 10, 9760, MODES, 3000):
            raise ValueError(f'Unexpected protocol in {rep}-{seed}')
        matched = {k: [rec['label_contrasts'][k]['matched_pairs'],
                       rec['label_contrasts'][k]['paired_request_sha256']] for _, _, k in PAIRS}
        if len(rec['label_contrasts']) != len(PAIRS):
            raise ValueError('Expected exactly ten class pairs')
        observed_hash, directions[rep, seed] = observations(rec)
        fingerprints = {k: rec[k] for k in ('request_sha256', 'head_sha256', 'noise_sha256',
                        'source_manifest_sha256', 'prepared_manifest_sha256', 'unique_draws',
                        'inferred_labels_changed', 'class_confusion', 'sample_batch')}
        confusion = rec['class_confusion']
        if len(confusion) != len(MODES) or any(len(row) != len(MODES) or sum(row) != 600 for row in confusion):
            raise ValueError('Invalid per-class counts in the header-only control')
        if rec['inferred_labels_changed'] != 3000 - sum(row[i] for i, row in enumerate(confusion)):
            raise ValueError('Header-only change count disagrees with the class confusion matrix')
        fingerprints.update(observed_summary_sha256=observed_hash, matched_support_sha256=digest(matched))
        if check is None:
            check = fingerprints
        if fingerprints != check:
            diffs = [k for k in check if check[k] != fingerprints[k]]
            raise ValueError(f'{rep}-{seed} differs on {diffs}')
        for policy in ('provided', 'header_only'):
            for _, path, _ in METRICS:
                macro = fmean(field(rec['results'][policy][m], path) for m in MODES)
                if not math.isclose(macro, field(rec['macro'][policy], path), abs_tol=1.e-12):
                    raise ValueError(f'{rep}-{seed} inconsistent macro for {policy} {path}')
        runs[rep, seed] = rec
    return runs, directions, check


def table(lines, headings, rows):
    lines += ['', '| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join('---' for _ in headings) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    lines.append('')


def signed(x):
    return f'{x:+.4f}'


def render(runs, directions, check):
    ref = runs['coordinate', 42]
    contrasts = ref['label_contrasts']
    supported = [k for _, _, k in PAIRS if contrasts[k]['matched_pairs'] >= 30]
    lines = ['# Vehicle-label control on SUMO validation trajectories', '',
             f'Each fit uses 200,000 training trips and 9,760 updates. Each evaluation uses '
             f'3,000 requests, 600 per vehicle class, and {ref["unique_draws"]:,} distinct request–label draws. '
             'The trip request includes average speed. Both representations receive the same eight attributes '
             'and use the same label embedding architecture.', '', '## Fixed-request vehicle swaps', '',
             'The observed contrast is the mean difference between matched SUMO trips. The generated contrast '
             'switches the vehicle label at each header using identical noise, subtracts in the same direction, '
             'and averages over both headers of each matched pair.', '',
             f'{len(supported)} of 10 vehicle pairs have at least 30 matches. The prespecified skill per feature '
             'is 1 − Σ|generated − observed| / Σ|observed| over these pairs. Every pair contributes one absolute error. '
             'A score of 1 reproduces the SUMO contrasts; 0 gives the total error of always predicting zero '
             'class effect. Negative scores have greater error. The denominator is the absolute SUMO contrast '
             'summed in the stated physical units. A zero denominator is shown as n/a.']
    skill_rows, direction_rows = [], []
    for feature, title in FEATURES.items():
        denom = sum(abs(contrasts[k]['features'][feature]['observed']) for k in supported)
        scores = []
        for run in runs.values():
            error = sum(run['label_contrasts'][k]['features'][feature]['absolute_error'] for k in supported)
            scores.append(f'{1 - error / denom:.3f}' if denom else 'n/a')
        skill_rows.append([title, f'{denom:.4f}', *scores])
        eligible = [k for _, _, k in PAIRS if directions[RUNS[0]].get((k, feature)) is not None]
        hits = [sum(directions[r][k, feature] for k in eligible) for r in RUNS]
        repeated = [sum(all(directions[rep, s][k, feature] for s in (42, 43)) for k in eligible)
                    for rep in ('coordinate', 'velocity')]
        direction_rows.append([title, len(eligible), *hits, *repeated])
    table(lines, ['Feature', 'Σ∣observed∣', *RUN_LABELS], skill_rows)
    lines.extend(['### Direction of the class effect', '',
                  'The descriptive direction screen uses at least 20 matches and a SUMO gap of at least '
                  '0.2 pooled standard deviations in magnitude, with positive pooled SD. Each entry counts '
                  'eligible pairs with the correct sign. The last two columns require agreement in both seeds '
                  'on the same pairs. A zero generated difference counts as a miss.'])
    table(lines, ['Feature', 'Eligible', *RUN_LABELS, 'Coordinate both', 'Velocity both'], direction_rows)
    lines += ['### Coverage and signed differences', '',
              'Pair labels give the order of subtraction. Requests can contribute to more than one comparison.']
    table(lines, ['Second − first', 'Matched pairs', 'Included in skill'], [
        [f'{b} − {a}', contrasts[k]['matched_pairs'], 'Yes' if k in supported else 'No'] for a, b, k in PAIRS])
    lines += ['<details>', '<summary>Signed effects in physical units for all observed pairs</summary>']
    effect_rows = []
    for a, b, k in PAIRS:
        if not contrasts[k]['matched_pairs']:
            continue
        for feature, title in FEATURES.items():
            obs = contrasts[k]['features'][feature]['observed']
            pred = [signed(r['label_contrasts'][k]['features'][feature]['generated']) for r in runs.values()]
            effect_rows.append([f'{b} − {a}', title, signed(obs), *pred])
    table(lines, ['Second − first', 'Feature', 'SUMO', *RUN_LABELS], effect_rows)
    lines += ['</details>', '', '## Generated trajectories with the supplied vehicle class', '',
              'Results are arithmetic means of five within-class scores. The four DiffTraj metrics '
              'use the native observed trajectories and generated paths interpolated to the comparison clock. '
              'Speed and acceleration use 199 equal-duration intervals on each 200-position path. '
              'Acceleration uses differences of consecutive scalar speeds. Destination hits require the '
              'generated endpoint to be in bounds and in the requested cell.']
    table(lines, ['Metric', *RUN_LABELS], [
        [name, *[f"{field(r['macro']['provided'], path) * scale:.4f}" for r in runs.values()]]
        for name, path, scale in METRICS])
    lines += ['### Value of supplying the correct vehicle label', '',
              f'A fixed draw from 32 training requests with similar headers changes the label for '
              f'{ref["inferred_labels_changed"]:,} of 3,000 requests. The table subtracts the header-only '
              'label result from the supplied-label result. Negative deltas favour the supplied label '
              'for ↓ metrics and positive deltas favour it for ↑ metrics. The hit-rate difference is in '
              'percentage points.']
    table(lines, ['Metric', *RUN_LABELS], [
        [name.replace('(%)', '(pp)'), *[signed(scale * (field(r['macro']['provided'], path) -
                                                       field(r['macro']['header_only'], path))) for r in runs.values()]]
        for name, path, scale in METRICS])
    lines += ['The next table gives the change in error for a class’s average motion statistic '
              '(supplied minus header-only). Each score averages the five absolute within-class '
              'mean errors in the listed units. Negative values favour the supplied label.']
    motion_changes = []
    for feat, title in FEATURES.items():
        if feat == 'mean_abs_acceleration_mps2':
            continue
        deltas = []
        for record in runs.values():
            def mean_error(policy):
                return fmean(abs(record['results'][policy][m]['signed_error'][feat]) for m in MODES)
            deltas.append(signed(mean_error('provided') - mean_error('header_only')))
        motion_changes.append([title, *deltas])
    table(lines, ['Absolute motion error change', *RUN_LABELS], motion_changes)
    lines += ['### Individual vehicle classes', '',
              'Each cell shows seed 42 / seed 43 with the supplied label. Destination hit is a percentage.']
    class_rows = []
    for mode in MODES:
        for rep in ('coordinate', 'velocity'):
            arm = [runs[rep, s]['results']['provided'][mode] for s in (42, 43)]
            paths = [(('speed_w1_mps',), 1), (('accel_w1_mps2',), 1),
                     (('endpoints', 'end_error_m', 'median'), 1), (('endpoints', 'end_cell_hit'), 100)]
            class_rows.append([mode, rep, *[' / '.join(f'{field(v, path) * scale:.3f}' for v in arm)
                                            for path, scale in paths]])
    table(lines, ['Class', 'Representation', 'Speed W1 (m/s)', 'Acceleration W1 (m/s²)',
                  'Median destination error (m)', 'Destination cell hit (%)'], class_rows)
    lines += ['## Pairing and provenance', '',
              'All four evaluations agree on these SHA-256 fingerprints. Matched support hashes each pair’s '
              'count and ordered request-ID fingerprint. The observed-summary digest covers the per-class '
              'observed features, median lengths, and the observed contrasts and pooled SDs.']
    table(lines, ['Field', 'SHA-256'], [[k, f'`{value}`'] for k, value in check.items() if k.endswith('_sha256')])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory containing coordinate-42.json, etc.')
    parser.add_argument('output', type=Path, help='Markdown output path')
    args = parser.parse_args()
    report = render(*load_runs(args.root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.pending')
    tmp.write_text(report)
    tmp.replace(args.output)


if __name__ == '__main__':
    main()
