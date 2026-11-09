"""Recompute closed aggregates with independent scalar formulas for review.

Reads no trajectories, weights or partial results. Does not choose a model,
change an endpoint, or write manuscript conclusions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('B', 'F-SV', 'J-S', 'J-V', 'J-SV', 'J-RAW', 'J-SH', 'C')
CLASSES = ('bus', 'motorcycle', 'private_car', 'bycicle', 'taxi_or_uber')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def average(values):
    values = list(values)
    assert values
    return None if any(v is None for v in values) else statistics.mean(values)


def f1_from_confusion(matrix):
    assert len(matrix) == 5 and all(len(row) == 5 for row in matrix)
    assert all(x >= 0 and math.isfinite(x) for row in matrix for x in row)
    support = list(map(sum, matrix))
    prediction = [sum(row[c] for row in matrix) for c in range(5)]
    f1 = [2 * matrix[c][c] / (support[c] + prediction[c])
          if support[c] + prediction[c] else 0. for c in range(5)]
    n = sum(support)
    assert n > 0
    return {'n': n, 'accuracy': sum(matrix[c][c] for c in range(5)) / n,
            'macro_f1': statistics.mean(f1), 'per_class_f1': f1,
            'weighted_f1': sum(v * w for v, w in zip(f1, support)) / n}


def main(args):
    r = json.loads(args.evidence.read_text())
    assert r['complete'] and r['all_stages']['complete']
    assert r['utility_metrics_independently_recomputed']
    assert len(r['evaluations']) == 128 and r['utility']['fit_count'] == 30
    assert r['report']['rows'] == 640
    flat = {}
    for cell, evaluation in r['evaluations'].items():
        episode, name = cell.split('/')
        arm, seed = name.rsplit('-', 1)
        assert arm in ARMS and int(seed) in (42, 43)
        assert 400201 <= int(episode) <= 400208
        flat[cell] = {}
        for cls in CLASSES:
            row = evaluation['results']['R']['by_class'][cls]
            flat[cell][cls] = {k: v for k, v in row.items() if k != 'physical'}
            for metric, values in row['physical'].items():
                for stat in ('n', 'w1', 'mae'):
                    flat[cell][cls][f'physical/{metric}/{stat}'] = values[stat]
    recomputed = {}; classes = {}; cohorts = {}; checks = 0
    for arm in ARMS:
        cells = {key: value for key, value in flat.items()
                 if key.split('/')[1].rsplit('-', 1)[0] == arm}
        assert len(cells) == 16
        recomputed[arm] = {}; classes[arm] = {}; cohorts[arm] = {}
        for metric, reported in r['report']['summary'][arm].items():
            value = average(row[metric] for cell in cells.values() for row in cell.values())
            equal(value, reported); checks += 1
            recomputed[arm][metric] = value
            classes[arm][metric] = {cls: average(cell[cls][metric] for cell in cells.values())
                                    for cls in CLASSES}
            cohorts[arm][metric] = [average(row[metric]
                for key, cell in cells.items() if (int(key.split('/')[0]) - 400201) // 2 == cohort
                for row in cell.values()) for cohort in range(4)]
    checked_fits = {}; utility = {}
    for name, fit in r['utility']['results'].items():
        pooled = f1_from_confusion(fit['pooled']['confusion'])
        for metric in ('n', 'accuracy', 'macro_f1', 'weighted_f1'):
            equal(pooled[metric], fit['pooled'][metric])
        for a, b in zip(pooled['per_class_f1'], fit['pooled']['per_class_f1']): equal(a, b)
        nonbus = f1_from_confusion(fit['nonbus_weighted_confusion'])
        equal(statistics.mean(nonbus['per_class_f1'][1:]), fit['nonbus_source_parent_equal_macro_f1'])
        checked_fits[name] = {'pooled': pooled,
                              'nonbus_parent_equal_macro_f1': statistics.mean(nonbus['per_class_f1'][1:])}
    for regime in sorted({name.rsplit('-', 1)[0] for name in checked_fits}):
        fits = [checked_fits[f'{regime}-{seed}'] for seed in (4001, 4002)]
        utility[regime] = {
            'pooled_macro_f1_seeds': [f['pooled']['macro_f1'] for f in fits],
            'pooled_macro_f1_mean': average(f['pooled']['macro_f1'] for f in fits),
            'nonbus_parent_equal_macro_f1_seeds': [f['nonbus_parent_equal_macro_f1'] for f in fits],
            'nonbus_parent_equal_macro_f1_mean': average(f['nonbus_parent_equal_macro_f1'] for f in fits),
            'class_f1_mean': [average(f['pooled']['per_class_f1'][c] for f in fits) for c in range(5)]}
    coverage = {}
    for metric in ('requested', 'generated', 'physical_pairs', 'route_pairs', 'catalog_contains_exact_reference'):
        coverage[metric] = sum(c[metric] for episode in r['coverage'].values() for c in episode.values())
    costs = {}
    for arm in ARMS:
        receipts = [v for key, v in r['generations'].items() if key.split('/')[1].rsplit('-', 1)[0] == arm]
        seconds = sum(v['seconds'] for v in receipts); candidates = sum(v['candidates'] for v in receipts)
        costs[arm] = {'cells': len(receipts), 'candidates': candidates, 'summed_cell_seconds': seconds,
                      'seconds_per_candidate_including_cell_load_and_export': seconds / candidates,
                      'motion_network_forwards': sum(v['network_forwards'] for v in receipts),
                      'prior_network_forwards': sum(v['prior_network_forwards'] for v in receipts)}
    out = {'complete': True, 'evidence_sha256': digest(args.evidence), 'source_sha256': digest(__file__),
           'macro_endpoint_checks': checks, 'utility_confusion_formula_checks': len(checked_fits),
           'summary': recomputed, 'class_means': classes, 'cohort_means': cohorts,
           'utility': utility, 'coverage_episode_totals': coverage, 'sampling_costs': costs,
           'limits': ['Independent scalar recomputation of exported aggregates/confusions; source artifact and prediction verification are performed upstream.',
                      'Four OD cohorts with repeated windows and shared bus services; counts are not independent domains.',
                      'No fitting, model selection, endpoint changes or automatic scientific superiority claim.']}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'recomputed.json'; path.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    (args.output / 'complete.json').write_text(json.dumps({'complete': True,
        'artifacts': {'recomputed.json': digest(path)}, 'evidence_sha256': digest(args.evidence),
        'source_sha256': digest(__file__)}, indent=2) + '\n')
    print(json.dumps({'complete': True, 'macro_endpoint_checks': checks,
                      'utility_confusion_formula_checks': len(checked_fits)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
