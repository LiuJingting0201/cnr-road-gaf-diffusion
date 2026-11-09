"""Summarize all completed factorial cells; retain weighted and selected outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


METHODS = ('R',) + tuple(f'{kind}-{view}' for kind in ('Global', 'FixedRoute', 'FixedRouteT') for view in ('S', 'V', 'SV'))
KEYS = ('weighted_native_speed_w1', 'weighted_native_acceleration_w1',
        'selected_native_speed_w1', 'selected_native_acceleration_w1',
        'route_edge_usage_js', 'expected_detour', 'effective_route_count')


def mean(values):
    values = list(values)
    assert values and all(v is not None for v in values)
    return statistics.mean(values)


def summarize(source, output):
    evidence = json.loads(source.read_text())
    assert evidence['complete'] and evidence['full_batch']
    receipt = evidence['factorial']
    cells = receipt['results']
    assert receipt['complete'] and len(cells) == 4
    assert receipt['same_expanded_pool_for_all_methods']
    assert set(cells) == {f'{ep}/{seed}' for ep in (380201, 380202) for seed in (42, 43)}
    classes = tuple(next(iter(cells.values()))['results']['R']['by_class'])
    assert len(classes) == 5
    rows = []
    for cell, result in sorted(cells.items()):
        assert set(result['results']) == set(METHODS)
        for method in METHODS:
            for cls, group in result['results'][method]['by_class'].items():
                baseline = result['results']['R']['by_class'][cls]
                assert group['paired'] == baseline['paired']
                row = {'cell': cell, 'method': method, 'class': cls}
                row.update({key: group[key] for key in KEYS})
                row.update({key: group[key] for key in ('paired', 'generated', 'requested', 'route_pairs', 'selected_route_changes_vs_R')})
                for key, value in group['physical'].items():
                    row['selected_physical_' + key + '_w1'] = value['w1']
                    row['selected_physical_' + key + '_mae'] = value['mae']
                    row['selected_physical_' + key + '_n'] = value['n']
                rows.append(row)
                if method.startswith('FixedRoute'):
                    for key in ('route_edge_usage_js', 'expected_detour', 'effective_route_count'):
                        assert abs(group[key] - baseline[key]) < 1e-12
    aggregates = {}
    for method in METHODS:
        endpoints = {}
        for key in KEYS:
            value = mean(c['results'][method]['macro'][key] for c in cells.values())
            base = mean(c['results']['R']['macro'][key] for c in cells.values())
            endpoints[key] = {'mean': value, 'R_mean': base, 'relative_percent': 100*(value/base-1),
                'lower_than_R_cells': sum(c['results'][method]['macro'][key] < c['results']['R']['macro'][key] for c in cells.values()),
                'paired_cell_deltas': {cell: c['results'][method]['macro'][key]-c['results']['R']['macro'][key] for cell, c in sorted(cells.items())}}
        by_class = {}
        for cls in classes:
            by_class[cls] = {}
            for key in KEYS:
                value = mean(c['results'][method]['by_class'][cls][key] for c in cells.values())
                base = mean(c['results']['R']['by_class'][cls][key] for c in cells.values())
                by_class[cls][key] = {'mean': value, 'R_mean': base, 'relative_percent': 100*(value/base-1) if base else None}
        aggregates[method] = {'endpoints': endpoints, 'by_class': by_class}
    result = {'complete': True, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'cells': len(cells), 'class_rows': len(rows), 'all_methods': list(METHODS),
        'scope': 'Exposed G38 scenes, frozen G35 generator; exploratory mechanism diagnostic, not independent confirmation.',
        'mechanism_estimand': 'Weighted native physical-time distributions with exactly held joint route/prior-T marginals. K1 selected metrics are secondary realizations; their selected route/T can change.',
        'weighting': 'class macro within each cell, then equal two scenes and two seeds; not individual-trajectory significance',
        'factorial_generation': evidence['factorial_generation'], 'methods': aggregates}
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    fields = list(rows[0])
    assert all(set(row) == set(fields) for row in rows)
    with (output/'all-class-metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fields); writer.writeheader(); writer.writerows(rows)
    lines = ['# Complete G39 route × duration mechanism diagnostic', '',
        'All four cells and ten equal-pool policies are retained. Negative changes mean lower distribution distance; effective-route count is diversity, not an error.', '',
        '| Policy | Weighted speed W1 | Δ% | Weighted acceleration W1 | Δ% | Lower speed / acceleration cells | Route JS Δ% |',
        '|---|---:|---:|---:|---:|---|---:|']
    for method in METHODS:
        e = aggregates[method]['endpoints']; s = e[KEYS[0]]; a = e[KEYS[1]]; r = e['route_edge_usage_js']
        lines.append(f"| {method} | {s['mean']:.6f} | {s['relative_percent']:+.2f} | {a['mean']:.6f} | {a['relative_percent']:+.2f} | {s['lower_than_R_cells']}/4 / {a['lower_than_R_cells']}/4 | {r['relative_percent']:+.2f} |")
    lines += ['', 'FixedRouteT preserves the joint route and drawn-duration distribution to numerical precision; it does not force the same individual selected route/duration under every policy. Weighted native metrics isolate this distributional mechanism. Selected K1 speed/acceleration, journey/stopping endpoints and every class remain in the complete JSON/CSV; do not reinterpret selected-duration differences as a fixed-duration intervention.', '',
        'These are two already-exposed OD scenes and two noise seeds, with repeated bus services. No population significance or fresh-replication claim. The larger factorial pool is not an equal-budget comparison with historical K8.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); summarize(args.source, args.output)
