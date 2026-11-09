"""Complete, fixed-scope LaTeX tables from a verified closed G40 export.

Presentation only: no sample exclusion, fitting, outcome-dependent ranking or
new endpoint. All five classes, eight arms and both fitted seeds are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ARMS = ('B', 'F-SV', 'J-S', 'J-V', 'J-SV', 'J-RAW', 'J-SH', 'C')
CLASSES = ('bus', 'motorcycle', 'private_car', 'bycicle', 'taxi_or_uber')
NAMES = ('Bus', 'Motorcycle', 'Private car', 'Bicycle', 'Taxi')
PHYSICAL = (
    ('duration_s', r'Duration (s)'),
    ('arc_length_m', r'Distance (m)'),
    ('mean_speed_mps', r'Mean speed (m/s)'),
    ('speed_p95_mps', r'Speed p95 (m/s)'),
    ('stopped_fraction', r'Stopped fraction'),
    ('mean_absolute_acceleration_mps2', r'Mean $|a|$ (m/s$^2$)'),
    ('longest_stop_s', r'Longest stop (s)'),
    ('stop_episodes_per_minute', r'Stops per min'),
    ('longest_steady_s', r'Longest steady (s)'),
    ('mean_absolute_lag4_speed_change_mps', r'Four-second change (m/s)'),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    assert values
    return None if any(v is None for v in values) else statistics.mean(values)


def fmt(value):
    return 'n/a' if value is None else f'{value:.4f}'


def main(args):
    evidence = json.loads(args.evidence.read_text())
    assert evidence['complete'] and evidence['all_stages']['complete']
    assert evidence['utility_metrics_independently_recomputed']
    assert len(evidence['evaluations']) == 128
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'complete': True, 'source_sha256': digest(__file__),
               'evidence_sha256': digest(args.evidence), 'tables': [],
               'scope': 'Presentation of already declared endpoints; no changed sample or metric rules.'}

    def table(name, caption, header, rows):
        columns = len(header) - 1
        text = (r'\tab{' + caption + '}{' + name + '}{\n'
                r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}' + '\n'
                r'\begin{tabularx}{\linewidth}{@{}l*{' + str(columns)
                + r'}{>{\centering\arraybackslash}X}@{}}\toprule' + '\n')
        text += ' & '.join(header) + r' \\\midrule' + '\n'
        text += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
        text += '\n' + r'\bottomrule\end{tabularx}}' + '\n'
        path = output / (name + '.tex')
        path.write_text(text)
        receipt['tables'].append(name)

    summary = evidence['report']['summary']
    for stat, description in (('w1', 'Class-marginal Wasserstein distances'),
                              ('mae', 'Request-paired mean absolute discrepancies')):
        for part in range(2):
            metrics = PHYSICAL[part * 5:(part + 1) * 5]
            rows = [[arm, *[fmt(summary[arm][f'physical/{key}/{stat}'])
                            for key, _ in metrics]] for arm in ARMS]
            table(f'g40-physical-{stat}-{part + 1}',
                  description + f', part {part + 1}. K1 direct draws, averaged equally '
                  'over classes, model seeds, windows and OD cohorts. All arms remain; '
                  'lower is closer to the reference. Units belong to each column.',
                  ['Arm', *[label for _, label in metrics]], rows)

    for signal, unit in (('speed', 'm/s'), ('acceleration', r'm/s$^2$')):
        rows = []
        for arm in ARMS:
            values = []
            for cls in CLASSES:
                selected = [v['results']['R']['by_class'][cls][f'selected_native_{signal}_w1']
                            for k, v in evidence['evaluations'].items()
                            if k.split('/')[1].rsplit('-', 1)[0] == arm]
                assert len(selected) == 16
                values.append(fmt(mean(selected)))
            rows.append([arm, *values])
        table(f'g40-class-{signal}', f'Native {signal} Wasserstein distances ({unit}) '
              'by class. Means of eight episodes and two fitted seeds, K1 direct draws. '
              'The five classes are displayed individually, without ranking or omission.',
              ['Arm', *NAMES], rows)

    rows = []
    for arm in ARMS:
        for seed in (42, 43):
            values = []
            for policy in ('selected', 'weighted'):
                for signal in ('speed', 'acceleration'):
                    selected = [v['results']['R']['macro'][f'{policy}_native_{signal}_w1']
                                for k, v in evidence['evaluations'].items()
                                if k.split('/')[1] == f'{arm}-{seed}']
                    assert len(selected) == 8
                    values.append(fmt(mean(selected)))
            rows.append([f'{arm} / {seed}', *values])
    table('g40-seed-native', 'Native distribution distances for every fitted seed. '
          'K1 is the first direct draw; all four gives equal mass to four draws per request. '
          'Speed is in m/s and acceleration in m/s$^2$. Means cover eight episodes and five classes.',
          ['Arm / seed', 'K1 speed', 'K1 accel.', 'All four speed', 'All four accel.'], rows)

    rows = []
    for episode, classes in sorted(evidence['coverage'].items()):
        rows.append([episode, *[sum(group[key] for group in classes.values()) for key in
                     ('requested', 'generated', 'physical_pairs', 'route_pairs',
                      'catalog_contains_exact_reference')]])
    table('g40-coverage', 'Final coverage by reference episode. Counts are requests, '
          'not independent vehicle populations: paired windows reuse source plans and '
          'bus templates are shared. Exact means the realised reference route is in the catalog.',
          ['Episode', 'Requested', 'Generated', 'Physical pairs', 'Route pairs', 'Exact route'], rows)

    fields = ('mean_pairwise_progress_distance', 'nearest_training_shape_distance_mean',
              'near_shape_rate_threshold_1e4', 'negative_decode_fraction', 'upper_decode_fraction')
    rows = []
    fallbacks = {}
    for arm in ARMS:
        selected = [v for k, v in evidence['diagnostics'].items()
                    if k.split('/')[1].rsplit('-', 1)[0] == arm]
        assert len(selected) == 16
        rows.append([arm, *[fmt(mean(v['by_class'][c][key] for v in selected for c in CLASSES))
                            for key in fields]])
        fallbacks[arm] = sum(v['by_class'][c]['constant_speed_fallbacks']
                             for v in selected for c in CLASSES)
    table('g40-diversity-decode', 'Progress-shape diversity and decoder accounting. '
          'Means weight classes and scene/seed cells equally. Near-shape is the fraction '
          'within $10^{-4}$ of a fixed, at-most-512-per-class training library; it is not a privacy guarantee. '
          'Negative and upper are pre-correction decoder-bin fractions.',
          ['Arm', 'Pairwise shape', 'Nearest shape', 'Near-shape rate', 'Negative', 'Upper'], rows)

    support = {}
    for key, _ in PHYSICAL:
        support[key] = {
            arm: {
                c: sorted(set(v['results']['R']['by_class'][c]['physical'][key]['n']
                              for name, v in evidence['evaluations'].items()
                              if name.split('/')[1].rsplit('-', 1)[0] == arm))
                for c in CLASSES}
            for arm in ARMS}
    receipt['physical_support_counts_by_class_cell'] = support
    receipt['constant_speed_fallbacks'] = fallbacks
    receipt['artifacts'] = {p.name: digest(p) for p in output.glob('*.tex')}
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'complete': True, 'tables': len(receipt['tables'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
