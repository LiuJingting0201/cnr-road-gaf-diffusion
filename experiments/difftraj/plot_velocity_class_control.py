"""Plot the four velocity/class-control evaluations, using their existing metrics.json files."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Vertical offsets separate both training seeds without implying uncertainty intervals.
SERIES = {
    ('coordinate', 42): ('#27699c', 'o', -.14), ('coordinate', 43): ('#27699c', '^', -.27),
    ('velocity', 42): ('#b95516', 'o', .14), ('velocity', 43): ('#b95516', '^', .27),
}
FEATURES = {
    'mean_speed_mps': 'Mean speed\n(m/s)', 'p90_speed_mps': '90th percentile\nspeed (m/s)',
    'slow_time_percent': 'Time below 1 m/s\n(percentage points)',
    'slow_entries_per_10min': 'Low-speed entries\n(per 10 min)',
    'mean_abs_acceleration_mps2': 'Mean |acceleration|\n(m/s²)',
}


def read_runs(paths):
    runs = {}
    for path in paths:
        record = json.loads(path.read_text())
        key = record['representation'], record['seed']
        if key in runs:
            raise ValueError(f'Duplicate run {key}')
        runs[key] = record
    if runs.keys() != SERIES.keys():
        raise ValueError('Expected coordinate and velocity evaluations for seeds 42 and 43')
    first = runs['coordinate', 42]
    for run in runs.values():
        for field in ('request_sha256', 'noise_sha256', 'head_sha256', 'source_manifest_sha256',
                      'classes', 'split', 'requests'):
            if run[field] != first[field]:
                raise ValueError(f'Evaluations differ at {field}')
        for key, ref in first['label_contrasts'].items():
            pair = run['label_contrasts'][key]
            if pair['paired_request_sha256'] != ref['paired_request_sha256']:
                raise ValueError(f'Matched requests differ for {key}')
            for name, feature in ref.get('features', {}).items():
                if pair['features'][name]['observed'] != feature['observed']:
                    raise ValueError(f'SUMO reference differs for {key}, {name}')
    return runs, first


def name(mode):
    return 'Passenger car' if mode == 'passenger' else mode.capitalize()


def legend(fig, include_reference=False):
    items = [Line2D([], [], linestyle='none', color=c, marker=m, markersize=5,
                    label=f'{model.capitalize()}, seed {seed}')
             for (model, seed), (c, m, _) in SERIES.items()]
    if include_reference:
        items.insert(0, Line2D([], [], color='#2a3035', marker='D', markersize=5,
                               linestyle='none', label='SUMO matched trips'))
    fig.legend(handles=items, loc='upper center', bbox_to_anchor=(.5, .935),
               ncol=len(items), frameon=False, fontsize=9, handletextpad=.3, columnspacing=1.3)


def style_axis(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_axisbelow(True)
    ax.grid(axis='x', lw=.65, color='#e5e9eb')
    ax.locator_params(axis='x', nbins=4)


def contrasts(runs, first):
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(15.2, 7.4), sharey=True)
    keys = [f'{b}_minus_{a}' for a, b in combinations(first['classes'], 2)]
    labels = []
    for row, key in enumerate(keys):
        pair = first['label_contrasts'][key]
        labels.append(f"{name(pair['second'])} − {name(pair['first'])}   ({pair['matched_pairs']})")
        if not pair['matched_pairs']:
            continue
        for ax, feature in zip(axes, FEATURES):
            ref = pair['features'][feature]
            # The per-feature direction screen is fixed by the scorer before sampling.
            opacity = 1 if ref['direction_evaluable'] else .34
            ax.scatter(ref['observed'], row, c='#2a3035', marker='D', s=27, alpha=opacity, zorder=3)
            for k, (color, marker, shift) in SERIES.items():
                value = runs[k]['label_contrasts'][key]['features'][feature]['generated']
                ax.scatter(value, row+shift, c=color, marker=marker, s=24, alpha=opacity, zorder=3)
    for ax, xlabel in zip(axes, FEATURES.values()):
        style_axis(ax)
        ax.axvline(0, color='#7b858e', lw=.85)
        for row in range(1, len(keys), 2):
            ax.axhspan(row-.5, row+.5, facecolor='#f5f6f7', zorder=-1)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels)-.5, -.5)
    fig.suptitle('How much does changing the vehicle label change generated motion?', fontsize=16, y=.976)
    legend(fig, include_reference=True)
    fig.text(.014, .043, 'Rows give the signed difference and the number of matched SUMO pairs. '
             'Generated differences hold the request and noise fixed.\n'
             'Faded marks indicate fewer than 20 matched pairs or a SUMO difference below 0.2 pooled SD '
             '(including undefined standardization). Empty rows have no matches.', fontsize=9, va='center')
    fig.subplots_adjust(left=.18, right=.99, top=.85, bottom=.18, wspace=.31)
    return fig


def quality(runs, first):
    modes = first['classes']
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.3), sharey=True)
    for row, mode in enumerate(modes):
        for k, (color, marker, shift) in SERIES.items():
            rec = runs[k]['results']['provided'][mode]
            values = (rec['speed_w1_mps'], rec['endpoints']['end_error_m']['median'],
                      100 * rec['endpoints']['end_cell_hit'])
            for ax, val in zip(axes, values):
                ax.scatter(val, row+shift, c=color, marker=marker, s=32, zorder=3)
    for ax, title in zip(axes, ('Speed distribution W₁ (m/s) ↓', 'Median destination error (m) ↓',
                                'Requested destination cell hit (%) ↑')):
        style_axis(ax)
        ax.set_xlabel(title)
        ax.set_xlim(left=0)
        for row in range(1, len(modes), 2):
            ax.axhspan(row-.5, row+.5, color='#f5f6f7', zorder=-1)
    axes[-1].set_xlim(0, 101)
    axes[0].set_yticks(range(len(modes)), [name(m) for m in modes])
    axes[0].set_ylim(len(modes)-.5, -.5)
    fig.suptitle('Motion and destination fidelity with the supplied vehicle label', fontsize=16, y=.976)
    legend(fig)
    n = [first['results']['provided'][m]['requests'] for m in modes]
    sizes = f'{n[0]:,} requests per class' if len(set(n)) == 1 else f'{sum(n):,} requests overall'
    fig.text(.015, .046, f'{sizes.capitalize()}. Speed W₁ compares interval speeds to SUMO. Distance is measured from the observed destination,\n'
             'and the hit rate counts generated endpoints inside the requested destination cell.', fontsize=9, va='center')
    fig.subplots_adjust(left=.12, right=.987, top=.80, bottom=.20, wspace=.23)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path, nargs=4, help='Four metrics.json files, in any order')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--format', choices=('pdf', 'png', 'svg'), default='pdf')
    args = parser.parse_args()
    runs, first = read_runs(args.metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    for filename, make in [('label-contrasts', contrasts), ('motion-and-endpoints', quality)]:
        fig = make(runs, first)
        fig.savefig(args.output / f'{filename}.{args.format}', dpi=180, bbox_inches='tight', pad_inches=.13)
        plt.close(fig)
