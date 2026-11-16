"""Draw preselected SUMO requests under two vehicle labels with common sampling noise."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from five_class_evaluate import MODES, array_digest, read_observations, stratify, string_digest
from plot_supervisor_sumo_maps import map_shapes
from velocity_label_contrasts import make_pairs


# The first mode supplies the observed request. This is the canonical left side of each match.
CLASSES = [('bicycle', 'bus'), ('motorcycle', 'passenger')]
NAMES = {'bicycle': 'Bicycle', 'bus': 'Bus', 'motorcycle': 'Motorcycle', 'passenger': 'Passenger car'}
STYLES = [('#273742', '-', 1.8), ('#007a9b', '--', 1.5), ('#c35413', '-.', 1.5)]


def select_examples(data):
    with np.load(data / 'validation/index.npz') as z:
        rows, _ = stratify(z['classes'])
        ids, actual = z['ids'][rows], np.searchsorted(MODES, z['classes'][rows])
    headers = np.load(data / 'validation/head.npy', mmap_mode='r')[rows]
    with np.load(data / 'normalization.npz') as z:
        speed = headers[:, 4].astype('f8') * z['header_std'][3] + z['header_mean'][3]
        durations = headers[:, 2].astype('f8') * z['header_std'][1] + z['header_mean'][1]
    matches = make_pairs(headers, actual)
    selected = []
    for first, second in CLASSES:
        a, b = MODES.index(first), MODES.index(second)
        lrows, rrows = matches[a, b]
        eligible = np.flatnonzero(np.abs(speed[lrows] - speed[rrows]) <= .5)
        if not len(eligible):
            raise ValueError(f'No qualifying pairs for {first}/{second} within 0.5 m/s')
        order = eligible[np.argsort(ids[lrows[eligible]], kind='stable')]
        i = order[(len(order) - 1) // 2]  # Lower median for an even count.
        row, partner = int(lrows[i]), int(rrows[i])
        selected.append(dict(row=row, codes=(a, b), request_id=str(ids[row]), speed=speed[row],
                             duration=durations[row], eligible=len(order), partner=str(ids[partner]),
                             partner_speed=speed[partner]))
    metadata = dict(source_manifest_sha256=hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(),
                    request_sha256=string_digest(ids), head_sha256=array_digest(headers))
    return selected, rows, ids, actual, metadata


def sampled_paths(run, seed, selected, rows, ids, actual, metadata):
    paths, noises = {}, []
    for rep in ('coordinate', 'velocity'):
        evaluation = run / f'{rep}-{seed}' / 'eval'
        report = json.loads((evaluation / 'metrics.json').read_text())
        if report['seed'] != seed:
            raise ValueError(f'Training seed differs for {rep}')
        with np.load(evaluation / 'draw-cache.npz', allow_pickle=False) as z:
            for name, expected in dict(metadata, representation=rep).items():
                if z[name].item() != expected or report[name] != expected:
                    raise ValueError(f'{rep} evaluation differs at {name}')
            noise = z['noise_sha256'].item()
            if noise != report['noise_sha256'] or z['checkpoint_sha256'].item() != report['checkpoint_sha256']:
                raise ValueError(f'{rep} cache differs from the scored samples')
            noises.append(noise)
            np.testing.assert_array_equal(z['validation_rows'], rows)
            np.testing.assert_array_equal(z['ids'], ids)
            np.testing.assert_array_equal(z['true_labels'], actual)
            keys = z['keys']
            np.testing.assert_array_equal(keys, z['draw_rows'] * len(MODES) + z['draw_labels'])
            if len(np.unique(keys)) != len(keys):
                raise ValueError(f'{rep} cache repeats a request-label combination')
            where = {k: i for i, k in enumerate(keys)}
            k = [[s['row'] * len(MODES) + c for c in s['codes']] for s in selected]
            paths[rep] = z['xy_m'][[[where[t] for t in pair] for pair in k]].astype('f8')
        if not np.isfinite(paths[rep]).all():
            raise ValueError(f'Nonfinite paths for {rep}')
    if noises[0] != noises[1]:
        raise ValueError('Sampling noise differs between model representations')
    return paths


def road_panel(ax, roads, native, generated):
    paths = [native / 1000, *(generated / 1000)]
    united = np.concatenate(paths)
    half = max(.18, np.ptp(united, axis=0).max() * .55)
    centre = (united.max(0) + united.min(0)) / 2
    ax.add_collection(LineCollection(roads, colors='#d9dfe1', linewidths=.55, zorder=0, rasterized=True))
    for path, (color, style, lw) in zip(paths, STYLES):
        ax.plot(*path.T, color=color, ls=style, lw=lw)
        ax.plot(*path[0], marker='o', ms=4.3, mec=color, mfc='white', zorder=4)
        ax.plot(*path[-1], marker='x', ms=5.3, color=color, zorder=4)
    ax.set(xlim=centre[0]+np.array([-half, half]), ylim=centre[1]+np.array([-half, half]),
           xlabel='Local east (km)', ylabel='Local north (km)', aspect='equal')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def speed_panel(ax, paths, duration):
    for path, (color, style, lw) in zip(paths, STYLES):
        dt = duration / (len(path) - 1)
        times = (np.arange(len(path) - 1) + .5) * dt
        speed = np.linalg.norm(np.diff(path, axis=0), axis=1) / dt
        ax.plot(times, speed, color=color, ls=style, lw=lw)
    ax.set(xlabel='Elapsed time (s)', xlim=(0, duration), ylim=(0, None))
    ax.grid(axis='y', color='#e6eaec', lw=.7)
    ax.xaxis.set_major_locator(MaxNLocator(5))


def render(args):
    selected, rows, ids, actual, metadata = select_examples(args.data)
    paths = sampled_paths(args.run, args.seed, selected, rows, ids, actual, metadata)
    native = read_observations(args.data, [s['request_id'] for s in selected])
    roads = map_shapes(args.net)
    if not roads:
        raise ValueError('No roads found in SUMO network')
    plt.rcParams.update({'font.size': 9.5, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(13.4, 8.6))
    outer = fig.add_gridspec(2, 1, left=.075, right=.975, top=.90, bottom=.12, hspace=.31)
    for i, (sample, observed, (a, b)) in enumerate(zip(selected, native, CLASSES)):
        grid = outer[i].subgridspec(2, 3, height_ratios=[.145, 1], width_ratios=[1, 1.25, 1.25],
                                    hspace=.27, wspace=.24)
        header = fig.add_subplot(grid[0, :])
        header.set_axis_off()
        header.text(0, .79, f"Observed {NAMES[a].lower()} · request mean speed {sample['speed']:.1f} m/s",
                    transform=header.transAxes, va='center', fontsize=10.8, weight='bold')
        header.text(0, .15, f"SUMO trip {sample['request_id']}", transform=header.transAxes,
                    va='center', fontsize=8.5)
        tags = [f'SUMO {NAMES[a].lower()}', f'{NAMES[a]} label', f'{NAMES[b]} label']
        handles = [Line2D([], [], color=c, ls=ls, lw=w, label=tag)
                   for (c, ls, w), tag in zip(STYLES, tags)]
        header.legend(handles=handles, loc='center right', frameon=False, ncol=3, fontsize=9)
        map_ax, vel_ax, coord_ax = [fig.add_subplot(grid[1, j]) for j in range(3)]
        road_panel(map_ax, roads, observed, paths['velocity'][i])
        duration = len(observed) - 1
        np.testing.assert_allclose(duration, sample['duration'], atol=.002, rtol=1e-6)
        t = np.linspace(0, duration, 200)
        obs_grid = np.column_stack([np.interp(t, np.arange(len(observed)), c) for c in observed.T])
        for rep, ax in [('velocity', vel_ax), ('coordinate', coord_ax)]:
            speed_panel(ax, [obs_grid, *paths[rep][i]], duration)
        top = max(vel_ax.get_ylim()[1], coord_ax.get_ylim()[1])
        for ax in (vel_ax, coord_ax):
            ax.set_ylim(0, top)
        vel_ax.set_ylabel('Interval speed (m/s)')
        map_ax.set_title('Velocity diffusion · paths', fontsize=10)
        vel_ax.set_title('Velocity diffusion · speeds', fontsize=10)
        coord_ax.set_title('Coordinate diffusion · speeds', fontsize=10)
        for ax in (map_ax, vel_ax, coord_ax):
            ax.spines[['top', 'right']].set_visible(False)
        print(f"Observed {a} {sample['request_id']}, matched {b} {sample['partner']}; "
              f"request speeds {sample['speed']:.3f}, {sample['partner_speed']:.3f} m/s; "
              f"eligible pairs {sample['eligible']}")
    fig.suptitle(f'Same trip request and noise, two vehicle labels · seed {args.seed}', y=.972, fontsize=15)
    fig.text(.075, .042, 'Each example is the median-ranked observed trip ID among matched pairs within 0.5 m/s of one another in requested mean speed.\n'
             'Circles mark starts and crosses mark ends. All speed curves use 199 equal time intervals over the observed trip duration.',
             fontsize=9, va='center')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight', pad_inches=.13)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'data', 'net', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--seed', default=42, choices=(42, 43), type=int)
    render(p.parse_args())
