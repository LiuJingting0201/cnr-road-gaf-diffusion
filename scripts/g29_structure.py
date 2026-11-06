"""Compare spatial and motion class structure before any image encoding.

Inputs are development-only, class-count-matched longest continuous trajectories.
Spatial features are derived from an arc-length parameterization and contain no
clock, residence counts or speed. Motion statistics have their own readout.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


CLASSES = ['pedestrian', 'bus', 'motorcycle', 'private_car', 'bycicle', 'taxi_or_uber']
PHYSICAL_NAMES = [
    'mean_speed_mps', 'median_speed_mps', 'p95_speed_mps', 'std_speed_mps',
    'stopped_fraction', 'mean_absolute_acceleration_mps2',
    'p95_absolute_acceleration_mps2', 'mean_absolute_four_second_speed_change_mps',
]
GEOMETRY_NAMES = [
    'start_x', 'start_y', 'end_x', 'end_y', 'arc_centroid_x', 'arc_centroid_y',
    'arc_spread_x', 'arc_spread_y', 'min_x', 'min_y', 'max_x', 'max_y',
    'log_arc_length_m', 'endpoint_distance_over_length',
    'total_absolute_turn', 'p90_absolute_turn',
] + [f'arc_occupancy_8x8_cell_{i}' for i in range(64)]


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {'count': int(len(values)), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'p05': float(np.quantile(values, .05)),
            'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def resample_arc(xy, count):
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    length = distances[-1]
    if length <= 1e-9:
        return np.repeat(xy[:1], count, axis=0), float(length)
    # Removing zero-length steps makes the interpolation independent of dwell time.
    keep = np.r_[True, np.diff(distances) > 0]
    target = np.linspace(0, length, count)
    result = np.column_stack([np.interp(target, distances[keep], axis[keep]) for axis in xy.T])
    return result, float(length)


def occupancy(xy, low, high, size):
    unit = np.clip((xy - low) / (high - low), 0, np.nextafter(1., 0.))
    cells = np.floor(size * unit).astype(int)
    histogram = np.bincount(size * cells[:, 0] + cells[:, 1], minlength=size * size).astype(float)
    return histogram / histogram.sum()


def trajectory_features(track, low, high):
    t, xy = track[:, 0], track[:, 1:].astype(float)
    dt = np.diff(t)
    if len(track) < 3 or not np.isfinite(track).all() or np.any(dt <= 0):
        raise ValueError('Invalid continuous observation sequence')
    if not np.allclose(dt, .25, atol=1e-7, rtol=0):
        raise ValueError('The source audit requires native quarter-second continuity')
    if t[-1] - t[0] < 5:
        raise ValueError('Motion features need a five-second source span')
    route, length = resample_arc(xy, 256)
    coarse, _ = resample_arc(xy, 32)
    step = np.diff(coarse, axis=0)
    heading = np.arctan2(step[:, 1], step[:, 0])
    turns = np.abs(np.angle(np.exp(1j * np.diff(heading))))
    straightness = np.linalg.norm(xy[-1] - xy[0]) / max(length, 1e-9)
    geometry = np.r_[xy[0], xy[-1], route.mean(0), route.std(0), route.min(0), route.max(0),
                     np.log(max(length, 1e-9)), straightness, turns.sum(), np.quantile(turns, .9),
                     occupancy(route, low, high, 8)]
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    middle = .5 * (t[1:] + t[:-1])
    acceleration = np.diff(speed) / np.diff(middle)
    valid = middle + 4 <= middle[-1]
    lag = np.abs(np.interp(middle[valid] + 4, middle, speed) - speed[valid])
    physical = np.array([speed.mean(), np.median(speed), np.quantile(speed, .95), speed.std(),
                         (speed < .3).mean(), np.abs(acceleration).mean(),
                         np.quantile(np.abs(acceleration), .95), lag.mean()])
    diagnostics = np.r_[t[-1] - t[0], length, straightness, route.mean(0), xy[0], xy[-1], physical]
    return geometry, physical, coarse, occupancy(route, low, high, 16), diagnostics


def dataset(path, low, high):
    data = read(path)
    identifiers, labels = data['ids'], data['labels']
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(f'Repeated IDs in {path}')
    if np.unique(labels).tolist() != list(range(6)):
        raise ValueError(f'Missing or extra participant classes in {path}')
    observations, offsets = data['txy'], data['offset']
    rows = [trajectory_features(observations[start:end], low, high)
            for start, end in zip(offsets[:-1], offsets[1:])]
    geometry, physical, routes, histograms, diagnostics = map(np.array, zip(*rows))
    if not all(np.isfinite(values).all() for values in (geometry, physical, routes, diagnostics)):
        raise ValueError(f'Nonfinite source features in {path}')
    return {'ids': identifiers, 'labels': labels, 'spatial': geometry, 'motion': physical,
            'routes': routes, 'histograms': histograms, 'diagnostics': diagnostics}


def classification(labels, predictions):
    confusion = np.bincount(6 * labels + predictions, minlength=36).reshape(6, 6)
    f1 = 2 * np.diag(confusion) / np.maximum(confusion.sum(0) + confusion.sum(1), 1)
    recall = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    return {'accuracy': float(np.trace(confusion) / len(labels)), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall.mean()), 'confusion': confusion.tolist(),
            'per_class_f1': dict(zip(CLASSES, f1.tolist())),
            'per_class_recall': dict(zip(CLASSES, recall.tolist()))}


def readouts(train, validation):
    reports = {}
    weights = compute_sample_weight('balanced', train['labels'])
    for kind in ('spatial', 'motion'):
        scaler = StandardScaler().fit(train[kind])
        fitting, evaluation = scaler.transform(train[kind]), scaler.transform(validation[kind])
        models = {
            'linear': LogisticRegression(C=1., max_iter=3000, solver='lbfgs', random_state=29042),
            'nonlinear': HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=29042),
        }
        reports[kind] = {}
        for name, model in models.items():
            model.fit(fitting, train['labels'], sample_weight=weights)
            reports[kind][name] = {
                'train': classification(train['labels'], model.predict(fitting)),
                'val': classification(validation['labels'], model.predict(evaluation)),
            }
    return reports


def route_neighbours(train, validation):
    # A pointwise Euclidean RMS after arc-length resampling. No timing remains.
    distances = cdist(validation['routes'].reshape(len(validation['ids']), -1),
                      train['routes'].reshape(len(train['ids']), -1)) / np.sqrt(32)
    nearest = np.argmin(distances, axis=1)
    minimum = distances[np.arange(len(nearest)), nearest]
    prediction = train['labels'][nearest]
    per_class = {}
    for label, name in enumerate(CLASSES):
        use = validation['labels'] == label
        own = distances[use][:, train['labels'] == label].min(1)
        other = distances[use][:, train['labels'] != label].min(1)
        per_class[name] = {'nearest_train_RMS_m': summarize(minimum[use]),
                           'same_class_RMS_m': summarize(own), 'different_class_RMS_m': summarize(other),
                           'same_class_closer_fraction': float((own < other).mean()),
                           'fraction_below_100m': float((minimum[use] < 100).mean()),
                           'fraction_below_250m': float((minimum[use] < 250).mean()),
                           'exact_route_fraction_tolerance_1mm': float((minimum[use] < .001).mean())}
    return {'definition': '32 equidistant arc-length points, original absolute XY, directed start-to-end order',
            'classification': classification(validation['labels'], prediction),
            'nearest_train_RMS_m': summarize(minimum),
            'fraction_below_100m': float((minimum < 100).mean()),
            'fraction_below_250m': float((minimum < 250).mean()),
            'exact_route_fraction_tolerance_1mm': float((minimum < .001).mean()),
            'by_class': per_class}


def source_summary(data):
    names = ['duration_s', 'arc_length_m', 'straightness', 'arc_centroid_x', 'arc_centroid_y',
             'start_x', 'start_y', 'end_x', 'end_y', *PHYSICAL_NAMES]
    return {name: {key: summarize(data['diagnostics'][data['labels'] == label, column])
                   for column, key in enumerate(names)} for label, name in enumerate(CLASSES)}


def occupancy_structure(data):
    means = np.array([data['histograms'][data['labels'] == label].mean(0) for label in range(6)])
    divergence = np.array([[jensenshannon(left, right, base=2) ** 2 for right in means] for left in means])
    return {'metric': 'Jensen-Shannon divergence in bits of class-mean arc occupancy on the common 16x16 grid',
            'classes': CLASSES, 'pairwise': divergence.tolist(),
            'mean_off_diagonal': float(divergence[np.triu_indices(6, k=1)].mean())}, means


def run(args):
    prepared = json.loads((args.raw / 'prepared.json').read_text())
    bounds = prepared['encoding']
    low, high = np.array(bounds['xy_min']), np.array(bounds['xy_max'])
    output = {'scope': 'existing development partitions, continuous source trajectories, six shared participant classes',
              'interpretation': 'Fixed diagnostic readouts. The CNR validation IDs share one export with training, '
                                'while SUMO validation comes from separate traffic episodes. These results measure '
                                'association under those splits, not causal loss of vehicle behaviour.',
              'models': {'linear': 'balanced sample weights, train-standardized logistic regression C=1, fixed',
                         'nonlinear': 'balanced sample weights, HistGradientBoosting max_iter=100, early_stopping=False, fixed'},
              'feature_names': {'spatial': GEOMETRY_NAMES, 'motion': PHYSICAL_NAMES},
              'uncertainty': 'Point estimates only. No tuning on validation and no independence claim for source routes.',
              'spatial_bounds': {'low': low.tolist(), 'high': high.tolist()}, 'domains': {}}
    occupancy = {}
    count_reference = None
    for domain in args.domains:
        train = dataset(args.raw / f'{domain}-train.npz', low, high)
        validation = dataset(args.raw / f'{domain}-val.npz', low, high)
        if set(train['ids']) & set(validation['ids']):
            raise ValueError(f'ID leakage in {domain}')
        counts = {split: np.bincount(data['labels'], minlength=6).tolist()
                  for split, data in [('train', train), ('val', validation)]}
        if count_reference is None:
            count_reference = counts
        if counts != count_reference:
            raise ValueError('Source domains must have the same per-class counts')
        occupancy_report = {}
        for split, data in [('train', train), ('val', validation)]:
            occupancy_report[split], occupancy[(domain, split)] = occupancy_structure(data)
        output['domains'][domain] = {
            'counts': counts, 'source_distributions': {'train': source_summary(train), 'val': source_summary(validation)},
            'readouts': readouts(train, validation), 'route_neighbours': route_neighbours(train, validation),
            'class_occupancy': occupancy_report,
        }
        print(json.dumps({'domain': domain, 'readouts': {
            kind: {model: report['val']['macro_f1'] for model, report in models.items()}
            for kind, models in output['domains'][domain]['readouts'].items()}}), flush=True)
    output['same_class_source_occupancy_divergence_bits'] = {
        domain: {split: dict(zip(CLASSES, [float(jensenshannon(a, b, base=2) ** 2)
                                         for a, b in zip(occupancy[('cnr', split)], occupancy[(domain, split)])]))
                 for split in ('train', 'val')}
        for domain in args.domains if domain != 'cnr'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False))
    temporary.replace(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('g29-data/raw'))
    parser.add_argument('--output', type=Path, default=Path('g29-data/structure.json'))
    parser.add_argument('--domains', nargs='+', default=['cnr', 'sumo', 'sumo-duration'])
    run(parser.parse_args())
