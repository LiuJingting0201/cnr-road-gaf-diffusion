"""CNR-versus-SUMO source comparison using only existing development partitions."""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cnr.data import continuous_ranges

DOMAINS = ('cnr', 'sumo')
SPLITS = ('train', 'val')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cnr_records(source, manifest):
    partitions = json.loads(manifest.read_text())['split_by_id']
    with np.load(io.BytesIO(lzma.decompress(source.read_bytes())), allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        identifiers, labels, counts = archive['ids'], archive['labels'], archive['counts']
        # The archive uses global delta coding. No test trace is turned into a record.
        coordinates = np.cumsum(archive['delta_xy'], axis=0, dtype=np.int64) / metadata['coordinate_scale']
        times = np.cumsum(archive['delta_t'], dtype=np.int64) / metadata['time_scale']
    offsets = np.r_[0, np.cumsum(counts)]
    result = {split: [] for split in SPLITS}
    for index, identifier in enumerate(identifiers):
        split = partitions[str(identifier)]
        if split not in result:
            continue
        start, end = offsets[index:index + 2]
        full = np.column_stack((times[start:end], coordinates[start:end]))
        result[split].append(record(str(identifier), int(labels[index]), full))
    return result, metadata


def sumo_records(root):
    result = {split: [] for split in SPLITS}
    hashes = {}
    for split, source_split in [('train', 'train'), ('val', 'monitor')]:
        files = sorted((root / source_split).glob('*/quarter.npz'))
        if not files:
            raise FileNotFoundError(root / source_split)
        for path in files:
            with np.load(path) as native, np.load(path.parent / 'trips.npz') as trip_archive:
                identifiers = native['ids']
                offsets, observations = native['offset'], native['txy']
                labels = trip_archive['labels']
                if not np.array_equal(identifiers, trip_archive['ids']):
                    raise ValueError(f'ID mismatch in {path}')
            for index in np.flatnonzero(labels < 6):
                start, end = offsets[index:index + 2]
                result[split].append(record(str(identifiers[index]), int(labels[index]), observations[start:end]))
            hashes[str(path)] = sha256(path)
    return result, hashes


def record(identifier, label, full):
    full = np.asarray(full, dtype=np.float64)
    if len(full) < 2 or not np.isfinite(full).all() or np.any(np.diff(full[:, 0]) <= 0):
        raise ValueError(f'Invalid source track {identifier}')
    ranges = continuous_ranges(full[:, 0], full[:, 1:])
    left, right = max(ranges, key=lambda pair: pair[1] - pair[0])
    core = full[left:right].copy()
    return {'id': identifier, 'label': label, 'full': full.copy(), 'core': core,
            'break_count': len(ranges) - 1, 'core_start': int(left), 'core_end': int(right)}


def span(row):
    return float(row['core'][-1, 0] - row['core'][0, 0])


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0}
    return {'count': len(values), 'mean': float(values.mean()),
            'p05': float(np.quantile(values, .05)), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def describe(rows, low, high):
    report = {}
    for label in range(6):
        selected = [row for row in rows if row['label'] == label]
        metrics = {key: [] for key in ('full_span_s', 'continuous_span_s', 'continuous_arc_m',
                                      'speed_mean_mps', 'speed_p95_mps', 'stopped_fraction',
                                      'absolute_acceleration_mean_mps2', 'unique_grid64_cells',
                                      'clipped_point_fraction', 'retained_point_fraction', 'break_count')}
        for row in selected:
            core, full = row['core'], row['full']
            displacement = np.linalg.norm(np.diff(core[:, 1:], axis=0), axis=1)
            velocities = displacement / np.diff(core[:, 0])
            unit = (core[:, 1:] - low) / (high - low)
            cell = np.minimum(np.floor(unit.clip(0, 1) * 64).astype(int), 63)
            metric = {'full_span_s': float(full[-1, 0] - full[0, 0]),
                      'continuous_span_s': span(row), 'continuous_arc_m': float(displacement.sum()),
                      'speed_mean_mps': float(velocities.mean()) if len(velocities) else None,
                      'speed_p95_mps': float(np.quantile(velocities, .95)) if len(velocities) else None,
                      'stopped_fraction': float((velocities < .3).mean()) if len(velocities) else None,
                      'absolute_acceleration_mean_mps2': float(np.abs(np.diff(velocities)).mean() / .25) if len(velocities) > 1 else None,
                      'unique_grid64_cells': int(len(np.unique(cell, axis=0))),
                      'clipped_point_fraction': float(np.any((unit < 0) | (unit > 1), axis=1).mean()),
                      'retained_point_fraction': len(core) / len(full), 'break_count': row['break_count']}
            for key, value in metric.items():
                if value is not None:
                    metrics[key].append(value)
        report[str(label)] = {key: summarize(value) for key, value in metrics.items()}
    return report


def write_rows(path, rows):
    core_offsets = np.r_[0, np.cumsum([len(row['core']) for row in rows])]
    full_offsets = np.r_[0, np.cumsum([len(row['full']) for row in rows])]
    counts = np.bincount([row['label'] for row in rows], minlength=6)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, ids=np.asarray([row['id'] for row in rows]),
                        labels=np.asarray([row['label'] for row in rows], dtype=np.int64),
                        txy=np.concatenate([row['core'] for row in rows]), offset=core_offsets,
                        full_txy=np.concatenate([row['full'] for row in rows]), full_offset=full_offsets,
                        duration=np.asarray([span(row) for row in rows]),
                        core_start=np.asarray([row['core_start'] for row in rows]),
                        core_end=np.asarray([row['core_end'] for row in rows]),
                        break_count=np.asarray([row['break_count'] for row in rows]),
                        eligible_512=np.asarray([len(row['core']) >= 512 for row in rows]))
    temporary.replace(path)
    return {'rows': len(rows), 'class_counts': counts.tolist(), 'sha256': sha256(path),
            'eligible_512_by_class': np.bincount([row['label'] for row in rows if len(row['core']) >= 512], minlength=6).tolist()}


def prepare(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'prepared.json').exists():
        raise FileExistsError('Preparation already complete. Use the recorded cohort, do not resample.')
    metadata = json.loads(args.encoding.read_text())
    low, high = np.array(metadata['xy_min']), np.array(metadata['xy_max'])
    original, original_metadata = cnr_records(args.canonical, args.splits)
    simulated, simulated_hashes = sumo_records(args.sumo)
    all_rows = {'cnr': original, 'sumo': simulated}
    report = {'scope': 'existing CNR train/val and SUMO train/monitor only, common first six classes',
              'classes': original_metadata['classes'], 'minimum_continuous_span_s': args.minimum_span,
              'continuity': 'dt=.25 seconds, maximum coordinate-derived speed 100 m/s, longest segment, earliest tie',
              'encoding': metadata, 'seed': 29042, 'sources': {}, 'available': {},
              'count_matched': {}, 'duration_matched': {}, 'source_diagnostics': {}, 'excluded': {}}
    report['sources'] = {'canonical_sha256': sha256(args.canonical), 'split_manifest_sha256': sha256(args.splits),
                         'encoding_metadata_sha256': sha256(args.encoding), 'sumo_quarter_hashes': simulated_hashes}
    eligible = {domain: {} for domain in DOMAINS}
    rng = np.random.default_rng(29042)
    for domain, splits in all_rows.items():
        if set(row['id'] for row in splits['train']) & set(row['id'] for row in splits['val']):
            raise ValueError(f'Train/validation ID overlap in {domain}')
        for split, rows in splits.items():
            key = f'{domain}-{split}'
            report['available'][key] = write_rows(output / f'{domain}-available-{split}.npz', rows)
            report['source_diagnostics'][key] = describe(rows, low, high)
            eligible[domain][split] = [row for row in rows if span(row) >= args.minimum_span]
            report['excluded'][key] = [{'id': row['id'], 'class': row['label'], 'continuous_span_s': span(row)}
                                       for row in rows if span(row) < args.minimum_span]
    for split in SPLITS:
        cohorts = {name: [] for name in ('cnr', 'sumo', 'sumo-duration')}
        matched_spans, random_spans = [], []
        for label in range(6):
            candidates = {domain: [row for row in eligible[domain][split] if row['label'] == label] for domain in DOMAINS}
            count = min(len(rows) for rows in candidates.values())
            if count == 0:
                raise ValueError(f'No shared eligible class {label} in {split}')
            left_indices = rng.choice(len(candidates['cnr']), count, replace=False)
            left = [candidates['cnr'][index] for index in left_indices]
            right_indices = rng.choice(len(candidates['sumo']), count, replace=False)
            right = [candidates['sumo'][index] for index in right_indices]
            costs = np.abs(np.log([span(row) for row in left])[:, None] -
                           np.log([span(row) for row in candidates['sumo']])[None, :])
            left_order, right_order = linear_sum_assignment(costs)
            assert np.array_equal(left_order, np.arange(count))
            matched = [candidates['sumo'][index] for index in right_order]
            cohorts['cnr'].extend(left)
            cohorts['sumo'].extend(right)
            cohorts['sumo-duration'].extend(matched)
            matched_spans.append({'class': label, 'count': count,
                                  'absolute_log_span_ratio': summarize(costs[left_order, right_order]),
                                  'cnr_median_span_s': float(np.median([span(row) for row in left])),
                                  'sumo_median_span_s': float(np.median([span(row) for row in matched]))})
            random_spans.append({'class': label, 'count': count,
                                 'absolute_log_span_ratio': summarize(np.abs(np.log([span(row) for row in left]) -
                                                                            np.log([span(row) for row in right])))})
        for domain, rows in cohorts.items():
            key = f'{domain}-{split}'
            summary = write_rows(output / f'{key}.npz', rows)
            summary['diagnostics'] = describe(rows, low, high)
            if domain == 'sumo-duration':
                report['duration_matched'][key] = summary
            else:
                report['count_matched'][key] = summary
        report['duration_matched'][split + '_matching'] = matched_spans
        report['count_matched'][split + '_random_pairs'] = random_spans
    report['interpretation'] = ('Count matching preserves the CNR class proportions subject to availability. '
        'Duration matching minimizes total absolute log-span discrepancy without replacement within class and partition. '
        'It preserves count and the same selected CNR IDs, but does not guarantee duration overlap. '
        'Available snapshots retain every allowed original ID including short traces for separate positive controls. '
        'No original test trajectories receive records, features, images or diagnostic statistics.')
    (output / 'prepared.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value['class_counts'] for key, value in report['count_matched'].items()
                      if isinstance(value, dict) and 'class_counts' in value}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical', type=Path, default=Path('../data/canonical/most.npz.xz'))
    parser.add_argument('--splits', type=Path, default=Path('../data/processed/most64/dataset.json'))
    parser.add_argument('--encoding', type=Path, default=Path('../data/processed/original-hilbert64grid224/dataset.json'))
    parser.add_argument('--sumo', type=Path, default=Path('../g22-job/g22-data'))
    parser.add_argument('--output', type=Path, default=Path('g29-data/raw'))
    parser.add_argument('--minimum-span', type=float, default=5.)
    prepare(parser.parse_args())
