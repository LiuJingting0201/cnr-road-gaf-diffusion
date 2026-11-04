"""Create evaluation references and future-free requests only after lock verification."""
import argparse
import io
import json
import lzma
from pathlib import Path
import time
import numpy as np
from cnr.data import CLASSES, continuous_ranges
from cnr.final_protocol import verify_lock, sha256, validate_requests
from cnr.original_gaf import trajectory_sequence, encode_frozen
from cnr.route_paths import RoutePaths

p = argparse.ArgumentParser(); p.add_argument('--lock', required=True)
p.add_argument('--output', default='data/processed/final-test-e7'); a = p.parse_args()
lock = verify_lock(a.lock)
package = Path('data/canonical/most.npz.xz')
if sha256(package) != lock['canonical_expected_sha256']:
    raise ValueError('Canonical input differs from frozen development source')
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
try:
    z = np.load(io.BytesIO(lzma.decompress(package.read_bytes())), allow_pickle=False)
    raw_meta = json.loads(str(z['metadata']))
    xy = np.cumsum(z['delta_xy'], axis=0, dtype=np.int64) / raw_meta['coordinate_scale']
    times = np.cumsum(z['delta_t'], dtype=np.int64) / raw_meta['time_scale']
    offsets = np.r_[0, np.cumsum(z['counts'])]
    splits = json.loads(Path('data/processed/most64/dataset.json').read_text())['split_by_id']
    base = json.loads(Path('data/processed/gaf-duration-e2/length512/dataset.json').read_text())
    coords, sequences, ids, labels, first_times, rejected = [], [], [], [], [], []
    clipping = 0; test_ids = 0
    for j, ident in enumerate(z['ids']):
        if splits[str(ident)] != 'test': continue
        test_ids += 1; begin, end = offsets[j:j+2]
        x, t = xy[begin:end], times[begin:end]
        left, right = max(continuous_ranges(t, x), key=lambda r: r[1] - r[0])
        if right - left < 512:
            rejected.append({'id': str(ident), 'label': int(z['labels'][j]), 'reason': 'insufficient_contiguous_duration'})
            continue
        start = left + (right - left - 512) // 2; window = x[start:start+512]
        try:
            seq, clip = trajectory_sequence(window, np.array(base['xy_min']), np.array(base['xy_max']), eps=base['hilbert_eps'])
        except ValueError as ex:
            rejected.append({'id': str(ident), 'label': int(z['labels'][j]), 'reason': 'original_encoding: ' + str(ex)})
            continue
        coords.append(window); sequences.append(seq); ids.append(str(ident)); labels.append(int(z['labels'][j]))
        first_times.append(float(t[start])); clipping += clip
    (out / 'encoding_rejections.json').write_text(json.dumps(rejected, indent=2))
    if not ids: raise RuntimeError('No encodable test windows; retain coverage failure')
    for split in ['train', 'val']:
        known = np.load(f'data/processed/gaf-duration-e2/length512/{split}.npz')['ids']
        assert set(ids).isdisjoint(set(known.tolist()))
    images, channel_clipped = encode_frozen(sequences, np.array(base['channel_min']), np.array(base['channel_max']))
    coords = np.asarray(coords, dtype=np.float32); labels = np.asarray(labels, dtype=np.int64)
    reference = out / 'reference'; reference.mkdir()
    np.savez_compressed(reference / 'test.npz', xy=coords, images=images, sequences=np.asarray(sequences),
                        labels=labels, ids=np.asarray(ids), first_time=np.asarray(first_times))
    (reference / 'dataset.json').write_text(json.dumps(base | {'scope': 'Locked final test references; training bounds unchanged'}, indent=2))
    matcher = RoutePaths('data/maps/most.net.xml', radius=2.)
    starts, accepted_labels, sources, progress, route_rejections = [], [], [], [], []
    for i, (track, label) in enumerate(zip(coords, labels)):
        reason = None; rec = None; name = CLASSES[int(label)]
        if label == 0: reason = 'pedestrian_not_in_vehicle_pilot'
        elif np.linalg.norm(track[-1] - track[0]) <= 5: reason = 'endpoint_displacement_le5m'
        else:
            trace = matcher.trace(track, name)
            if trace is None: reason = 'ambiguous_or_incompatible_trace'
            else:
                rec = matcher.reconstruct(trace, name)
                if rec is None or rec['progress'][-1] >= 4084: reason = 'reconstruction_or_horizon'
        if reason:
            route_rejections.append({'source_index': i, 'label': int(label), 'reason': reason}); continue
        starts.append(track[0]); accepted_labels.append(label); sources.append(i); progress.append(rec['progress'])
    (out / 'route_rejections.json').write_text(json.dumps(route_rejections, indent=2))
    requests = {'starts': np.asarray(starts, dtype=np.float32).reshape(-1, 2),
                'labels': np.asarray(accepted_labels, dtype=np.int64), 'source_index': np.asarray(sources, dtype=np.int64)}
    validate_requests(requests)
    np.savez_compressed(out / 'requests.npz', **requests)
    np.savez_compressed(reference / 'progress.npz', source_index=requests['source_index'],
                        progress=np.asarray(progress, dtype=np.float32))
    report = {'complete': True, 'locked_commit': lock['git_commit'], 'lock_sha256': sha256(a.lock),
              'test_ids_total': test_ids, 'encodable_windows': len(ids), 'strict_generation_requests': len(starts),
              'class_counts': {CLASSES[c]: int(np.sum(labels == c)) for c in range(6)},
              'generation_class_counts': {CLASSES[c]: int(np.sum(requests['labels'] == c)) for c in range(6)},
              'coordinate_clipped_points': clipping, 'channel_clipped_values': channel_clipped,
              'test_reference_sha256': sha256(reference / 'test.npz'), 'requests_sha256': sha256(out / 'requests.npz')}
    (out / 'complete.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)
except Exception as ex:
    (out / 'failed.json').write_text(json.dumps({'error': str(ex), 'time': time.time()})); raise
