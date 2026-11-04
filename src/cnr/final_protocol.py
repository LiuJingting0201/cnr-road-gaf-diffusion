"""Fail-closed final-test lock and future-free synthesis condition schema."""
import hashlib
import json
from pathlib import Path
import numpy as np

GROUPS = ['real_only', 'repeat', 'timewarp', 'synthetic_random', 'synthetic_gaf', 'synthetic_tcn']
REQUEST_KEYS = {'starts', 'labels', 'source_index'}
CONDITION_KEYS = REQUEST_KEYS | {'route', 'route_length', 'condition_id'}


def frozen_inputs():
    files = ['docs/FINAL_TEST_PROTOCOL.md', 'data/maps/most.net.xml',
             'data/processed/most64/dataset.json', 'data/processed/original-hilbert64grid224/dataset.json',
             'data/processed/gaf-duration-e2/length512/dataset.json',
             'data/processed/gaf-duration-e2/length512/train.npz',
             'data/processed/long-path-e3-512/train.npz', 'data/processed/long-path-e3-512/dataset.json',
             'data/processed/long-path-e3-512/route_prior.json', 'data/processed/augmentation-e5/dataset.json']
    for seed in [42, 43, 44]:
        for directory, weight in [(f'runs/long-path-e3/seed{seed}', 'latest.pt'),
                                  (f'runs/gaf-duration-e2/length512-seed{seed}', 'final.pt'),
                                  (f'runs/long-selection-e4/tcn-seed{seed}', 'latest.pt')]:
            files.extend([f'{directory}/{weight}', f'{directory}/config.json'])
    for arch in ['tcn', 'gru']:
        for group in GROUPS:
            for seed in [100, 101, 102]:
                directory = f'runs/augmentation-e5/{arch}-{group}-seed{seed}'
                files.extend([f'{directory}/final.pt', f'{directory}/config.json'])
    return files


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_lock(path, root=Path('.')):
    """Called before opening test rows or loading synthesis models."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get('status') != 'locked' or not manifest.get('checks_passed'):
        raise ValueError('Final test is not locked and checked')
    files = manifest.get('files', {})
    if not set(frozen_inputs()).issubset(files) or not manifest.get('git_commit'):
        raise ValueError('Incomplete final-test lock')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid manifest path')
        if sha256(Path(root) / relative) != expected:
            raise ValueError(f'Locked file changed: {name}')
    return manifest


def validate_requests(requests):
    if set(requests) != REQUEST_KEYS:
        raise ValueError('Only starts, labels and source indices may reach the planner')
    starts, labels, source = [np.asarray(requests[k]) for k in ['starts', 'labels', 'source_index']]
    n = len(starts)
    if starts.shape != (n, 2) or labels.shape != (n,) or source.shape != (n,):
        raise ValueError('Invalid request shapes')
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(source.dtype, np.integer):
        raise ValueError('Labels and source indices must be integers')
    if (not np.isfinite(starts).all() or not np.isin(labels, [1, 2, 3, 4, 5]).all()
            or len(np.unique(source)) != n or np.any(source < 0)):
        raise ValueError('Invalid vehicle requests')


def plan_conditions(requests, sample_path, *, k=8, seed=20261109):
    """No reference/future argument. A failed K-pool is recorded as unserved."""
    validate_requests(requests)
    rng = np.random.default_rng(seed)
    records = {key: [] for key in CONDITION_KEYS}; failures = []
    for condition_id, (start, label, source) in enumerate(zip(requests['starts'], requests['labels'], requests['source_index'])):
        draws = [sample_path(start.copy(), int(label), rng) for _ in range(k)]
        missing = [i for i, draw in enumerate(draws) if draw is None]
        if missing:
            failures.append({'condition_id': condition_id, 'source_index': int(source),
                             'failed_candidate_slots': missing, 'reason': 'incomplete_K_pool'})
            continue
        for route, length in draws:
            values = {'starts': start, 'labels': label, 'source_index': source,
                      'condition_id': condition_id, 'route': route, 'route_length': length}
            for key, value in values.items(): records[key].append(value)
    result = {key: np.asarray(values, dtype=np.int64 if key in ['labels', 'source_index', 'condition_id'] else np.float32)
              for key, values in records.items()}
    if records['labels']:
        validate_conditions(result, k=k)
    return result, failures


def validate_conditions(conditions, *, k=8):
    if set(conditions) != CONDITION_KEYS:
        raise ValueError('Unexpected synthesis input, possibly reference future')
    n = len(conditions['labels'])
    if (not n or any(conditions[k].shape != (n,) for k in ['labels', 'source_index', 'condition_id'])
            or not np.isin(conditions['labels'], [1, 2, 3, 4, 5]).all()
            or conditions['starts'].shape != (n, 2) or conditions['route'].shape != (n, 1025, 2)
            or conditions['route_length'].shape != (n,) or np.any(conditions['route_length'] <= 0)
            or not all(np.isfinite(v).all() for v in conditions.values())):
        raise ValueError('Invalid synthesis conditions')
    for c in np.unique(conditions['condition_id']):
        ix = np.flatnonzero(conditions['condition_id'] == c)
        if (len(ix) != k or len(np.unique(conditions['source_index'][ix])) != 1
                or len(np.unique(conditions['labels'][ix])) != 1
                or not np.all(conditions['starts'][ix] == conditions['starts'][ix[0]])):
            raise ValueError('Unpaired candidate pool')
