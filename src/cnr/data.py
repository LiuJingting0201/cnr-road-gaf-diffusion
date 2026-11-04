"""Split by vehicle first; never mirror-pad or bridge temporal gaps."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ['pedestrian', 'bus', 'motorcycle', 'private_car', 'bycicle', 'taxi_or_uber']


def split_ids(frame, seed=42):
    counts = frame.groupby('vehicle_id').class2.nunique()
    if (counts != 1).any():
        raise ValueError('A vehicle has multiple classes')
    rng = np.random.default_rng(seed)
    result = {}
    for label in CLASSES:
        ids = sorted(frame.loc[frame.class2 == label, 'vehicle_id'].unique())
        rng.shuffle(ids)
        a, b = int(.8*len(ids)), int(.9*len(ids))
        for split, values in [('train', ids[:a]), ('val', ids[a:b]), ('test', ids[b:])]:
            for ident in values:
                result[ident] = split
    if set(result) != set(frame.vehicle_id):
        raise ValueError('Unknown/missing class')
    return result


def continuous_ranges(t, xy, expected_dt=.25, max_rate=100.):
    dt = np.diff(t)
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    rate = np.divide(distance, dt, out=np.full_like(distance, np.inf), where=dt > 0)
    breaks = (~np.isclose(dt, expected_dt, atol=1e-7, rtol=0)) | (rate > max_rate)
    edges = np.r_[0, np.flatnonzero(breaks)+1, len(t)]
    return list(zip(edges[:-1], edges[1:]))


def prepare(source, output, length=64, seed=42, max_per_vehicle=16):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    frame = pd.read_csv(source, dtype={'vehicle_id': str})
    required = ['timestep_time', 'vehicle_id', 'vehicle_x', 'vehicle_y', 'class2']
    if frame[required].isna().any().any():
        raise ValueError('Missing required field')
    if not np.isfinite(frame[['timestep_time', 'vehicle_x', 'vehicle_y']]).all().all():
        raise ValueError('Nonfinite required numeric field')
    if frame.duplicated(['vehicle_id', 'timestep_time']).any():
        raise ValueError('Duplicate vehicle timestamp')
    split_by_id = split_ids(frame, seed)
    arrays = {s: [] for s in ('train', 'val', 'test')}
    labels = {s: [] for s in arrays}
    manifest, excluded = [], []
    for ident, group in frame.groupby('vehicle_id', sort=True):
        group = group.sort_values('timestep_time')
        xy = group[['vehicle_x', 'vehicle_y']].to_numpy(dtype=np.float64)
        t = group.timestep_time.to_numpy(dtype=np.float64)
        split = split_by_id[ident]
        label = CLASSES.index(group.class2.iloc[0])
        candidates = []
        stride = length//2 if split == 'train' else length
        for segment, (a,b) in enumerate(continuous_ranges(t, xy)):
            for start in range(a, b-length+1, stride):
                candidates.append((segment, start))
        if not candidates:
            excluded.append({'vehicle_id': ident, 'class': CLASSES[label], 'split': split})
            continue
        indices = np.unique(np.linspace(0, len(candidates)-1, min(max_per_vehicle, len(candidates))).astype(int))
        for k in indices:
            segment, start = candidates[k]
            index = len(arrays[split])
            arrays[split].append(xy[start:start+length])
            labels[split].append(label)
            manifest.append({'split': split, 'array_index': index, 'vehicle_id': ident,
                             'class': CLASSES[label], 'segment': segment,
                             'first_time': t[start], 'last_time': t[start+length-1]})
    if any(not x for x in arrays.values()):
        raise ValueError('Empty split')
    train = np.stack(arrays['train'])
    mean, std = train.mean(axis=(0,1)), train.std(axis=(0,1))
    if (std <= 0).any():
        raise ValueError('Degenerate coordinate variance')
    output.mkdir(parents=True)
    for split in arrays:
        np.savez_compressed(output/f'{split}.npz', xy=np.stack(arrays[split]).astype(np.float32),
                            labels=np.array(labels[split], dtype=np.int64))
    pd.DataFrame(manifest).to_csv(output/'manifest.csv', index=False)
    pd.DataFrame(excluded).to_csv(output/'excluded_vehicles.csv', index=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {'source_sha256': digest, 'classes': CLASSES, 'length': length,
               'dt_raw_units': .25, 'span_raw_units': (length-1)*.25,
               'seed': seed, 'max_windows_per_vehicle': max_per_vehicle,
               'break_rate_threshold': 100., 'mean': mean.tolist(), 'std': std.tolist(),
               'windows': {s:len(v) for s,v in arrays.items()},
               'vehicles': {s:sum(v == s for v in split_by_id.values()) for s in arrays},
               'excluded_vehicles': len(excluded), 'split_by_id': split_by_id,
               'normalization_source': 'training_windows_only'}
    (output/'dataset.json').write_text(json.dumps(payload, indent=2))
    return {k:v for k,v in payload.items() if k != 'split_by_id'}
