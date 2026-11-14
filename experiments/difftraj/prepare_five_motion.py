"""Align SUMO's native speed measurements to the 199 intervals used by DiffTraj."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np


MODES = ('bicycle', 'bus', 'motorcycle', 'passenger', 'taxi')
SPLITS = ('train', 'validation')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(samples):
    """Trapezoidal mean speed per native second, duration-weighted into model intervals."""
    t, x, y, v = samples.astype(np.float64, copy=False).T
    if len(t) < 120 or not np.isfinite(samples).all() or np.any(v < 0) or not np.all(np.diff(t) == 1):
        raise ValueError('Expected continuous one-second native observations and finite nonnegative speed')
    # Coordinates and velocity must use the very same fractional native indices.
    native = np.arange(len(t))
    edges = np.linspace(0, len(t) - 1, 200)
    duration = t[-1] - t[0]
    dt = duration / 199
    integrated = np.r_[0., np.cumsum((v[:-1] + v[1:]) * .5)]
    v199 = np.diff(np.interp(edges, native, integrated)) / dt
    xy = np.column_stack((np.interp(edges, native, x), np.interp(edges, native, y))).astype(np.float32)
    return xy, v199.astype(np.float32), dt


def prepare_split(data, output, split, normalization, expected):
    source = data / split
    hashes = {name: sha256(source / name) for name in ('index.npz', 'traj.npy', 'head.npy')}
    if hashes != expected['sha256']:
        raise ValueError(f'{split} release checksum mismatch')
    with np.load(source / 'index.npz', allow_pickle=False) as index:
        ids = index['ids'].tolist()
        classes = index['classes'].tolist()
    n = len(ids)
    if len(set(ids)) != n or len(classes) != n or Counter(classes) != expected['classes']:
        raise ValueError(f'{split} inconsistent release membership')
    coordinates = np.load(source / 'traj.npy', mmap_mode='r')
    headers = np.load(source / 'head.npy', mmap_mode='r')
    if coordinates.shape != (n, 200, 2) or headers.shape != (n, 8):
        raise ValueError(f'{split} invalid model array shapes')
    row_groups = defaultdict(dict)
    for i, key in enumerate(ids):
        row_groups[key.split('_')[0]][key] = i
    labels = np.array([MODES.index(label) for label in classes], dtype=np.int64)
    duration = headers[:, 2] * normalization['header_std'][1] + normalization['header_mean'][1]
    duration = np.rint(duration).astype(np.int64)
    with TemporaryDirectory(dir=output, prefix=split + '-') as stage:
        stage = Path(stage)
        interval_speeds = np.lib.format.open_memmap(stage / 'v.npy', mode='w+', dtype='f4', shape=(n, 199))
        dt_array = np.zeros(n, dtype=np.float32)
        for seed, rows in row_groups.items():
            with np.load(data / 'raw_shards' / seed / 'trips.npz', allow_pickle=False) as shard:
                keys = shard['ids'].tolist()
                types = shard['classes'].tolist()
                raw = shard['raw']
                h = shard['h']
            ends = np.r_[0, np.cumsum(h[:, 3].astype(np.int64))]
            if ends[-1] != len(raw) or raw.ndim != 2 or raw.shape[1] != 4:
                raise ValueError(f'{split}/{seed} invalid native sample layout')
            for j, key in enumerate(keys):
                i = rows.get(key)
                if i is None:
                    continue
                if classes[i] != types[j] or dt_array[i] or ends[j+1] - ends[j] != duration[i] + 1:
                    raise ValueError(f'{split}/{key} label, duration or row mismatch')
                xy, v, dt = aggregate(raw[ends[j]:ends[j+1]])
                encoded = ((xy - normalization['mean']) / normalization['std']).astype(np.float32)
                if not np.allclose(encoded, coordinates[i], rtol=1e-6, atol=2e-6):
                    raise ValueError(f'{split}/{key} speed/geometry timing mismatch')
                interval_speeds[i] = v
                dt_array[i] = dt
        if np.any(dt_array <= 0):
            raise ValueError(f'{split} missing observations')
        interval_speeds.flush()
        del interval_speeds
        np.save(stage / 'dt.npy', dt_array)
        np.save(stage / 'y.npy', labels)
        destination = output / split
        destination.mkdir(exist_ok=True)
        derived = {}
        for name in ('dt.npy', 'v.npy', 'y.npy'):
            derived[name] = sha256(stage / name)
            (stage / name).replace(destination / name)
    return {'rows': n, 'episodes': len(row_groups), 'source_sha256': hashes, 'output_sha256': derived,
            'class_counts': dict(Counter(classes)), 'interval_seconds_range': dt_array[[dt_array.argmin(), dt_array.argmax()]].tolist()}


def prepare(data, output):
    if not (data / 'VERIFIED').is_file():
        raise ValueError('Missing independent data-release verification')
    manifest = json.loads((data / 'manifest.json').read_text())
    source_sha = sha256(data / 'manifest.json')
    output.mkdir(exist_ok=True, parents=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = output / 'PREPARED.json'
        if complete.is_file():
            prior = json.loads(complete.read_text())
            if prior['manifest_sha256'] != source_sha:
                raise ValueError('Existing preparation belongs to a different release')
            for split in SPLITS:
                for name, checksum in prior['splits'][split]['output_sha256'].items():
                    if sha256(output / split / name) != checksum:
                        raise ValueError(f'{split}/{name} prepared array checksum mismatch')
            print(f'Already prepared {output}', flush=True)
            return
        if sha256(data / 'normalization.npz') != manifest['normalization_sha256']:
            raise ValueError('Normalization checksum mismatch')
        with np.load(data / 'normalization.npz') as z:
            normalization = dict(z)
        result = {'manifest_sha256': source_sha, 'classes': list(MODES), 'splits': {},
                  'speed_unit': 'm/s', 'method': 'Trapezoidal mean on each native second, time-weighted overlap with each of 199 model intervals.'}
        start = time.monotonic()
        for split in SPLITS:
            result['splits'][split] = prepare_split(data, output, split, normalization, manifest['splits'][split])
            print(f'Prepared {split} in {time.monotonic() - start:.1f} elapsed seconds', flush=True)
        result['elapsed_seconds'] = round(time.monotonic() - start, 2)
        pending = complete.with_suffix('.pending')
        pending.write_text(json.dumps(result, indent=2) + '\n')
        pending.replace(complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.out)
