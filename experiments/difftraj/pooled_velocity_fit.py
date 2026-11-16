"""Train class-conditioned DiffTraj on a shared representation of interval velocities."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from velocity_difftraj import REPRESENTATION, decode, encode, sha


CLASSES = ('bicycle', 'bus', 'motorcycle', 'passenger', 'taxi')
PER_CLASS = {'train': 40_000, 'validation': 1_000}
CHUNK = 1_024


def batches(x, head, scale):
    for start in range(0, len(x), CHUNK):
        rows = slice(start, min(start + CHUNK, len(x)))
        xy = x[rows].astype('f8') * scale['std'] + scale['mean']
        yield rows, xy, encode(xy, head[rows], scale)


def fit_scale(x, head, scale):
    # Token zero is a position offset in metres. Later tokens are velocities in m/s.
    # Both statistics pool all training vehicles, regardless of their label.
    sums = np.zeros((2, 2))
    squares = np.zeros_like(sums)
    for _, _, units in batches(x, head, scale):
        for group, samples in enumerate((units[:, :1], units[:, 1:])):
            sums[group] += samples.sum(axis=(0, 1))
            squares[group] += np.square(samples).sum(axis=(0, 1))
    counts = np.array([len(x), len(x) * 199])[:, None]
    mean = sums / counts
    variance = squares / counts - np.square(mean)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError('A shared training scale is zero or nonfinite')
    return np.repeat(mean, [1, 199], axis=0), np.repeat(np.sqrt(variance), [1, 199], axis=0)


def check_layout(manifest):
    if (manifest['mode'] != 'all' or manifest['representation'] != REPRESENTATION
            or manifest['classes'] != list(CLASSES)):
        raise ValueError('Expected the pooled, class-conditioned velocity release')
    for split, quota in PER_CLASS.items():
        if manifest['splits'][split]['classes'] != dict.fromkeys(CLASSES, quota):
            raise ValueError(f'Unexpected class counts in {split}')


def prepare(source, out):
    if not (source / 'VERIFIED').is_file():
        raise ValueError('Source release has not passed the independent audit')
    source_digest = sha(source / 'manifest.json')
    if out.exists():
        manifest = json.loads((out / 'manifest.json').read_text())
        check_layout(manifest)
        if manifest['source_manifest_sha256'] != source_digest:
            raise ValueError('Existing velocity data came from another release')
        for filename, expected in manifest['files'].items():
            if sha(out / filename) != expected:
                raise ValueError(f'Prepared file changed: {filename}')
        return
    records = json.loads((source / 'manifest.json').read_text())
    if records['normalization_sha256'] != sha(source / 'normalization.npz'):
        raise ValueError('Source normalization has changed')
    with np.load(source / 'normalization.npz') as z:
        scale = dict(z)
    manifest = dict(source_manifest_sha256=source_digest, representation=REPRESENTATION,
                    mode='all', classes=list(CLASSES), splits={}, files={},
                    normalization_fit='all training trips pooled, separate offset and velocity statistics per axis')
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / 'prepared'
        stage.mkdir()
        for split, quota in PER_CLASS.items():
            src = source / split
            dst = stage / split
            dst.mkdir()
            for name in ('head.npy', 'traj.npy', 'index.npz'):
                if sha(src / name) != records['splits'][split]['sha256'][name]:
                    raise ValueError(f'Source file changed: {split}/{name}')
            with np.load(src / 'index.npz') as index:
                labels, counts = np.unique(index['classes'], return_counts=True)
            classes = dict(zip(labels.tolist(), counts.tolist()))
            if classes != dict.fromkeys(CLASSES, quota):
                raise ValueError(f'Wrong class counts in {split}: {classes}')
            x = np.load(src / 'traj.npy', mmap_mode='r')
            head = np.load(src / 'head.npy', mmap_mode='r')
            rows = len(CLASSES) * quota
            if x.shape != (rows, 200, 2) or head.shape != (rows, 8):
                raise ValueError(f'Unexpected source shape in {split}')
            if split == 'train':
                mean, sd = fit_scale(x, head, scale)
                np.savez(stage / 'normalization.npz', **scale,
                         velocity_token_mean=mean, velocity_token_std=sd)
            tokens = np.lib.format.open_memmap(dst / 'traj.npy', mode='w+', dtype='f4', shape=x.shape)
            worst = 0.
            for chunk, xy, units in batches(x, head, scale):
                tokens[chunk] = (units - mean) / sd
                replay = decode(tokens[chunk], head[chunk], scale, mean, sd)
                err = np.max(np.abs(replay - xy))
                if not np.isfinite(err) or err > .01:
                    raise ValueError(f'{split} metres lost in round trip: {err}')
                worst = max(worst, float(err))
            tokens.flush()
            del tokens
            for name in ('head.npy', 'index.npz'):
                shutil.copyfile(src / name, dst / name)
            manifest['splits'][split] = dict(rows=rows, classes=classes, max_abs_roundtrip_error_m=worst)
        manifest['files'] = {str(f.relative_to(stage)): sha(f)
                             for f in sorted(stage.rglob('*')) if f.is_file()}
        check_layout(manifest)
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(stage, out)
    print(json.dumps(manifest['splits'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare')
    train = commands.add_parser('train')
    for command in (prep, train):
        command.add_argument('--data', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
    train.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with (args.out.parent / (args.out.name + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prepare(args.data, args.out)
    else:
        check_layout(json.loads((args.data / 'manifest.json').read_text()))
        subprocess.run([sys.executable, str(Path(__file__).with_name('five_class_train.py')),
                        '--data', str(args.data), '--speed-dir', str(args.data),
                        '--out', str(args.out), '--arm', 'C', '--mode', 'all',
                        '--seed', str(args.seed), '--epochs', '10',
                        '--batch-size', '205', '--microbatch', '205'], check=True)


if __name__ == '__main__':
    main()
