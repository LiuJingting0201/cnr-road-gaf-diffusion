"""Fit the original DiffTraj denoiser to start-cell offsets and interval velocities."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

MODE = 'taxi'
REPRESENTATION = 'start_offset_then_199_interval_velocities_v1'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def centers_and_dt(h, scale):
    cells = h[:, 6].astype('int64')
    if not np.array_equal(cells, h[:, 6]) or not ((cells >= 0) & (cells < 256)).all():
        raise ValueError('Invalid origin-cell attribute')
    grid = np.column_stack((cells // 16, cells % 16))
    center = scale['lo'] + (grid + .5) * (scale['hi'] - scale['lo']) / 16
    duration = h[:, 2].astype('f8') * scale['header_std'][1] + scale['header_mean'][1]
    seconds = np.rint(duration)
    if not (seconds > 0).all() or not np.allclose(seconds, duration, atol=.002, rtol=1.e-7):
        raise ValueError('Expected a positive rounded duration in the one-second release')
    return center, seconds[:, None, None] / 199


def encode(xy, head, scale):
    center, dt = centers_and_dt(head, scale)
    result = np.empty_like(xy, dtype='f8')
    result[:, 0] = xy[:, 0] - center
    result[:, 1:] = np.diff(xy, axis=1) / dt
    return result


def decode(encoded, head, scale, mean, sd):
    center, dt = centers_and_dt(head, scale)
    units = encoded.astype('f8') * sd + mean
    start = center + units[:, 0]
    return np.concatenate((start[:, None, :], start[:, None, :] + np.cumsum(units[:, 1:] * dt, axis=1)), axis=1)


def prepare(source, out):
    if not (source / 'VERIFIED').is_file():
        raise ValueError('Source release not verified')
    parent = sha(source / 'manifest.json')
    if (out/'manifest.json').is_file():
        manifest = json.loads((out/'manifest.json').read_text())
        if manifest['source_manifest_sha256'] != parent or manifest['representation'] != REPRESENTATION:
            raise ValueError('Existing representation differs')
        for file, digest in manifest['files'].items():
            if sha(out/file) != digest:
                raise ValueError(f'Prepared file changed: {file}')
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / 'release'
        stage.mkdir()
        manifest = dict(source_manifest_sha256=parent, representation=REPRESENTATION, mode=MODE,
                        splits={}, files={})
        with np.load(source/'normalization.npz') as z:
            scale = dict(z)
        for split, expected in [('train',40000), ('validation',1000)]:
            folder = source/split
            with np.load(folder/'index.npz') as z:
                rows = np.flatnonzero(z['classes'] == MODE)
                ids, labels = z['ids'][rows], z['classes'][rows]
            if len(rows) != expected:
                raise ValueError(f'Expected {expected} {MODE} trips in {split}, found {len(rows)}')
            h = np.load(folder/'head.npy', mmap_mode='r')[rows]
            xy = np.load(folder/'traj.npy', mmap_mode='r')[rows].astype('f8')*scale['std'] + scale['mean']
            units = encode(xy, h, scale)
            if split == 'train':
                mean, sd = [np.empty((200,2),'f8') for _ in range(2)]
                for token_slice in [slice(0,1),slice(1,None)]:
                    samples = units[:, token_slice]
                    mean[token_slice] = samples.mean(axis=(0,1))
                    sd[token_slice] = samples.std(axis=(0,1))
                if not (sd > 0).all():
                    raise ValueError('Degenerate training scale')
                np.savez(stage/'normalization.npz', **scale, velocity_token_mean=mean, velocity_token_std=sd)
            encoded = ((units-mean)/sd).astype('f')
            recovery = np.abs(decode(encoded, h, scale, mean, sd)-xy).max()
            if not np.isfinite(recovery) or recovery > .01:
                raise ValueError(f'Round-trip position error {recovery} m')
            dest = stage/split
            dest.mkdir()
            np.save(dest/'traj.npy', encoded)
            np.save(dest/'head.npy', h)
            np.savez_compressed(dest/'index.npz', ids=ids, classes=labels)
            manifest['splits'][split] = dict(rows=len(rows), max_abs_roundtrip_error_m=float(recovery))
            del units, encoded, xy
        manifest['files']={str(f.relative_to(stage)):sha(f) for f in sorted(stage.rglob('*')) if f.is_file()}
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        os.replace(stage,out)
    print(json.dumps(manifest['splits'],indent=2))


def sample_checkpoint(args):
    import torch
    from five_class_evaluate import MODES, sample, stratify
    from five_class_train import make_model

    velocity = args.representation == 'velocity'
    prepared = json.loads((args.prepared/'manifest.json').read_text())
    if prepared['source_manifest_sha256'] != sha(args.data/'manifest.json') or prepared['representation'] != REPRESENTATION:
        raise ValueError('Prepared/source manifest mismatch')
    with np.load(args.prepared/'normalization.npz') as z:
        scale = dict(z)
    folder = args.data/'validation'
    with np.load(folder/'index.npz') as z:
        labels = z['classes']
        rows = np.flatnonzero(labels == MODE)
        ids = z['ids'][rows]
    head = np.load(folder/'head.npy', mmap_mode='r')[rows]
    pool, _ = stratify(labels)
    rng = np.random.default_rng(42)
    prior = rng.standard_normal((len(pool),2,200),dtype='f')
    prior_rows = pool[labels[pool] == MODE]
    missing = np.setdiff1d(rows,prior_rows)
    all_rows = np.r_[prior_rows,missing]
    noise = np.concatenate((prior[labels[pool] == MODE],rng.standard_normal((len(missing),2,200),dtype='f')))
    assert np.array_equal(np.sort(all_rows),rows) and len(rows)==1000
    noise = noise[np.argsort(all_rows)]
    ck = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg = json.loads((args.checkpoint.parent/'config.json').read_text())
    expected_manifest = (args.prepared if velocity else args.data)/'manifest.json'
    if cfg['release_manifest_sha256'] != sha(expected_manifest) or (ck['mode'],ck['arm']) != (MODE,'A'):
        raise ValueError('Checkpoint does not correspond to requested representation')
    net = make_model('A').eval()
    net.load_state_dict(ck['weights'])
    result = sample(net,head,np.full(len(head),MODES.index(MODE),dtype='i8'),noise,
                    {'std':np.ones(2), 'mean':np.zeros(2)},64)
    xy = (decode(result,head,scale,scale['velocity_token_mean'],scale['velocity_token_std'])
          if velocity else result*scale['std']+scale['mean'])
    if not np.isfinite(xy).all():
        raise FloatingPointError('Invalid decoded coordinates')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pending = args.output.with_suffix('.pending')
    with pending.open('wb') as stream:
        np.savez_compressed(stream,ids=ids,xy_m=xy.astype('f'),
                            noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                            representation=args.representation,training_seed=ck['seed'],epoch=ck['epoch'])
    pending.replace(args.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prepare_p=sub.add_parser('prepare')
    prepare_p.add_argument('--data',type=Path,required=True)
    prepare_p.add_argument('--out',type=Path,required=True)
    train=sub.add_parser('train')
    train.add_argument('--data',type=Path,required=True)
    train.add_argument('--out',type=Path,required=True)
    train.add_argument('--seed',type=int,required=True)
    train.add_argument('--epochs',type=int,default=50)
    sp=sub.add_parser('sample')
    sp.add_argument('--data',type=Path,required=True)
    sp.add_argument('--prepared',type=Path,required=True)
    sp.add_argument('--checkpoint',type=Path,required=True)
    sp.add_argument('--representation',choices=('velocity','coordinate'),required=True)
    sp.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with (args.out.parent/(args.out.name+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prepare(args.data,args.out)
    elif args.command=='train':
        if json.loads((args.data/'manifest.json').read_text())['representation'] != REPRESENTATION:
            raise ValueError('Velocity preparation required')
        subprocess.run([sys.executable,'-m','five_class_train','--data',str(args.data),
                        '--speed-dir',str(args.data),'--out',str(args.out),'--arm','A','--mode',MODE,
                        '--epochs',str(args.epochs),'--batch-size','205','--microbatch','205',
                        '--seed',str(args.seed)],check=True)
    else:
        sample_checkpoint(args)

if __name__=='__main__':
    main()
