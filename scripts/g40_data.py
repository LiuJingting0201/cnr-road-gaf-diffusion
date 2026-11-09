"""Lossless frozen-prefix caches, aligned to immutable training identities."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from g33_train import digest, save
from g35c_teachers import load_cache
from g28_teachers import image_inputs
from g40_model import reader, split_reader

VIEWS = ('gaf-spatial', 'gaf-speed', 'raw-spatial', 'raw-speed')


def source_states(teachers, raw_teachers, kind):
    root = raw_teachers if kind == 'raw' else teachers
    result = {}; sources = {}
    for view in ('spatial', 'speed'):
        folder = Path(root)/(view+'-42'); r = json.loads((folder/'complete.json').read_text())
        assert r['complete'] and not (folder/'failed.json').exists()
        sha = digest(folder/'model.pt'); assert sha == r['artifacts']['model.pt']['sha256']
        result[view] = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
        sources[view] = sha
    return result, sources


def prepare(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    try:
        torch.set_num_threads(2); torch.manual_seed(40001); torch.backends.cudnn.benchmark = False
        kind, view = args.view.split('-')
        states, sources = source_states(args.teachers, args.raw_teachers, kind)
        model = reader(kind, view); model.load_state_dict(states[view]); model.cuda().eval()
        prefix, tail, head = split_reader(model, kind)
        if kind == 'gaf':
            receipt, rows, values = load_cache(args.cache, view)
            input_receipt = digest(args.cache/'complete.json')
        else:
            receipt = json.loads((args.raw_cache/'complete.json').read_text())
            assert receipt['complete'] and digest(args.raw_cache/'raw.npz') == receipt['raw_sha256']
            with np.load(args.raw_cache/'raw.npz', allow_pickle=False) as f:
                rows = {k: f[k] for k in ('ids','parent_ids','labels','partition')}; values = f[view]
            input_receipt = digest(args.raw_cache/'complete.json')
        train = np.flatnonzero(rows['partition'] == 'train'); val = rows['partition'] == 'val'
        assert not set(rows['parent_ids'][train]) & set(rows['parent_ids'][val])
        root = args.raw_teachers if kind == 'raw' else args.teachers
        refpath = root/(view+'-42')/'train-predictions.npz'
        rr = json.loads((root/(view+'-42')/'complete.json').read_text())
        assert digest(refpath) == rr['artifacts'][refpath.name]['sha256']
        with np.load(refpath, allow_pickle=False) as f:
            lookup = {str(s): i for i, s in enumerate(f['ids'])}
            order = np.array([lookup[str(s)] for s in rows['ids'][train]])
            assert np.array_equal(rows['labels'][train], f['labels'][order])
            assert np.array_equal(rows['parent_ids'][train], f['parent_ids'][order])
            reference = f['features'][order]
        shards = []; artifacts = {}; max_error = 0.; byte_count = 0
        for begin in range(0, len(train), 256):
            if shutil.disk_usage(out).free < 12*1024**3: raise RuntimeError('12GiB disk reserve')
            if time.monotonic()-start > 1800: raise TimeoutError('30 minute per-prefix limit')
            end = min(begin+256, len(train)); batches = []
            for a in range(begin, end, 64):
                b = min(a+64, end); ix = train[a:b]
                with torch.no_grad():
                    x = image_inputs(view, values[ix]) if kind == 'gaf' else torch.as_tensor(values[ix], device='cuda')
                    h = prefix(x); v = tail(h); feature = v.mean(tuple(range(2, v.ndim))).cpu().numpy()
                error = np.abs(feature-reference[a:b]); max_error = max(max_error, float(error.max()))
                assert np.allclose(feature, reference[a:b], atol=2e-4, rtol=2e-4), (args.view, a, float(error.max()))
                batches.append(h.cpu().numpy())
            array = np.concatenate(batches); assert array.dtype == np.float32
            filename = f'prefix-{begin:06d}-{end:06d}.npz'; path = out/filename
            np.savez_compressed(path, prefix=array)
            with np.load(path, allow_pickle=False) as check: assert np.array_equal(check['prefix'], array)
            shards.append({'file': filename, 'start': begin, 'end': end})
            artifacts[filename] = {'sha256': digest(path), 'bytes': path.stat().st_size}
            byte_count += array.nbytes
            save(out/'progress.json', {'rows': end, 'total': len(train), 'seconds': time.monotonic()-start})
            if begin % 2560 == 0: print(json.dumps({'view': args.view, 'rows': end, 'total': len(train)}), flush=True)
        np.savez_compressed(out/'index.npz', **{k: rows[k][train] for k in ('ids','parent_ids','labels')})
        artifacts['index.npz'] = {'sha256': digest(out/'index.npz'), 'bytes': (out/'index.npz').stat().st_size}
        save(out/'complete.json', {'complete': True, 'schema': 'g40-frozen-prefix-train-only-v1',
            'view': args.view, 'rows': len(train), 'shape_per_row': list(array.shape[1:]), 'dtype': 'float32',
            'uncompressed_bytes': byte_count, 'source_model_sha256': sources[view], 'source_receipt_sha256': input_receipt,
            'source_training_features_sha256': digest(refpath), 'classification_feature_max_abs_error': max_error,
            'feature_tolerance': {'atol': 2e-4, 'rtol': 2e-4}, 'all_training_rows_verified': True,
            'validation_forwarded': False, 'prefix_frozen': True, 'batchnorm_running_stats_frozen': True,
            'shards': shards, 'artifacts': artifacts, 'seconds': time.monotonic()-start,
            'lossless_archive_roundtrip': True})
    except BaseException as error:
        save(out/'failed.json', {'error': repr(error), 'seconds': time.monotonic()-start}); raise


def load_prefix(folder, rows, device='cuda'):
    folder = Path(folder); receipt = json.loads((folder/'complete.json').read_text())
    assert receipt['complete'] and receipt['schema'] == 'g40-frozen-prefix-train-only-v1'
    assert not receipt['validation_forwarded'] and not (folder/'failed.json').exists()
    for name, spec in receipt['artifacts'].items(): assert digest(folder/name) == spec['sha256']
    with np.load(folder/'index.npz', allow_pickle=False) as f:
        for k in ('ids','parent_ids','labels'): assert np.array_equal(f[k], rows[k]), k
    values = np.empty((receipt['rows'], *receipt['shape_per_row']), np.float32); end = 0
    for shard in receipt['shards']:
        assert shard['start'] == end
        with np.load(folder/shard['file'], allow_pickle=False) as f: values[end:shard['end']] = f['prefix']
        end = shard['end']
    assert end == len(rows['ids']) and np.isfinite(values).all()
    return torch.as_tensor(values, device=device), receipt


def class_centers(teachers, rows, device):
    centers = {}
    for view in ('spatial','speed'):
        folder = Path(teachers)/(view+'-42'); receipt = json.loads((folder/'complete.json').read_text())
        path = folder/'train-predictions.npz'; assert digest(path) == receipt['artifacts'][path.name]['sha256']
        with np.load(path, allow_pickle=False) as f:
            lookup = {str(s): i for i,s in enumerate(f['ids'])}; order = [lookup[str(s)] for s in rows['ids']]
            assert np.array_equal(f['labels'][order], rows['labels'])
            assert np.array_equal(f['parent_ids'][order], rows['parent_ids'])
            features = f['features'][order]
        centers[view] = torch.as_tensor(np.stack([features[rows['labels']==c].mean(0) for c in range(5)]), device=device)
    return centers


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--view', choices=VIEWS, required=True)
    for k in ('cache','raw-cache','teachers','raw-teachers','output'): p.add_argument('--'+k, type=Path, required=True)
    prepare(p.parse_args())
