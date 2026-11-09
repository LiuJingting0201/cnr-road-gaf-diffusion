"""Describe class redundancy in the exact G39 training targets, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(teachers, output):
    assert not output.exists()
    result = {'complete': True, 'views': {},
        'scope': 'Existing G39 train-only targets; descriptive variance decomposition. No training, model selection, or development targets.',
        'normalization': 'Exact G39 row-weighted global mean subtraction followed by unit length. Report both row and class-equal descriptive weighting.',
        'limits': 'Between-class variance is redundancy with known labels, but remaining variance is not automatically useful movement information. No causal failure attribution.'}
    for view in ('spatial', 'speed'):
        folder = teachers / (view + '-42')
        path = folder / 'train-predictions.npz'
        receipt = json.loads((folder / 'complete.json').read_text())
        source_hash = sha(path)
        assert receipt['complete'] and source_hash == receipt['artifacts'][path.name]['sha256']
        with np.load(path, allow_pickle=False) as archive:
            raw = archive['features'].astype(np.float32); labels = archive['labels']
            assert len(archive['ids']) == len(set(archive['ids'])) == len(labels)
        assert set(labels) == set(range(5)) and raw.shape == (len(labels), 512)
        unit = raw - raw.mean(0); unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        means = np.stack([unit[labels == c].mean(0) for c in range(5)])
        centers = means / np.linalg.norm(means, axis=1, keepdims=True)
        cosine = 1 - (unit * centers[labels]).sum(1)
        by_class = {str(c): {'n': int((labels == c).sum()),
            'class_centroid_cosine_distance_mean': float(cosine[labels == c].mean()),
            'within_class_squared_distance_to_mean': float(np.square(unit[labels == c] - means[c]).sum(1).mean())}
            for c in range(5)}
        weighted = {}; counts = np.bincount(labels, minlength=5)
        for name, weights in [('row', np.ones(len(labels))/len(labels)),
                              ('class_equal', 1/(5*counts[labels]))]:
            overall = np.sum(unit * weights[:, None], axis=0)
            total = float(np.sum(np.square(unit - overall).sum(1) * weights))
            residual = float(np.sum(np.square(unit - means[labels]).sum(1) * weights))
            weighted[name] = {'class_mean_explained_fraction': 1-residual/total,
                'total_variance': total, 'within_class_variance': residual,
                'class_centroid_cosine_distance': float(np.sum(cosine * weights))}
        result['views'][view] = {'source_sha256': source_hash, 'rows': len(labels),
            'weighting': weighted, 'by_class': by_class}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({v: r['weighting'] for v, r in result['views'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.teachers, args.output)
