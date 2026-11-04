"""Evaluate the 36 frozen E5 learners, without any training or selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cnr.final_protocol import GROUPS, verify_lock, sha256
from cnr.representations import SequenceClassifier
from train_augmentation_downstream import GRUClassifier

p = argparse.ArgumentParser(); p.add_argument('--lock', required=True)
p.add_argument('--reference', default='data/processed/final-test-e7/reference'); a = p.parse_args()
verify_lock(a.lock); torch.set_num_threads(4)
data = np.load(Path(a.reference) / 'test.npz'); y = data['labels']; ids = data['ids']
meta = json.loads(Path('data/processed/augmentation-e5/dataset.json').read_text()); stats = meta['stats']
out = Path('docs/results/final-downstream-e7'); out.mkdir(parents=True, exist_ok=False)
predroot = Path('runs/final-downstream-e7'); predroot.mkdir(parents=True, exist_ok=False)
x = torch.from_numpy(data['xy']).cuda(); delta = torch.diff(x, dim=1, prepend=x[:, :1])
def normalize(values, name):
    return (values - torch.tensor(stats[name+'_mean'], device='cuda')) / torch.tensor(stats[name+'_std'], device='cuda')
features = torch.cat((normalize(x, 'xy'), normalize(delta, 'delta')), -1).transpose(1, 2).contiguous()
def metrics(pred, labels):
    cm = np.bincount(labels * 6 + pred, minlength=36).reshape(6, 6)
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return {'accuracy': float(np.mean(pred == labels)), 'macro_f1': float(f1.mean()), 'confusion': cm.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_recall': [float(cm[c,c] / cm[c].sum()) if cm[c].sum() else None for c in range(6)]}
rng = np.random.default_rng(20261113)
bootstrap = [np.concatenate([rng.choice(np.flatnonzero(y == c), np.sum(y == c), replace=True)
                            for c in np.unique(y)]) for _ in range(2000)]
result = {'class_counts': np.bincount(y, minlength=6).tolist(), 'vehicles': len(y), 'models': {},
          'scope': 'Locked final test, fixed 36 learners; 2000 paired vehicle resamples, no retraining; fixed six-class macro-F1 (undefined-class terms are zero).'}
for arch in ['tcn', 'gru']:
    rows = {}; boots = {}
    for group in GROUPS:
        predictions = []; individual = []
        for seed in [100, 101, 102]:
            checkpoint = Path(f'runs/augmentation-e5/{arch}-{group}-seed{seed}/final.pt')
            model = (SequenceClassifier(4) if arch == 'tcn' else GRUClassifier()).cuda().eval()
            model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
            with torch.no_grad():
                probs = torch.cat([model(features[i:i+64]).softmax(-1) for i in range(0, len(y), 64)]).cpu().numpy()
            pred = probs.argmax(-1); predictions.append(pred)
            record = metrics(pred, y) | {'seed': seed, 'checkpoint_sha256': sha256(checkpoint)}
            individual.append(record)
            np.savez_compressed(predroot / f'{arch}-{group}-seed{seed}.npz', probabilities=probs, labels=y, ids=ids)
            del model
        boots[group] = np.array([np.mean([metrics(pred[ix], y[ix])['macro_f1'] for pred in predictions]) for ix in bootstrap])
        rows[group] = {'macro_f1_mean': float(np.mean([r['macro_f1'] for r in individual])),
                       'accuracy_mean': float(np.mean([r['accuracy'] for r in individual])), 'individual': individual}
        print(json.dumps({'arch': arch, 'group': group, 'macro_f1': rows[group]['macro_f1_mean']}), flush=True)
    for group in GROUPS:
        rows[group]['paired_delta_f1'] = {base: {'point': rows[group]['macro_f1_mean'] - rows[base]['macro_f1_mean'],
            'percentile95': np.quantile(boots[group] - boots[base], [.025, .975]).tolist()} for base in GROUPS}
    result['models'][arch] = rows
(out / 'summary.json').write_text(json.dumps(result, indent=2))
(predroot / 'complete.json').write_text(json.dumps({'complete': True, 'lock_sha256': sha256(a.lock)}))
