"""Evaluate frozen participant networks on the full six-class test cohort."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from cnr.final_protocol import verify_lock, sha256
from cnr.representations import RepresentationClassifier

p = argparse.ArgumentParser(); p.add_argument('--lock', required=True); a = p.parse_args()
verify_lock(a.lock); torch.set_num_threads(4)
data = np.load('data/processed/final-test-e7/reference/test.npz'); y = data['labels']
out = Path('docs/results/final-auxiliary-e7'); out.mkdir(parents=True, exist_ok=False)
predroot = Path('runs/final-auxiliary-e7'); predroot.mkdir(parents=True, exist_ok=False)
mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
result = {'vehicles': len(y), 'class_counts': np.bincount(y, minlength=6).tolist(), 'models': {},
          'scope': 'Self-trained participant networks, all encodable six-class held-out windows; final fixed checkpoints, no tuning.'}
for family in ['gaf', 'tcn']:
    rows = []
    for seed in [42, 43, 44]:
        if family == 'gaf':
            path = Path(f'runs/gaf-duration-e2/length512-seed{seed}/final.pt')
            model = resnet18(weights=None); model.fc = torch.nn.Linear(model.fc.in_features, 6)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            values = data['images']
        else:
            directory = Path(f'runs/long-selection-e4/tcn-seed{seed}')
            path = directory / 'latest.pt'; cfg = json.loads((directory / 'config.json').read_text())
            model = RepresentationClassifier('xy_motion_tcn', cfg['stats'])
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
            values = data['xy']
        model = model.cuda().eval(); probabilities = []
        with torch.no_grad():
            for i in range(0, len(y), 32):
                x = torch.from_numpy(values[i:i+32]).cuda().float()
                if family == 'gaf': x = (x.permute(0, 3, 1, 2) / 255 - mean) / std
                probabilities.append(model(x).softmax(-1).cpu().numpy())
        prob = np.concatenate(probabilities); pred = prob.argmax(-1)
        cm = np.bincount(y*6+pred, minlength=36).reshape(6, 6)
        f1 = 2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1), 1)
        rows.append({'seed': seed, 'accuracy': float(np.mean(pred == y)), 'macro_f1': float(f1.mean()),
                     'confusion': cm.tolist(), 'per_class_f1': f1.tolist(),
                     'per_class_recall': [float(cm[c,c]/cm[c].sum()) if cm[c].sum() else None for c in range(6)],
                     'checkpoint_sha256': sha256(path)})
        np.savez_compressed(predroot / f'{family}-seed{seed}.npz', probabilities=prob, labels=y, ids=data['ids'])
        del model
    result['models'][family] = {'individual': rows, 'macro_f1_mean': float(np.mean([r['macro_f1'] for r in rows])),
                               'accuracy_mean': float(np.mean([r['accuracy'] for r in rows]))}
(out / 'summary.json').write_text(json.dumps(result, indent=2))
(predroot / 'complete.json').write_text(json.dumps({'complete': True}))
