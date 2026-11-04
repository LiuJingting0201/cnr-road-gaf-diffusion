"""E1: new participant CNN on inherited Hilbert RGB, not paper reproduction."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--initialization', choices=['random', 'imagenet'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(a.data)
    tr, va = [np.load(root / f'{s}.npz', allow_pickle=False) for s in ['train', 'val']]
    assert set(tr['ids']).isdisjoint(set(va['ids']))
    assert tr['images'].shape[1:] == va['images'].shape[1:] == (224, 224, 3)
    assert np.unique(tr['labels']).tolist() == list(range(6))
    weights = ResNet18_Weights.IMAGENET1K_V1 if a.initialization == 'imagenet' else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model = model.cuda()
    tx, vx = [torch.from_numpy(z['images']).cuda() for z in [tr, va]]
    ty, vy = [torch.from_numpy(z['labels']).cuda() for z in [tr, va]]
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]

    def features(x):
        # Keep the entire encoded time-time plane: no spatial crop/flip/jitter.
        return (x.permute(0, 3, 1, 2).float() / 255 - mean) / std

    cw = torch.bincount(ty, minlength=6).float().rsqrt()
    cw /= cw.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    config = vars(a) | {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'dataset': json.loads((root / 'dataset.json').read_text()),
        'data_hashes': {s: hashlib.sha256((root / f'{s}.npz').read_bytes()).hexdigest() for s in ['train', 'val']},
        'weights': str(weights), 'class_weights': cw.cpu().tolist(),
        'optimizer': {'name': 'AdamW', 'lr': 2e-4, 'weight_decay': 1e-4},
        'scope': 'Self-trained auxiliary CNN. Fixed final epoch primary; val-best secondary. No test access.',
        'preprocess': 'RGB /255, ImageNet mean/std; no resize, crop, flip or color jitter',
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2))

    @torch.no_grad()
    def evaluate():
        model.eval()
        logits = torch.cat([model(features(vx[i:i+64])) for i in range(0, len(vx), 64)])
        probs = logits.softmax(-1)
        pred = probs.argmax(-1)
        cm = torch.bincount(vy * 6 + pred, minlength=36).reshape(6, 6).float()
        f1 = 2 * cm.diag() / (cm.sum(0) + cm.sum(1)).clamp_min(1)
        confidence = probs.max(-1).values
        correct = (pred == vy).float()
        ece = 0.
        for i in range(10):
            mask = (confidence >= i / 10) & (confidence < (i+1) / 10 if i < 9 else confidence <= 1)
            if mask.any():
                ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
        return {
            'macro_f1': float(f1.mean()), 'accuracy': float(correct.mean()),
            'per_class_f1': f1.tolist(), 'per_class_recall': (cm.diag()/cm.sum(1).clamp_min(1)).tolist(),
            'confusion': cm.int().tolist(), 'ece_10bins': ece,
            'nll': float(torch.nn.functional.cross_entropy(logits, vy)),
        }, probs.cpu().numpy()

    start = time.monotonic()
    gen = torch.Generator(device='cuda').manual_seed(a.seed + 1000)
    best = -1.
    with (out / 'metrics.jsonl').open('x', buffering=1) as f:
        for epoch in range(1, a.epochs+1):
            model.train()
            order = torch.randperm(len(tx), device='cuda', generator=gen)
            loss_sum = 0.
            for ix in order.split(a.batch_size):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(features(tx[ix])), ty[ix], weight=cw)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                loss_sum += float(loss.detach()) * len(ix)
            report, probs = evaluate()
            record = {'epoch': epoch, 'train_weighted_ce': loss_sum/len(tx), 'elapsed_s': time.monotonic()-start, **report}
            f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            if report['macro_f1'] > best:
                best = report['macro_f1']
                torch.save(model.state_dict(), out / 'best.pt')
                (out / 'best_validation.json').write_text(json.dumps(record, indent=2))
            if epoch % 10 == 0 or epoch == a.epochs:
                torch.save(model.state_dict(), out / 'latest.pt')
        torch.save(model.state_dict(), out / 'final.pt')
        np.savez_compressed(out / 'final_validation.npz', probabilities=probs, labels=va['labels'], ids=va['ids'])
    (out / 'complete.json').write_text(json.dumps({'complete': True, 'epochs': a.epochs, 'final_validation': report, 'best_validation_macro_f1': best, 'elapsed_s': time.monotonic()-start}, indent=2))


if __name__ == '__main__':
    main()
