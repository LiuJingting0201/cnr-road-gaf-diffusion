"""Final synthesis loads no real reference or test future, only checked conditions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from cnr.final_protocol import verify_lock, validate_conditions, sha256
from cnr.path_diffusion import PathDenoiser, decode

p = argparse.ArgumentParser(); p.add_argument('--lock', required=True)
p.add_argument('--conditions', default='data/processed/final-test-e7/planned/conditions.npz')
p.add_argument('--family', choices=['diffusion', 'empirical'], required=True)
p.add_argument('--seed', type=int, choices=[42, 43, 44], required=True); a = p.parse_args()
lock = verify_lock(a.lock); torch.set_num_threads(4)
c = dict(np.load(a.conditions)); validate_conditions(c)
out = Path(f'runs/final-{a.family}-e7/seed{a.seed}'); out.mkdir(parents=True, exist_ok=False)
try:
    old = Path(f'runs/long-path-e3/seed{a.seed}')
    cfg = json.loads((old / 'config.json').read_text()); mu, sd = cfg['log_increment_mean'], cfg['log_increment_std']
    device = 'cuda' if a.family == 'diffusion' else 'cpu'
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    parents = None; latent = None; model = None; beg = time.monotonic()
    if a.family == 'empirical':
        training = np.load('data/processed/long-path-e3-512/train.npz')
        rng = np.random.default_rng(a.seed)
        parents = np.array([rng.choice(np.flatnonzero(training['labels'] == label)) for label in c['labels']])
        latent = to((np.log1p(np.maximum(0, np.diff(training['progress'][parents], axis=1))) - mu) / sd)[:, None]
    else:
        model = PathDenoiser().to(device).eval()
        model.load_state_dict(torch.load(old / 'latest.pt', map_location=device, weights_only=True)['ema'])
        alpha = torch.cumprod(1 - torch.linspace(1e-4, .02, 1000, device=device), 0)
        times = torch.linspace(999, 0, 100).long().tolist()
        mean, std = to(cfg['dataset']['mean']), to(cfg['dataset']['std'])
    torch.manual_seed(20261110); xys, progresses = [], []
    with torch.no_grad():
        for i in range(0, len(c['labels']), 32):
            route, length, start = [to(c[k][i:i+32]) for k in ['route', 'route_length', 'starts']]
            label = torch.as_tensor(c['labels'][i:i+32], device=device)
            if a.family == 'empirical': sample = latent[i:i+32]
            else:
                route_features = ((route[:, ::8] - start[:, None]) / 128).transpose(1, 2)
                sample = torch.randn(len(label), 1, 511, device=device)
                for j, t in enumerate(times):
                    eps = model(sample, torch.full((len(label),), t, device=device), label, (start - mean) / std, route_features)
                    clean = (sample - (1-alpha[t]).sqrt() * eps) / alpha[t].sqrt()
                    prev = alpha[times[j+1]] if j+1 < len(times) else 1.
                    sample = prev ** .5 * clean + (1-prev) ** .5 * eps
            xy, progress = decode(sample, mu, sd, route, length, start)
            assert torch.isfinite(xy).all() and torch.allclose(xy[:, 0], start, atol=1e-4)
            xys.append(xy.cpu().numpy()); progresses.append(progress.cpu().numpy())
    xy, progress = np.concatenate(xys), np.concatenate(progresses)
    np.savez_compressed(out / 'candidates.npz', xy=xy, progress=progress,
                        **{key: value for key, value in c.items() if key != 'starts'})
    config = cfg | {'final_test': True, 'locked_commit': lock['git_commit'], 'lock_sha256': sha256(a.lock),
                    'family': a.family, 'seed': a.seed, 'steps': 0, 'noise_seed': 20261110,
                    'conditions_sha256': sha256(a.conditions), 'scope': 'Locked held-out-ID synthesis; no test future loaded',
                    'parent_training_indices': parents.tolist() if parents is not None else None}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    report = {'complete': True, 'samples': len(xy), 'route_end_capped_fraction': float(np.mean(progress[:, -1] >= c['route_length'])),
              'elapsed_s': time.monotonic() - beg}
    (out / 'complete.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)
except Exception as ex:
    (out / 'failed.json').write_text(json.dumps({'error': str(ex), 'time': time.time()})); raise
