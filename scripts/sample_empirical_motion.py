"""E3b: transfer complete class-matched training motion to the frozen E4r paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from cnr.path_diffusion import decode

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
a = p.parse_args()
torch.set_num_threads(4)
beg = time.monotonic()
reference = Path(f'runs/route-diverse-e4r/seed{a.seed}')
cfg = json.loads((reference / 'config.json').read_text())
data = Path(cfg['data'])
training = np.load(data / 'train.npz')
source = Path(cfg['dataset']['source_directory'])
raw_train = np.load(source / 'train.npz')
raw_val = np.load(source / 'val.npz')
train_ids = raw_train['ids'][training['source_index']]
assert not set(train_ids.tolist()) & set(raw_val['ids'].tolist())
assert np.array_equal(training['labels'], raw_train['labels'][training['source_index']])
pool = np.load(reference / 'candidates.npz')
labels = pool['labels']
starts = pool['xy'][:, 0].copy()  # Only the given start, never the validation future.
routes = pool['route']
lengths = pool['route_length']
assert training['progress'].shape[1] == 512
assert all(np.sum(pool['condition_id'] == c) == 8 for c in np.unique(pool['condition_id']))
rng = np.random.default_rng(a.seed)
parents = np.array([rng.choice(np.flatnonzero(training['labels'] == label)) for label in labels])
increments = np.maximum(0, np.diff(training['progress'][parents], axis=1))
loginc = np.log1p(increments)
mu, sd = cfg['log_increment_mean'], cfg['log_increment_std']
latent = torch.as_tensor((loginc - mu) / sd, dtype=torch.float32)[:, None]
with torch.no_grad():
    xy, progress = decode(latent, mu, sd, torch.from_numpy(routes), torch.from_numpy(lengths), torch.from_numpy(starts))
xy, progress = xy.numpy(), progress.numpy()
assert np.isfinite(xy).all() and np.allclose(xy[:, 0], starts, atol=1e-4)
assert np.array_equal(training['labels'][parents], labels)
# Verify the shared decoder preserved the selected motion except its declared log cap.
expected = np.cumsum(np.expm1(np.clip(loginc, 0, 5)), axis=1)
assert np.allclose(progress[:, 1:], expected, atol=.02, rtol=1e-5)
out = Path(f'runs/empirical-motion-e3b/seed{a.seed}')
out.mkdir(parents=True, exist_ok=False)
np.savez_compressed(out / 'candidates.npz', xy=xy, progress=progress, labels=labels,
    source_index=pool['source_index'], condition_id=pool['condition_id'], route=routes,
    route_length=lengths, parent_training_index=parents)
newcfg = cfg | {'steps': 0, 'pretrained_steps': 0, 'generator': 'class_matched_empirical_motion',
    'checkpoint': None, 'checkpoint_sha256': None, 'noise_seed': None,
    'resampling_seed': a.seed, 'training_parent_indices': parents.tolist(),
    'training_parent_source_indices': training['source_index'][parents].tolist(),
    'training_parent_ids': train_ids[parents].tolist(),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'reference_pool_sha256': hashlib.sha256((reference / 'candidates.npz').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((data / 'train.npz').read_bytes()).hexdigest(),
    'route_array_sha256': hashlib.sha256(routes.tobytes()).hexdigest(),
    'scope': 'No learned generator. Full training motion sampled by class with replacement, same E4r paths/starts/K. Three resampling seeds are not three model fits. No real validation future used.'}
(out / 'config.json').write_text(json.dumps(newcfg, indent=2))
result = {'complete': True, 'samples': len(xy), 'points': 512,
    'unique_training_parents': len(np.unique(parents)),
    'log_increment_clipped_fraction': float(np.mean(loginc > 5)),
    'route_end_capped_fraction': float(np.mean(progress[:, -1] >= lengths)),
    'elapsed_s': time.monotonic() - beg}
(out / 'complete.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
