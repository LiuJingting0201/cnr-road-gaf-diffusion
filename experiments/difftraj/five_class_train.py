"""Stabilized DiffTraj training on completed SUMO trips. See the experiment protocol."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from utils.Traj_UNet import Guide_UNet
from utils.config import args as upstream_config

CLASSES = ('bicycle', 'bus', 'motorcycle', 'passenger', 'taxi')


class VehicleDiffTraj(Guide_UNet):
    def __init__(self, arm):
        config = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in upstream_config.items()})
        super().__init__(config)
        if arm in ('C', 'D'):
            # Construct from zeros directly so allocating this arm consumes no extra RNG.
            self.class_embedding = torch.nn.Embedding.from_pretrained(
                torch.zeros(len(CLASSES), self.ch), freeze=False)
        elif arm not in ('A', 'B'):
            raise ValueError(arm)

    def forward(self, x, t, head, y):
        if not hasattr(self, 'class_embedding'):
            return super().forward(x, t, head)
        cond = self.guide_emb(head) + self.class_embedding(y)
        placeholder = self.place_emb(torch.zeros_like(head))
        guided = self.unet(x, t, cond)
        unguided = self.unet(x, t, placeholder)
        return guided + self.guidance_scale * (guided - unguided)


def make_model(arm, device='cuda'):
    return VehicleDiffTraj(arm).to(device)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def save_checkpoint(path, state):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def train_rows(path, mode):
    with np.load(path / 'index.npz') as index:
        labels = index['classes']
    encoded = np.searchsorted(CLASSES, labels)
    if (encoded >= len(CLASSES)).any() or not np.array_equal(np.asarray(CLASSES)[encoded], labels):
        raise ValueError(f'Unexpected class in {path}')
    ids = np.arange(len(labels)) if mode == 'all' else np.flatnonzero(labels == mode)
    if not len(ids):
        raise ValueError(f'No {mode} examples in {path}')
    return ids, encoded


def load_split(opts, split, device):
    folder = opts.data / split
    ids, labels = train_rows(folder, opts.mode)
    if split == 'validation':
        ids = np.random.default_rng(142).permutation(ids)[:512]
    def load(path):
        return torch.from_numpy(np.load(path, mmap_mode='r')[ids]).to(device)
    x = load(folder / 'traj.npy').transpose(1, 2).contiguous()
    h = load(folder / 'head.npy')
    y = torch.from_numpy(labels[ids]).to(device)
    speed = dt = None
    if opts.arm in ('B', 'D'):
        folder = opts.speed_dir / split
        if not np.array_equal(np.load(folder / 'y.npy', mmap_mode='r')[ids], labels[ids]):
            raise ValueError(f'Target labels disagree with release: {split}')
        speed, dt = load(folder / 'v.npy'), load(folder / 'dt.npy')
        if speed.shape != (len(ids), 199) or dt.shape != (len(ids),):
            raise ValueError(f'Bad motion target shape: {split}')
        if not torch.isfinite(speed).all() or (speed < 0).any() or not torch.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError(f'Invalid native-speed or interval target: {split}')
    return x, h, y, speed, dt, ids


def speed_supervision(z, eps, alpha_bar, sd, target, dt):
    clean = (z - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
    estimate = torch.linalg.vector_norm(clean.diff(dim=2) * sd, dim=1) / dt[:, None]
    residual = F.smooth_l1_loss(torch.log1p(estimate), torch.log1p(target), reduction='none')
    return (0.05 * alpha_bar[:, 0] * residual).mean()


class ValidationNoise:
    def __init__(self, data, alpha, device):
        x, self.h, self.y, _, _, self.ids = data
        self.t = torch.arange(len(x), device=device) * 500 // len(x)
        gen = torch.Generator(device=device).manual_seed(142)
        self.noise = torch.randn(x.shape, device=device, generator=gen)
        a = alpha[self.t, None, None]
        self.z = a.sqrt() * x + (1-a).sqrt() * self.noise

    @torch.no_grad()
    def measure(self, net):
        net.eval()
        total = torch.zeros((), device=self.z.device)
        for start in range(0, len(self.z), 64):
            rows = slice(start, start + 64)
            prediction = net(self.z[rows], self.t[rows], self.h[rows], self.y[rows])
            total += (prediction-self.noise[rows]).square().mean((1, 2)).sum()
        net.train()
        result = float(total / len(self.z))
        if not math.isfinite(result):
            raise FloatingPointError('Nonfinite validation noise MSE')
        return result


def train(opts):
    if (opts.mode == 'all') != (opts.arm in ('C', 'D')):
        raise ValueError('A/B use a single vehicle class. Pooled C/D explicitly condition on class.')
    if min(opts.epochs, opts.batch_size, opts.microbatch) <= 0:
        raise ValueError('Positive epoch and batch budgets required')
    device = torch.device(opts.device)
    torch.manual_seed(opts.seed)
    net = make_model(opts.arm, device)
    optim = torch.optim.AdamW(net.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opts.epochs, eta_min=2e-5)
    alpha = (1 - torch.linspace(.0001, .05, 500, device=device)).cumprod(0)
    x, h, y, v, dt, ids = load_split(opts, 'train', device)
    validation = ValidationNoise(load_split(opts, 'validation', device), alpha, device)
    with np.load(opts.data / 'normalization.npz') as norm:
        sd = torch.as_tensor(norm['std'], dtype=x.dtype, device=device).reshape(1, 2, 1)
    updates = math.ceil(len(x)/opts.batch_size)
    spec = dict(arm=opts.arm, mode=opts.mode, seed=opts.seed, classes=list(CLASSES), epochs=opts.epochs,
                batch_size=opts.batch_size, microbatch=opts.microbatch, train_n=len(x),
                updates_per_epoch=updates, lr_start=2e-4, lr_floor=2e-5, gradient_norm_clip=1.,
                noise_levels=500, speed_loss_weight=.05 if v is not None else 0.,
                train_index_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                validation_index_sha256=hashlib.sha256(validation.ids.astype('<i8').tobytes()).hexdigest(),
                release_manifest_sha256=hashlib.sha256((opts.data/'manifest.json').read_bytes()).hexdigest())
    out = opts.out
    specfile = out/'config.json'
    if specfile.exists() and json.loads(specfile.read_text()) != spec:
        raise ValueError(f'Existing training configuration differs: {out}')
    atomic_json(specfile, spec)
    done = out/'TRAIN_COMPLETE'
    if done.is_file():
        if not (out/'final.pt').is_file() or not (out/'best.pt').is_file():
            raise RuntimeError('Training marker without both retained checkpoints')
        return
    last = out/'last.pt'
    log = out/'epochs.jsonl'
    history = []
    best = float('inf')
    torch.manual_seed(opts.seed)
    if last.is_file():
        state = torch.load(last, weights_only=True, map_location='cpu')
        if state['spec'] != spec:
            raise ValueError('Resume checkpoint specification mismatch')
        net.load_state_dict(state['weights'])
        optim.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['device_rng'], device)
        torch.set_rng_state(state['cpu_rng'])
        best, history = state['best'], state['history']
        log.write_text(''.join(json.dumps(r)+'\n' for r in history))
        del state
    elapsed_before = history[-1]['elapsed_s'] if history else 0.
    started = time.monotonic()
    def elapsed():
        return elapsed_before + time.monotonic() - started
    def snapshot(epoch, val):
        return dict(weights=net.state_dict(), arm=opts.arm, mode=opts.mode, seed=opts.seed,
                    classes=list(CLASSES), epoch=epoch, validation_noise_mse=val)

    for epoch in range(len(history)+1, opts.epochs+1):
        gen = torch.Generator().manual_seed(opts.seed + epoch)
        order = torch.randperm(len(x), generator=gen).to(device)
        totals = torch.zeros(2, device=device)
        norms = []
        lr = optim.param_groups[0]['lr']
        for step, group in enumerate(order.split(opts.batch_size), start=1):
            optim.zero_grad(set_to_none=True)
            group_loss = torch.zeros_like(totals)
            for idx in group.split(opts.microbatch):
                t = torch.randint(500, (len(idx)//2+1,), device=device)
                t = torch.cat((t, 499-t))[:len(idx)]
                a = alpha[t, None, None]
                eps = torch.randn_like(x[idx])
                z = a.sqrt()*x[idx] + (1-a).sqrt()*eps
                pred = net(z, t, h[idx], y[idx])
                noise_loss = F.mse_loss(pred, eps)
                aux = (speed_supervision(z, pred, a, sd, v[idx], dt[idx])
                       if v is not None else noise_loss.new_zeros(()))
                ((noise_loss + aux) * (len(idx)/len(group))).backward()
                group_loss += len(idx)*torch.stack((noise_loss.detach(), aux.detach()))
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            norms.append(norm.detach())
            optim.step()
            totals += group_loss
            update = (epoch-1)*updates + step
            if update == 1 or update % 25 == 0:
                noise, speed = (group_loss/len(group)).tolist()
                atomic_json(out/'progress.json', dict(arm=opts.arm, mode=opts.mode, seed=opts.seed,
                            epoch=epoch, step=step, updates=update, target_updates=updates*opts.epochs,
                            elapsed_s=elapsed(), noise_loss=noise, weighted_speed_loss=speed, lr=lr))
        scheduler.step()
        val = validation.measure(net)
        noise, aux = (totals/len(x)).tolist()
        norm = torch.stack(norms)
        weights = snapshot(epoch, val)
        if val < best:
            best = val
            save_checkpoint(out/'best.pt', weights)
        record = dict(epoch=epoch, updates=epoch*updates, noise_loss=noise, weighted_speed_loss=aux,
                      val_loss=val, best_val_loss=best, lr=lr, grad_norm_max=norm.max().item(),
                      clipped_steps=(norm>1.).sum().item(), elapsed_s=elapsed())
        history.append(record)
        save_checkpoint(last, dict(**weights, spec=spec, optimizer=optim.state_dict(),
                                   scheduler=scheduler.state_dict(), history=history, best=best,
                                   cpu_rng=torch.get_rng_state(),
                                   device_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
        with log.open('a') as output:
            print(json.dumps(record, allow_nan=False), file=output)
        print(json.dumps(record, allow_nan=False), flush=True)
    final = snapshot(opts.epochs, history[-1]['val_loss'])
    save_checkpoint(out/'final.pt', final)
    atomic_json(done, dict(epoch=opts.epochs, updates=updates*opts.epochs, best_val_loss=best,
                           best_epoch=min(history, key=lambda r:r['val_loss'])['epoch']))
    last.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--speed-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices='ABCD', required=True)
    parser.add_argument('--mode', choices=('all', *CLASSES), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--microbatch', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    opts = parser.parse_args()
    opts.out.mkdir(parents=True, exist_ok=True)
    with (opts.out/'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(opts)


if __name__ == '__main__':
    main()
