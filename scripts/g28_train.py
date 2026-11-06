"""Align a route-conditioned denoiser to paired frozen representation teachers."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cnr.path_diffusion import PathDenoiser
from g22_diffusion import STEPS, inputs, load, physical_stats, targets

ARMS = ("B", "P", "S", "V", "SV", "SH")


class Denoiser(PathDenoiser):
    def __init__(self):
        super().__init__(classes=8)
        # Heads exist in every arm. Their construction leaves the base RNG intact.
        with torch.random.fork_rng(devices=[]):
            self.projections = nn.ModuleDict({
                view: nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 512))
                for view in ("spatial", "speed")
            })
        self.hidden = None
        self.backbone.blocks[4].register_forward_hook(self._capture)

    def _capture(self, module, arguments, output):
        self.hidden = output.mean(-1)

    def projected(self, view):
        return F.normalize(self.projections[view](self.hidden), dim=1)


def latent_targets(trips, teachers, output):
    vectors, centroids, means = {}, {}, {}
    for view in ("spatial", "speed"):
        directory = teachers / f"{view}-42"
        if not (directory / "complete.json").exists():
            raise FileNotFoundError(f"Teacher incomplete: {directory}")
        with np.load(directory / "train.npz") as feature:
            lookup = {ident: i for i, ident in enumerate(feature["ids"])}
            index = np.asarray([lookup[ident] for ident in trips["ids"]])
            if not np.array_equal(trips["labels"], feature["labels"][index]):
                raise ValueError(f"Teacher label mismatch for {view}")
            raw = feature["features"][index].copy()
        mean = raw.mean(0)
        centered = raw - mean
        unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
        centers = np.stack([unit[trips["labels"] == c].mean(0) for c in range(8)])
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        vectors[view] = torch.as_tensor(unit, device="cuda")
        centroids[view] = torch.as_tensor(centers, device="cuda")
        means[view] = mean
    rng = np.random.default_rng(28042)
    permutation = np.arange(len(trips["ids"]))
    groups = []
    for label in range(8):
        eligible = np.flatnonzero(trips["labels"] == label)
        thresholds = np.quantile(trips["length"][eligible], [.25, .5, .75])
        bins = np.searchsorted(thresholds, trips["length"][eligible])
        for band in range(4):
            members = eligible[bins == band]
            order = rng.permutation(members)
            permutation[order] = np.roll(order, 1)
            groups.append({"class": label, "length_quartile": band, "count": len(order)})
    np.savez(output / "normalizer.npz", **means)
    np.savez(output / "shuffle.npz", ids=trips["ids"], paired_ids=trips["ids"][permutation])
    metadata = {"normalization": "training global mean then row unit norm, no variance scaling",
                "teacher_seed": 42, "shuffle": "fixed cyclic derangement within class and class-specific length quartile",
                "shuffle_self_matches": int(np.sum(permutation == np.arange(len(permutation)))),
                "shuffle_groups": groups,
                "train_ids_sha256": hashlib.sha256("\n".join(trips["ids"]).encode()).hexdigest()}
    (output / "targets.json").write_text(json.dumps(metadata, indent=2))
    return vectors, centroids, torch.as_tensor(permutation, device="cuda")


def alignment(model, arm, indices, labels, vectors, centroids, permutation, alpha):
    views = {"P": ("spatial", "speed"), "S": ("spatial",), "V": ("speed",),
             "SV": ("spatial", "speed"), "SH": ("spatial", "speed")}[arm]
    losses = []
    for view in views:
        selected = permutation[indices] if arm == "SH" else indices
        target = centroids[view][labels] if arm == "P" else vectors[view][selected]
        cosine_loss = 1 - (model.projected(view) * target).sum(1)
        losses.append((alpha * cosine_loss).mean())
    return torch.stack(losses).mean()


def save_atomic(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def fit(args):
    torch.set_num_threads(2)
    output = args.output / f"{args.arm}-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        return
    trips = load("train")
    statistics = physical_stats(trips)
    vectors, centroids, permutation = latent_targets(trips, args.teachers, output)
    (output / "stats.json").write_text(json.dumps(statistics, indent=2))
    torch.manual_seed(args.seed)
    model = Denoiser().cuda()
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    raw = torch.from_numpy(targets(trips, statistics)).cuda()[:, None]
    labels, starts, paths = inputs(trips, statistics)
    alpha_bars = torch.cumprod(1 - torch.linspace(.0001, .02, 1000, device="cuda"), 0)
    rng = np.random.default_rng(args.seed + 22000)
    classes = rng.integers(0, 8, (STEPS, 64))
    batches = np.empty_like(classes)
    for label in range(8):
        selected = classes == label
        batches[selected] = rng.choice(np.flatnonzero(trips["labels"] == label), selected.sum())
    checkpoint_path = output / "checkpoint.pt"
    previous_step, elapsed = 0, 0.
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        if state["arm"] != args.arm or state["seed"] != args.seed:
            raise ValueError("Mismatched resume checkpoint")
        model.load_state_dict(state["model"])
        averaged.load_state_dict(state["averaged"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        previous_step, elapsed = state["step"], state["seconds"]
    beginning = time.monotonic()
    with (output / "fit.jsonl").open("a" if previous_step else "w", buffering=1) as stream:
        for offset in range(previous_step, STEPS):
            indices = torch.from_numpy(batches[offset]).cuda()
            clean = raw[indices]
            timestep = torch.randint(1000, (len(indices),), device="cuda")
            noise = torch.randn_like(clean)
            alpha = alpha_bars[timestep, None, None]
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            prediction = model(noisy, timestep, labels[indices], starts[indices], paths[indices])
            error = (prediction - noise).square()
            denoising = error[:, :, 1:].mean() + .5 * error[:, :, 0].mean()
            auxiliary = denoising.new_tensor(0.)
            if args.arm != "B":
                auxiliary = alignment(model, args.arm, indices, labels[indices], vectors,
                                      centroids, permutation, alpha[:, 0, 0])
            loss = denoising + .1 * auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite alignment training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(averaged.parameters()), list(model.parameters()), .001)
            step = offset + 1
            if step == 1 or step % 1000 == 0:
                seconds = elapsed + time.monotonic() - beginning
                stream.write(json.dumps({"step": step, "seconds": seconds, "loss": loss.item(),
                                         "denoising": denoising.item(), "alignment": auxiliary.item()}) + "\n")
                save_atomic({"arm": args.arm, "seed": args.seed, "step": step, "seconds": seconds,
                             "model": model.state_dict(), "averaged": averaged.state_dict(),
                             "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state()}, checkpoint_path)
    save_atomic(averaged.state_dict(), output / "model.pt")
    result = {"arm": args.arm, "seed": args.seed, "steps": STEPS, "training_trips": len(raw),
              "seconds": elapsed + time.monotonic() - beginning, "teacher_seed": 42,
              "scope": "training complete, generation evaluation pending"}
    temporary = output / "complete.tmp.json"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("seed", type=int, choices=(42, 43))
    parser.add_argument("--output", type=Path, default=Path("g28-data/models"))
    parser.add_argument("--teachers", type=Path, default=Path("g28-data/teachers"))
    fit(parser.parse_args())
