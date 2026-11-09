"""G39 fixed-budget native diffusion with frozen train-only representation targets."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from g33_model import CLASSES, MotionDenoiser, clock, encode, fit_duration, pack_motion, predict
from g39_model import AlignedDenoiser, ARMS, alignment
from g39_targets import targets as load_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def load_train(path, arm):
    """Discard validation rows before computing any target/statistic."""
    receipt = json.loads(path.with_name("complete.json").read_text())
    if (receipt.get("complete") is not True or receipt.get("classes") != CLASSES
            or receipt.get("schema") != "g33-direct-sumo-full-legs-v1"
            or receipt["artifacts"][path.name]["sha256"] != digest(path)):
        raise ValueError("Completed immutable G33 dataset receipt required")
    with np.load(path, allow_pickle=False) as archive:
        partitions = archive["partition"].astype(str)
        indices = np.flatnonzero(partitions == "train")
        if not len(indices) or not set(partitions).issubset({"train", "val"}):
            raise ValueError("Only train and development val allowed; frozen test forbidden")
        rows = {name: archive[name][indices].copy() for name in
                ("ids", "parent_ids", "labels", "paths", "length", "duration")}
        offsets = archive["offset"]
        all_time, all_progress = archive["time"], archive["progress"]
        clocks, targets = [], []
        for i in indices:
            a, b = offsets[i:i+2]
            times = all_time[a:b].copy()
            progress = all_progress[a:b].copy()
            if arm == "relative_257":
                new_times = clock(float(times[-1]), arm)
                progress = np.interp(new_times, times, progress)
                times = new_times
            clocks.append(times)
            targets.append(encode(progress, times))
        # Group membership check uses identity metadata only, never val targets.
        train_parents = set(rows["parent_ids"].astype(str))
        val_parents = set(archive["parent_ids"][partitions == "val"].astype(str))
        if train_parents & val_parents:
            raise ValueError("Parent appears in both train and val")
    if set(rows["labels"]) != set(range(len(CLASSES))):
        raise ValueError("Five-class label support differs")
    if (not np.isfinite(rows["paths"]).all() or np.any(rows["length"] <= 0)
            or np.any(rows["duration"] <= 0) or len(set(rows["ids"])) != len(indices)):
        raise ValueError("Bad training routes/identity")
    # Each trajectory has equal mass; within it weight samples by physical dt.
    means = [float(np.average(v, weights=np.diff(t))) for v, t in zip(targets, clocks)]
    seconds = [float(np.average(v.astype(float)**2, weights=np.diff(t))) for v, t in zip(targets, clocks)]
    mean = float(np.mean(means)); sd = float(max(np.mean(seconds)-mean*mean, 1e-6)**.5)
    stats = {"motion_mean": mean, "motion_sd": sd,
             "motion_target": "log1p((delta_progress)/(delta_time/duration))",
             "stats_weighting": "equal_trajectory_then_physical_time_within_trajectory",
             "start_mean": rows["paths"][:, 0].mean(0).astype(float).tolist(),
             "start_sd": np.maximum(rows["paths"][:, 0].std(0), 1.).astype(float).tolist(),
             "duration_prior": fit_duration(rows["length"], rows["duration"], rows["labels"]),
             "classes": CLASSES, "arm": arm}
    targets = [((v-mean)/sd).astype(np.float32) for v in targets]
    return rows, clocks, targets, stats


def split_batch(indices, targets, token_budget, maximum_batch):
    """Length-sort the selected effective batch, without altering its sampling."""
    indices = sorted(indices, key=lambda i: len(targets[i]))
    chunks, current = [], []
    for i in indices:
        padded = int(math.ceil(len(targets[i])/4)*4)
        if current and ((len(current)+1)*padded > token_budget or len(current) >= maximum_batch):
            chunks.append(current); current = []
        current.append(i)
    if current:
        chunks.append(current)
    return chunks


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        rows, clocks, targets, stats = load_train(Path(args.data), "native_0_25s")
        device = torch.device(args.device)
        vectors, centers, permutation = load_features(rows, args.teachers, output, device)
        model = AlignedDenoiser(base=args.width).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        schedule = torch.cumprod(1-torch.linspace(.0001, .02, 1000, device=device), 0)
        config = {**vars(args), "classes": CLASSES, "training_rows": len(rows["ids"]),
                  "training_parents": len(set(rows["parent_ids"])),
                  "training_class_counts": np.bincount(rows["labels"], minlength=5).tolist(),
                  "data_sha256": digest(args.data), "model_source_sha256": digest(Path(__file__).with_name("g33_model.py")),
                  "alignment_source_sha256": digest(Path(__file__).with_name("g39_model.py")),
                  "targets_sha256": digest(output/"targets.json"),
                  "auxiliary_weight": .1, "auxiliary_t_weight": "alpha_bar",
                  "base_initial_sha256": hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.generator_state().values())).hexdigest(),
                  "train_source_sha256": digest(__file__), "parameters": sum(p.numel() for p in model.parameters()),
                  "fixed_final_ema_only": True, "validation_used_for_selection": False,
                  "duration_source": "training_classwise_loglength_ridge_empirical_residual_shared_between_arms",
                  "route_condition_source": "observed_training_routes; not an allowed final evaluation route source",
                  "loss": "noise_mse_physical_time_weighted_within_trajectory_then_equal_trajectory_mean",
                  "effective_batch": args.batch_size, "learning_rate": .0002, "ema_update": .001,
                  "note": "G39 train-only auxiliary alignment; original G33 denoiser exported without heads"}
        save(output/"config.json", config); save(output/"stats.json", stats)
        config_hash = digest(output/"config.json")
        rng = np.random.default_rng(args.seed+33000)
        class_indices = [np.flatnonzero(rows["labels"] == c) for c in range(5)]
        lengths = np.asarray([len(v) for v in targets])
        token_count = 0
        with (output/"training.jsonl").open("w", buffering=1) as log:
            for step in range(1, args.steps+1):
                classes = rng.integers(0, 5, args.batch_size)
                indices = np.asarray([rng.choice(class_indices[c]) for c in classes])
                chunks = split_batch(indices, targets, args.token_budget, args.microbatch)
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.; total_noise = 0.; total_aux = 0.
                for chunk in chunks:
                    ix = np.asarray(chunk)
                    batch = pack_motion([targets[i] for i in ix], [clocks[i] for i in ix],
                                        rows["labels"][ix], rows["paths"][ix], rows["length"][ix],
                                        rows["duration"][ix], stats, device)
                    x = batch["x"]
                    t = torch.randint(1000, (len(ix),), device=device)
                    noise = torch.randn_like(x)*batch["mask"]
                    alpha = schedule[t, None, None]
                    prediction = predict(model, batch, (alpha.sqrt()*x+(1-alpha).sqrt()*noise)*batch["mask"], t)
                    per_sample = ((prediction-noise).square()*batch["time_weights"]).sum((1, 2))
                    denoising = per_sample.sum()/args.batch_size
                    auxiliary = denoising.new_tensor(0.)
                    if args.arm != "B":
                        auxiliary = alignment(model, args.arm, torch.as_tensor(ix,device=device),
                            batch["labels"], vectors, centers, permutation, alpha[:,0,0]).sum()/args.batch_size
                    loss = denoising + .1*auxiliary
                    total_noise += float(denoising.detach()); total_aux += float(auxiliary.detach())
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    total_loss += float(loss.detach())
                    token_count += int(lengths[ix].sum())
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
                if not math.isfinite(grad_norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), .001)
                if step % 1000 == 0 or step == args.steps:
                    torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                                "torch_cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                                "numpy_rng": rng.bit_generator.state, "stats": stats,
                                "config_sha256": config_hash, "valid_tokens_seen": token_count},
                               output/"checkpoint-latest.tmp")
                    (output/"checkpoint-latest.tmp").replace(output/"checkpoint-latest.pt")
                if step == 1 or step % 100 == 0 or step == args.steps:
                    elapsed = time.monotonic()-started
                    record = {"step": step, "loss": total_loss, "denoising_loss": total_noise, "auxiliary_loss": total_aux, "gradient_norm": grad_norm,
                              "elapsed_s": elapsed, "updates_per_s": step/elapsed,
                              "microbatches": len(chunks), "valid_tokens_seen": token_count,
                              "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
                    log.write(json.dumps(record)+"\n"); print(json.dumps(record), flush=True)
        torch.save({"state_dict": ema.generator_state(), "width": args.width, "classes": CLASSES,
                    "arm": "native_0_25s", "alignment_arm": args.arm, "stats": stats, "steps": args.steps, "seed": args.seed}, output/"model.tmp")
        (output/"model.tmp").replace(output/"model.pt")
        if digest(output/"config.json") != config_hash:
            raise ValueError("Frozen training config changed")
        save(output/"complete.json", {"complete": True, "training_completed": True,
              "generation_evaluated": False, "seed": args.seed, "arm": args.arm, "steps": args.steps,
              "elapsed_s": time.monotonic()-started, "valid_tokens_seen": token_count,
              "parameters": config["parameters"], "model_sha256": digest(output/"model.pt"),
              "config_sha256": digest(output/"config.json"), "stats_sha256": digest(output/"stats.json")})
    except BaseException as error:
        save(output/"failed.json", {"complete": False, "error": repr(error), "elapsed_s": time.monotonic()-started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teachers", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.microbatch, args.token_budget, args.width) <= 0:
        parser.error("Positive training dimensions and budget required")
    train(args)
