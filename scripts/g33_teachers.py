"""Fixed matched five-class spatial/speed GAF feedback readers for G33.

Only the 925/126 inherited G31 complete vehicle legs are used. Spatial images
use the original frozen Hilbert/RGB pipeline on unaltered source coordinates;
speed images reuse G23/G28. No validation checkpoint selection or encoding fit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
from pathlib import Path
import time

import numpy as np

from g33_prepare import CLASSES, SCHEMA, digest, save


def source_hashes():
    from g29_encode import sequence_job
    from cnr.original_gaf import encode_frozen, hilbert_index, squeeze, duplicate
    from cnr.representations import hilbert_numpy
    from g23_modes import speed_images, speed_series
    from g28_teachers import image_inputs
    functions = (sequence_job, encode_frozen, hilbert_index, hilbert_numpy, squeeze, duplicate,
                 speed_images, speed_series, image_inputs, digest)
    return {str(Path(inspect.getsourcefile(fn)).name): digest(Path(inspect.getsourcefile(fn)))
            for fn in functions} | {"g33_teachers.py": digest(Path(__file__))}


def prepare(args):
    from g29_encode import sequence_job
    from cnr.original_gaf import encode_frozen
    from g23_modes import speed_series
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = json.loads((args.data / "complete.json").read_text())
        if (receipt.get("complete") is not True or receipt["schema"] != SCHEMA
                or receipt["classes"] != CLASSES
                or digest(args.data / "dataset.npz") != receipt["artifacts"]["dataset.npz"]["sha256"]
                or digest(args.source / "legs.npz") != receipt["sources"]["legs_sha256"]):
            raise ValueError("Completed direct-leg dataset and immutable original observations required")
        with np.load(args.data / "dataset.npz", allow_pickle=False) as data:
            records = {key: data[key] for key in ("ids", "parent_ids", "partition", "labels")}
        with np.load(args.source / "legs.npz", allow_pickle=False) as src:
            raw = {key: src[key] for key in ("ids", "parent_base_ids", "partitions", "labels", "offset", "txy")}
        lookup = {str(ident): i for i, ident in enumerate(raw["ids"])}
        metadata = json.loads(args.encoding.read_text())
        for key in ("xy_min", "xy_max", "hilbert_eps", "channel_min", "channel_max"):
            if key not in metadata:
                raise ValueError("Frozen original encoding metadata incomplete: " + key)
        if not np.isclose(metadata["hilbert_eps"], .0009):
            raise ValueError("G33 inherits the E1/G29 frozen grid64 epsilon=.0009")
        jobs, velocity = [], []
        for i, ident in enumerate(records["ids"]):
            j = lookup[str(ident)]
            if (str(raw["parent_base_ids"][j]) != str(records["parent_ids"][i])
                    or str(raw["partitions"][j]) != str(records["partition"][i])
                    or int(raw["labels"][j])-1 != int(records["labels"][i])):
                raise ValueError("Raw source and generation preparation IDs/labels/splits differ")
            a, b = raw["offset"][j:j+2]
            track = raw["txy"][a:b]
            jobs.append((str(ident), int(records["labels"][i]), track, metadata))
            profile, _ = speed_series(track[:, 0], track[:, 1:])
            velocity.append(profile)
        velocity = np.stack(velocity)
        if not np.isfinite(velocity).all() or np.any(velocity < 0):
            raise ValueError("Nonfinite/negative physical speed profile")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            encoded = list(pool.map(sequence_job, jobs, chunksize=8))
        if any("failure" in row for row in encoded):
            raise ValueError("Original spatial encoder failed; no silent common-cohort exclusions")
        sequences = np.stack([row["sequence"] for row in encoded])
        images_path = output / "spatial.npy"
        images = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8,
                                          shape=(len(encoded), 224, 224, 3))
        channel_clipped = 0
        for start in range(0, len(encoded), 32):
            batch, clipped = encode_frozen(sequences[start:start+32], np.asarray(metadata["channel_min"]),
                                           np.asarray(metadata["channel_max"]), batch=32)
            images[start:start+len(batch)] = batch
            channel_clipped += clipped
        images.flush()
        del images
        np.savez_compressed(output / "index.npz", **records, velocity=velocity)
        groups = {}
        for split in ("train", "val"):
            indices = np.flatnonzero(records["partition"] == split)
            if len(indices) != {"train": 925, "val": 126}[split]:
                raise ValueError("Fixed G33 population changed")
            groups[split] = {"rows": len(indices), "class_counts": np.bincount(records["labels"][indices], minlength=5).tolist(),
                "spatial_coordinate_clipped_points": sum(encoded[i]["audit"]["clipped_points"] for i in indices),
                "spatial_native_points": sum(encoded[i]["audit"]["native_points"] for i in indices),
                "spatial_constant_sequences": sum(encoded[i]["audit"]["constant"] for i in indices),
                "speed_values_above_35_mps": int(np.sum(velocity[indices] > 35.)),
                "speed_values_total": int(velocity[indices].size),
                "speed_max_mps": float(velocity[indices].max())}
        a = set(records["parent_ids"][records["partition"] == "train"])
        b = set(records["parent_ids"][records["partition"] == "val"])
        if a & b:
            raise ValueError("Parent split overlap")
        result = {"complete": True, "classes": CLASSES, "groups": groups,
            "paired_cohort_all_encoded": True, "parent_overlap": 0,
            "spatial": {"source": "Original unaltered G31 XY; no speed feature input", "sequence": "Inherited Hilbert/squeeze/duplicate 224",
                        "encoding_metadata_sha256": digest(args.encoding),
                        "frozen_parameters": {key: metadata[key] for key in ("xy_min", "xy_max", "hilbert_eps", "channel_min", "channel_max")},
                        "channel_values_clipped": int(channel_clipped)},
            "speed": {"source": "G23 speed_series: one-second cumulative observed XY arc increments; resample 128 by relative index",
                      "image": "G23 speed_images; fixed35m/s scale,8bins GASF/GADF/MTF; G28 normalization/resizing",
                      "label": "five participant classes, not legacy six speed modes",
                      "high_frequency_limit": "128 relative-time samples of one-second speed summaries do not retain all .25s acceleration"},
            "sources": {"data_receipt_sha256": digest(args.data / "complete.json"),
                        "data_sha256": digest(args.data / "dataset.npz"), "raw_legs_sha256": digest(args.source / "legs.npz"),
                        "code_sha256": source_hashes()},
            "artifacts": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                          for name in ("index.npz", "spatial.npy")}}
        save(output / "complete.json", result)
        print(json.dumps({"complete": True, "component": "paired_gaf_cache", "groups": groups}), flush=True)
    except Exception as error:
        save(output / "failed.json", {"complete": False, "error": repr(error)})
        raise


def metrics(labels, logits):
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if (logits.shape != (len(labels), 5) or not np.isfinite(logits).all()
            or labels.dtype.kind not in "iu" or not np.isin(labels, np.arange(5)).all()):
        raise ValueError("Invalid five-class predictions")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    prob = np.exp(log_prob)
    prediction = logits.argmax(1)
    cm = np.bincount(5*labels+prediction, minlength=25).reshape(5, 5)
    precision = cm.diagonal() / np.maximum(cm.sum(0), 1)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    f1 = 2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1), 1)
    confidence, correct = prob.max(axis=1), prediction == labels
    bins = np.minimum((confidence*15).astype(int), 14)
    ece = sum(np.mean(bins == k)*abs(float(confidence[bins == k].mean())-float(correct[bins == k].mean()))
              for k in range(15) if np.any(bins == k))
    onehot = np.eye(5)[labels]
    return {"n": len(labels), "accuracy": float(correct.mean()), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.sum(f1*cm.sum(1))/len(labels)), "per_class_f1": f1.tolist(),
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "confusion": cm.tolist(), "nll": float(-log_prob[np.arange(len(labels)), labels].mean()),
            "brier_multiclass_sum": float(np.square(prob-onehot).sum(1).mean()),
            "ece_15_equal_width": float(ece), "calibration": "uncalibrated final-model probabilities; no temperature fitting"}


def train(args):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet18
    from g28_teachers import image_inputs
    output = args.output / f"{args.view}-42"
    output.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    try:
        prepared = json.loads((args.cache / "complete.json").read_text())
        if not prepared.get("complete") or prepared["classes"] != CLASSES:
            raise ValueError("Completed paired five-class cache required")
        if prepared["sources"]["code_sha256"] != source_hashes():
            raise ValueError("Encoding/training source changed after cache preparation")
        for name, spec in prepared["artifacts"].items():
            if digest(args.cache / name) != spec["sha256"]:
                raise ValueError("Cache artifact hash mismatch")
        with np.load(args.cache / "index.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        values = (np.load(args.cache / "spatial.npy", mmap_mode="r")
                  if args.view == "spatial" else data["velocity"])
        indices = {split: np.flatnonzero(data["partition"] == split) for split in ("train", "val")}
        if len(indices["train"]) != 925 or len(indices["val"]) != 126:
            raise ValueError("G33 cohort size changed")
        counts = np.bincount(data["labels"][indices["train"]], minlength=5)
        if np.any(counts == 0):
            raise ValueError("Every class must be supported in training")
        torch.set_num_threads(2)
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.benchmark = False
        model = resnet18(weights=None)
        pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        del pretrained["fc.weight"], pretrained["fc.bias"]
        model.fc = nn.Linear(512, 5)
        missing = model.load_state_dict(pretrained, strict=False)
        if set(missing.missing_keys) != {"fc.weight", "fc.bias"} or missing.unexpected_keys:
            raise ValueError("Unexpected ImageNet initialization mismatch")
        model = model.cuda()
        weights = counts.astype(float)**-.5
        weights /= weights.mean()
        class_weights = torch.as_tensor(weights, dtype=torch.float32, device="cuda")
        labels = torch.as_tensor(data["labels"], dtype=torch.long, device="cuda")
        config = {"view": args.view, "classes": CLASSES, "seed": 42, "epochs": 50, "batch_size": 64,
            "optimizer": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4,
            "scheduler": "constant", "class_weight": "inverse_sqrt_training_counts_normalized_mean_one",
            "class_counts": counts.tolist(), "class_weights": weights.tolist(),
            "checkpoint": "final_epoch_50_only", "cache_receipt_sha256": digest(args.cache / "complete.json"),
            "pretrained_sha256": digest(args.pretrained), "code_sha256": source_hashes(),
            "device": torch.cuda.get_device_name(), "precision": "float32", "torch_version": torch.__version__,
            "label": "five participant classes", "scope": "self-trained feedback component, not generation-quality result"}
        save(output / "config.json", config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        generator = np.random.default_rng(42)
        with (output / "history.jsonl").open("w", buffering=1) as stream:
            for epoch in range(1, 51):
                model.train()
                order = generator.permutation(indices["train"])
                losses = []
                for base in range(0, len(order), 64):
                    batch = order[base:base+64]
                    logits = model(image_inputs(args.view, values[batch]))
                    loss = F.cross_entropy(logits, labels[batch], weight=class_weights)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append((float(loss.item()), len(batch)))
                row = {"epoch": epoch, "training_loss_batch_size_weighted": sum(x*n for x, n in losses)/sum(n for _, n in losses),
                       "seconds": time.monotonic()-start_time}
                stream.write(json.dumps(row)+"\n")
                print(json.dumps({"view": args.view, **row}), flush=True)
        torch.save(model.state_dict(), output / "model.pt")
        model.eval()
        scores = {}
        for split, selected in indices.items():
            logits, features = [], []
            with torch.no_grad():
                for base in range(0, len(selected), 64):
                    batch = selected[base:base+64]
                    hidden = []
                    handle = model.avgpool.register_forward_hook(lambda _module, _input, value: hidden.append(value.flatten(1)))
                    try:
                        output_logits = model(image_inputs(args.view, values[batch]))
                    finally:
                        handle.remove()
                    logits.append(output_logits.cpu().numpy())
                    features.append(hidden[0].cpu().numpy())
            logits, features = np.concatenate(logits), np.concatenate(features)
            scores[split] = metrics(data["labels"][selected], logits)
            np.savez_compressed(output / f"{split}-predictions.npz", ids=data["ids"][selected],
                                parent_ids=data["parent_ids"][selected], labels=data["labels"][selected],
                                logits=logits, features=features)
        result = {"complete": True, "view": args.view, "classes": CLASSES, "selected_epoch": 50,
                  "train": scores["train"], "val": scores["val"], "seconds": time.monotonic()-start_time,
                  "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                  "config_sha256": digest(output / "config.json"),
                  "artifacts": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                                for name in ("model.pt", "train-predictions.npz", "val-predictions.npz")},
                  "limits": ["Single-episode parent holdout, not independent-city or unseen-route validation.",
                             "This CNN is self-trained, not supplied by the CNR group.",
                             "Teacher classification and calibration do not establish GAF generation benefit.",
                             "Spatial original encoding and speed angular encoding remain distinct."]}
        save(output / "complete.json", result)
        print(json.dumps({"complete": True, "view": args.view, "val": scores["val"]}), flush=True)
    except Exception as error:
        save(output / "failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic()-start_time})
        raise


def self_test():
    labels = np.arange(5, dtype=np.int64)
    logits = np.eye(5)*20.
    result = metrics(labels, logits)
    assert result["accuracy"] == result["macro_f1"] == 1.
    assert result["nll"] < 1e-6 and result["brier_multiclass_sum"] < 1e-12
    uniform = metrics(labels, np.zeros((5, 5)))
    assert abs(uniform["nll"]-np.log(5.)) < 1e-12
    assert abs(uniform["brier_multiclass_sum"]-.8) < 1e-12
    try:
        metrics(labels, np.full((5, 5), np.nan))
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite predictions accepted")
    print("g33_teacher_metric_checks_passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "self-test"))
    parser.add_argument("--view", choices=("spatial", "speed"))
    parser.add_argument("--data", type=Path, default=Path("g33-data/direct-legs"))
    parser.add_argument("--source", type=Path, default=Path("../g31-job/g31-data/legs/pilot-300001"))
    parser.add_argument("--encoding", type=Path, default=Path("../data/processed/original-hilbert64grid224/dataset.json"))
    parser.add_argument("--cache", type=Path, default=Path("g33-data/gaf-cache"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", type=Path, default=Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
    elif args.action == "prepare":
        args.output = args.output or args.cache
        prepare(args)
    else:
        if args.view is None:
            parser.error("train requires --view")
        args.output = args.output or Path("g33-data/teachers")
        train(args)
