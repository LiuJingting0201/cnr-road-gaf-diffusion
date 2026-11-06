"""Matched original-spatial and speed GAF teachers on the complete G35 cohort.

Only storage and dynamic cohort handling change from G33's fixed ResNet18 fit.
Raw native FCD coordinates feed spatial Hilbert encoding; speed is derived from
these coordinates. Compressed image shards avoid a large uncompressed disk cache.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from g33_prepare import CLASSES, SCHEMA, digest, save
from g33_teachers import metrics
from g35c_runtime import MIN_FREE_BYTES, MAX_NEW_BYTES, storage_state


CACHE_SCHEMA = "g35-matched-full-trip-gaf-cache-v1"
MAX_IMAGE_RAM = 10 * 1024**3
SHARD_ROWS = 256


def source_hashes():
    from g33_teachers import source_hashes as inherited_hashes
    files = inherited_hashes()
    files["g35c_teachers.py"] = digest(Path(__file__))
    files["g35c_runtime.py"] = digest(Path(__file__).with_name("g35c_runtime.py"))
    for name in ("g35_data_run.py", "g35b_data_run.py"):
        files[name] = digest(Path(__file__).with_name(name))
    files["g35_simulate.py"] = digest(Path(__file__).with_name("g35_simulate.py"))
    return files


class BudgetGuard:
    def __init__(self, root, output):
        self.root, output = Path(root).resolve(), Path(output).resolve()
        if output == self.root or not output.is_relative_to(self.root):
            raise ValueError("New teacher output must be inside the dedicated G35 budget root")
        self.last = -float("inf")

    def check(self, force=False, reserve=0):
        now = time.monotonic()
        if force or now - self.last >= 10:
            state = storage_state(self.root)
            if state["free_bytes"] - reserve < MIN_FREE_BYTES:
                raise RuntimeError("G35 teacher disk guard: less than 8 GiB free after reserved write")
            if state["budget_root_bytes"] + reserve > MAX_NEW_BYTES:
                raise RuntimeError("G35 teacher disk guard: combined G35b/G35c trees exceed the declared amended capacity")
            self.last = now
            return state


def validate_records(data, groups=None):
    n = len(data["ids"])
    if (n == 0 or len(set(data["ids"].astype(str))) != n
            or any(len(data[k]) != n for k in ("parent_ids", "partition", "labels"))
            or data["labels"].dtype.kind not in "iu" or not np.isin(data["labels"], np.arange(5)).all()
            or not np.isin(data["partition"], ("train", "val")).all()):
        raise ValueError("Invalid paired five-class identity/label/partition arrays")
    a = set(data["parent_ids"][data["partition"] == "train"].astype(str))
    b = set(data["parent_ids"][data["partition"] == "val"].astype(str))
    if a & b or not a or not b or "" in a | b:
        raise ValueError("Parent train/val overlap or missing parent support")
    for split in ("train", "val"):
        idx = np.flatnonzero(data["partition"] == split)
        counts = np.bincount(data["labels"][idx], minlength=5)
        if np.any(counts == 0):
            raise ValueError("Each G35 partition must cover every declared class")
        if groups is not None:
            group = groups[split]
            if group["legs"] != len(idx) or group["class_counts"] != dict(zip(CLASSES, map(int, counts))):
                raise ValueError("Dataset rows disagree with the merged population receipt")
    return n


def load_dataset(folder):
    folder = Path(folder)
    receipt = json.loads((folder / "complete.json").read_text())
    if ((folder / "failed.json").exists() or receipt.get("complete") is not True
            or receipt.get("schema") != SCHEMA or receipt.get("classes") != CLASSES
            or receipt.get("provenance_schema") != "g35-full-day-merged-v1"
            or receipt.get("training_ready") is not True):
        raise ValueError("Training-ready merged G35 full-day dataset required")
    checks = receipt.get("checks", {})
    for key in ("source_artifact_hashes", "source_metadata_alignment", "parent_train_val_disjoint",
                "parent_episode_disjoint", "source_test_absent", "preassigned_train_realization",
                "native_time_and_zero_motion_intervals_retained"):
        if checks.get(key) is not True:
            raise ValueError("Missing merged-cohort isolation check: " + key)
    spec = receipt["artifacts"]["dataset.npz"]
    path = folder / "dataset.npz"
    if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError("G35 immutable merged dataset hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = ("ids", "parent_ids", "partition", "labels", "offset", "time", "xy", "episodes")
        if any(key not in archive.files for key in required):
            raise ValueError("G35 raw FCD XY and identity arrays are required")
        data = {key: archive[key] for key in required}
    n = validate_records(data, receipt["groups"])
    off = data["offset"]
    if (off.dtype.kind not in "iu" or off.shape != (n + 1,) or off[0] != 0
            or off[-1] != len(data["time"]) or np.any(np.diff(off) < 2)
            or data["xy"].shape != (len(data["time"]), 2)
            or not np.isfinite(data["xy"]).all() or not np.isfinite(data["time"]).all()
            or len(data["episodes"]) != n):
        raise ValueError("Invalid raw native FCD XY/time layout")
    for i in range(n):
        a, b = off[i:i+2]
        t = data["time"][a:b]
        if t[0] != 0 or not np.allclose(np.diff(t), .25, atol=1e-7, rtol=0):
            raise ValueError("Original native clock changed")
        expected = (350101, 350102) if data["partition"][i] == "train" else (350201,)
        if int(data["episodes"][i]) not in expected:
            raise ValueError("Unexpected simulation episode/partition")
    return receipt, data


def speed_profile(times, xy):
    """G23's 128-point profile, including declared handling of valid short legs."""
    t, xy = np.asarray(times, dtype=np.float64), np.asarray(xy, dtype=np.float64)
    if (len(t) < 2 or xy.shape != (len(t), 2) or not np.isfinite(t).all()
            or not np.isfinite(xy).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Finite ordered native coordinate observations required")
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    elapsed = t - t[0]
    if elapsed[-1] < 1:
        return np.full(128, arc[-1] / elapsed[-1], dtype=np.float32), "subsecond_mean_arc_speed"
    seconds = np.arange(int(np.floor(elapsed[-1])) + 1)
    velocity = np.diff(np.interp(seconds, elapsed, arc))
    profile = np.interp(np.linspace(0, len(velocity) - 1, 128), np.arange(len(velocity)), velocity)
    return profile.astype(np.float32), "one_second_arc_increments"


def write_shard(path, images):
    if images.dtype != np.uint8 or images.shape[1:] != (224, 224, 3):
        raise ValueError("Original RGB uint8 image batch required")
    np.savez_compressed(path, images=images)
    # Round-trip equality matters for byte-valued original images, not lossy storage.
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["images"], images):
            raise ValueError("Compressed image shard changed the original RGB bytes")


def prepare(args):
    from g29_encode import sequence_job
    from cnr.original_gaf import encode_frozen
    from g23_modes import speed_series
    output = Path(args.output or args.cache).resolve()
    guard = BudgetGuard(args.budget_root, output)
    if output.is_relative_to(args.data.resolve()) or args.data.resolve().is_relative_to(output):
        raise ValueError("Teacher cache must not overlap the immutable source dataset")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        guard.check(force=True)
        receipt, raw = load_dataset(args.data)
        records = {key: raw[key] for key in ("ids", "parent_ids", "partition", "labels")}
        n = len(records["ids"])
        metadata = json.loads(args.encoding.read_text())
        keys = ("xy_min", "xy_max", "hilbert_eps", "channel_min", "channel_max")
        if any(key not in metadata for key in keys) or not np.isclose(metadata["hilbert_eps"], .0009):
            raise ValueError("Frozen E1/G29 grid64 epsilon=.0009 metadata required")
        velocity = np.empty((n, 128), dtype=np.float32)
        audit = {name: np.zeros(n, dtype=np.int64) for name in ("clipped", "native_points", "constant", "subsecond", "under_five")}
        shard_rows, artifacts, channel_clipped, equivalent = [], {}, 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, n, SHARD_ROWS):
                guard.check(force=True, reserve=SHARD_ROWS * 224 * 224 * 3 + 1024**2)
                end = min(n, start + SHARD_ROWS)
                jobs = []
                for i in range(start, end):
                    a, b = raw["offset"][i:i+2]
                    times, xy = raw["time"][a:b], raw["xy"][a:b]
                    velocity[i], method = speed_profile(times, xy)
                    audit["subsecond"][i] = method == "subsecond_mean_arc_speed"
                    audit["under_five"][i] = times[-1] < 5
                    if times[-1] >= 5:
                        original, _ = speed_series(times, xy)
                        if not np.array_equal(original, velocity[i]):
                            raise ValueError("Long-leg speed profile differs from inherited G23 bytes")
                        equivalent += 1
                    jobs.append((str(records["ids"][i]), int(records["labels"][i]), np.c_[times, xy], metadata))
                encoded = list(pool.map(sequence_job, jobs, chunksize=8))
                if any("failure" in row for row in encoded):
                    raise ValueError("Original spatial encoder failed; no silent cohort exclusions")
                sequences = np.stack([row["sequence"] for row in encoded])
                images = np.empty((end - start, 224, 224, 3), dtype=np.uint8)
                for local in range(0, len(encoded), 32):
                    guard.check()
                    batch, clipped = encode_frozen(sequences[local:local+32],
                        np.asarray(metadata["channel_min"]), np.asarray(metadata["channel_max"]), batch=32)
                    images[local:local+len(batch)] = batch
                    channel_clipped += clipped
                for i, row in enumerate(encoded, start):
                    audit["clipped"][i] = row["audit"]["clipped_points"]
                    audit["native_points"][i] = row["audit"]["native_points"]
                    audit["constant"][i] = row["audit"]["constant"]
                name = f"spatial-{start:08d}-{end:08d}.npz"
                guard.check(force=True, reserve=images.nbytes + 1024**2)
                write_shard(output / name, images)
                artifacts[name] = {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                shard_rows.append({"file": name, "start": start, "end": end})
                save(output / "progress.json", {"status": "encoding", "rows_complete": end,
                    "total_rows": n, "seconds": time.monotonic() - started})
                print(json.dumps({"component": "G35 paired GAF cache", "rows": end, "total": n}), flush=True)
        if not np.isfinite(velocity).all() or np.any(velocity < 0):
            raise ValueError("Nonfinite/negative speed profile")
        guard.check(force=True, reserve=velocity.nbytes + n * 2048 + 1024**2)
        np.savez_compressed(output / "index.npz", **records, velocity=velocity)
        artifacts["index.npz"] = {"sha256": digest(output / "index.npz"), "bytes": (output / "index.npz").stat().st_size}
        groups = {}
        for split in ("train", "val"):
            ix = np.flatnonzero(records["partition"] == split)
            groups[split] = {"rows": len(ix), "parents": len(set(records["parent_ids"][ix].astype(str))),
                "class_counts": np.bincount(records["labels"][ix], minlength=5).tolist(),
                "spatial_coordinate_clipped_points": int(audit["clipped"][ix].sum()),
                "spatial_native_points": int(audit["native_points"][ix].sum()),
                "spatial_constant_sequences": int(audit["constant"][ix].sum()),
                "speed_subsecond_fallback_legs": int(audit["subsecond"][ix].sum()),
                "speed_under_five_second_legs": int(audit["under_five"][ix].sum()),
                "speed_values_above_35_mps": int(np.sum(velocity[ix] > 35.)),
                "speed_values_total": int(velocity[ix].size), "speed_max_mps": float(velocity[ix].max())}
        result = {"complete": True, "schema": CACHE_SCHEMA, "classes": CLASSES, "groups": groups,
            "paired_cohort_all_encoded": True, "parent_overlap": 0, "training_ready": True,
            "spatial_shards": shard_rows, "storage": "Lossless uint8 RGB compressed NPZ shards; loaded in RAM for training",
            "spatial": {"source": "Unaltered raw G35 native FCD XY; no speed features or decoded-coordinate substitution",
                "sequence": "Inherited Hilbert/squeeze/duplicate 224", "encoding_metadata_sha256": digest(args.encoding),
                "frozen_parameters": {key: metadata[key] for key in keys}, "channel_values_clipped": int(channel_clipped)},
            "speed": {"source": "Observed XY cumulative arc; one-second increments, 128 relative-index samples",
                "image": "Unchanged G23 speed_images, fixed35m/s scale,8bins GASF/GADF/MTF; G28 resizing/normalization",
                "short_leg_rule": ">=1s uses inherited profile; <1s repeats observed arc/span mean128 times; no row exclusion",
                "inherited_profile_bitwise_verified_for_duration_at_least_5s": equivalent,
                "label": "Five participant classes, not speed-bin labels",
                "high_frequency_limit": "One-second summaries resampled to128 points do not preserve all native acceleration"},
            "sources": {"data_receipt_sha256": digest(args.data / "complete.json"),
                "data_sha256": digest(args.data / "dataset.npz"), "code_sha256": source_hashes(),
                "source_population": receipt["groups"]}, "artifacts": artifacts,
            "resource_limits": {"minimum_free_bytes": MIN_FREE_BYTES, "maximum_g35_tree_bytes": MAX_NEW_BYTES},
            "seconds": time.monotonic() - started}
        guard.check(force=True, reserve=1024**2)
        save(output / "complete.json", result)
        save(output / "progress.json", {"status": "complete", "rows": n, "seconds": time.monotonic() - started})
        print(json.dumps({"complete": True, "component": "G35 paired GAF cache", "groups": groups}), flush=True)
    except BaseException as error:
        save(output / "failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic() - started})
        raise


def load_cache(folder, view):
    folder = Path(folder)
    receipt = json.loads((folder / "complete.json").read_text())
    if ((folder / "failed.json").exists() or receipt.get("complete") is not True
            or receipt.get("schema") != CACHE_SCHEMA or receipt.get("classes") != CLASSES
            or receipt.get("training_ready") is not True or receipt.get("paired_cohort_all_encoded") is not True):
        raise ValueError("Completed matched full-cohort G35 GAF cache required")
    if receipt["sources"]["code_sha256"] != source_hashes():
        raise ValueError("Encoding/training source changed after cache preparation")
    shards = receipt["spatial_shards"]
    if set(receipt["artifacts"]) != {"index.npz", *[row["file"] for row in shards]}:
        raise ValueError("Unexpected cache artifact inventory")
    for name, spec in receipt["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Cache artifact filename must remain local")
        path = folder / name
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("Cache artifact hash mismatch: " + name)
    with np.load(folder / "index.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n = validate_records(data)
    if (data["velocity"].shape != (n, 128) or not np.isfinite(data["velocity"]).all()
            or np.any(data["velocity"] < 0)):
        raise ValueError("Invalid cached speed profiles")
    if view not in ("spatial", "speed"):
        raise ValueError("Unknown teacher view")
    if view == "spatial" and n * 224 * 224 * 3 > MAX_IMAGE_RAM:
        raise MemoryError("Spatial uint8 cache exceeds the declared10GiB RAM loading budget")
    values = np.empty((n, 224, 224, 3), np.uint8) if view == "spatial" else data["velocity"]
    end = 0
    for shard in shards:
        if shard["start"] != end or not end < shard["end"] <= n:
            raise ValueError("Image shard rows are not a contiguous nonoverlapping full cohort")
        if view == "spatial":
            with np.load(folder / shard["file"], allow_pickle=False) as archive:
                images = archive["images"]
                if images.dtype != np.uint8 or images.shape != (shard["end"] - end, 224, 224, 3):
                    raise ValueError("Image shard dimensions/dtype changed")
                values[end:shard["end"]] = images
        end = shard["end"]
    if end != n:
        raise ValueError("Image shards do not cover all paired rows")
    for split in ("train", "val"):
        ix = np.flatnonzero(data["partition"] == split)
        group = receipt["groups"][split]
        if (len(ix) != group["rows"] or np.bincount(data["labels"][ix], minlength=5).tolist() != group["class_counts"]):
            raise ValueError("Cache cohort counts changed")
    return receipt, data, values


def train(args):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet18
    from g28_teachers import image_inputs
    output = args.output / f"{args.view}-42"
    guard = BudgetGuard(args.budget_root, output)
    output.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    try:
        guard.check(force=True)
        prepared, data, values = load_cache(args.cache, args.view)
        indices = {split: np.flatnonzero(data["partition"] == split) for split in ("train", "val")}
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
            "label": "five participant classes", "scope": "self-trained feedback component, not generation-quality result",
            "cohort": prepared["groups"], "baseline_training_recipe": "Unchanged G33 ResNet18/ImageNet final50 seed42",
            "resource_limits": {"minimum_free_bytes": MIN_FREE_BYTES, "maximum_g35_tree_bytes": MAX_NEW_BYTES}}
        save(output / "config.json", config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        generator = np.random.default_rng(42)
        with (output / "history.jsonl").open("w", buffering=1) as stream:
            for epoch in range(1, 51):
                model.train()
                order = generator.permutation(indices["train"])
                losses = []
                for base in range(0, len(order), 64):
                    guard.check()
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
        guard.check(force=True, reserve=64 * 1024**2)
        torch.save(model.state_dict(), output / "model.pt")
        model.eval()
        scores = {}
        for split, selected in indices.items():
            logits, features = [], []
            with torch.no_grad():
                for base in range(0, len(selected), 64):
                    guard.check()
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
            guard.check(force=True, reserve=logits.nbytes + features.nbytes + len(selected)*2048 + 1024**2)
            np.savez_compressed(output / f"{split}-predictions.npz", ids=data["ids"][selected],
                                parent_ids=data["parent_ids"][selected], labels=data["labels"][selected],
                                logits=logits, features=features)
        result = {"complete": True, "view": args.view, "classes": CLASSES, "selected_epoch": 50,
                  "train": scores["train"], "val": scores["val"], "seconds": time.monotonic()-start_time,
                  "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                  "config_sha256": digest(output / "config.json"),
                  "artifacts": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                                for name in ("model.pt", "train-predictions.npz", "val-predictions.npz")},
                  "limits": ["Preassigned-parent holdout with separate training/development simulation episodes, same public demand/map; not unseen-city or unseen-bus-line validation.",
                             "This CNN is self-trained, not supplied by the CNR group.",
                             "Teacher classification and calibration do not establish GAF generation benefit.",
                             "Spatial original encoding and speed angular encoding remain distinct."]}
        guard.check(force=True, reserve=1024**2)
        save(output / "complete.json", result)
        print(json.dumps({"complete": True, "view": args.view, "val": scores["val"]}), flush=True)
    except BaseException as error:
        save(output / "failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic()-start_time})
        raise


def self_test():
    import unittest
    from unittest.mock import patch

    class Checks(unittest.TestCase):
        def records(self):
            return {"ids": np.asarray([f"leg{i}" for i in range(10)]),
                "parent_ids": np.asarray([f"parent{i}" for i in range(10)]),
                "labels": np.tile(np.arange(5), 2),
                "partition": np.asarray(["train"] * 5 + ["val"] * 5)}

        def fixture_cache(self, folder):
            data = self.records()
            np.savez_compressed(folder / "index.npz", **data, velocity=np.full((10, 128), 3, np.float32))
            image = np.random.default_rng(42).integers(0, 256, (10, 224, 224, 3), dtype=np.uint8)
            shards = []
            for start, end in ((0, 4), (4, 10)):
                name = f"spatial-{start}-{end}.npz"
                write_shard(folder / name, image[start:end])
                shards.append({"file": name, "start": start, "end": end})
            names = ["index.npz", *[row["file"] for row in shards]]
            receipt = {"complete": True, "schema": CACHE_SCHEMA, "classes": CLASSES,
                "training_ready": True, "paired_cohort_all_encoded": True,
                "sources": {"code_sha256": {"fixture": "sha"}}, "spatial_shards": shards,
                "groups": {split: {"rows": 5, "class_counts": [1] * 5} for split in ("train", "val")},
                "artifacts": {name: {"sha256": digest(folder / name), "bytes": (folder / name).stat().st_size}
                              for name in names}}
            save(folder / "complete.json", receipt)
            return image

        def test_parent_and_population_checks(self):
            data = self.records()
            self.assertEqual(validate_records(data), 10)
            data["parent_ids"][5] = data["parent_ids"][0]
            with self.assertRaises(ValueError):
                validate_records(data)

        def test_missing_class_and_test_partition_rejected(self):
            for key, value in (("labels", 0), ("partition", "test")):
                data = self.records()
                data[key][4] = value
                with self.assertRaises(ValueError):
                    validate_records(data)

        def test_subsecond_profile_preserves_known_mean_speed(self):
            value, rule = speed_profile(np.asarray([0., .25, .5]), np.asarray([[0., 0.], [1., 0.], [2., 0.]]))
            self.assertEqual(rule, "subsecond_mean_arc_speed")
            np.testing.assert_array_equal(value, np.full(128, 4, np.float32))

        def test_short_and_long_constant_speed_profiles(self):
            for duration in (1, 1.25, 4.75, 5, 20):
                t = np.arange(int(duration * 4) + 1) * .25
                value, rule = speed_profile(t, np.c_[3 * t, 4 * t])
                self.assertEqual(rule, "one_second_arc_increments")
                np.testing.assert_array_equal(value, np.full(128, 5, np.float32))

        def test_stop_then_motion_arc_profile(self):
            t = np.arange(25) * .25
            xy = np.c_[np.maximum(t - 2, 0) * 3, np.zeros(len(t))]
            value, _ = speed_profile(t, xy)
            expected = np.interp(np.linspace(0, 5, 128), np.arange(6), [0, 0, 3, 3, 3, 3]).astype(np.float32)
            np.testing.assert_array_equal(value, expected)

        def test_invalid_speed_input(self):
            with self.assertRaises(ValueError):
                speed_profile(np.asarray([0., 0.]), np.zeros((2, 2)))

        def test_compressed_cache_matches_rgb_and_speed(self):
            with tempfile.TemporaryDirectory() as temporary, patch(__name__ + ".source_hashes", return_value={"fixture": "sha"}):
                folder = Path(temporary)
                original = self.fixture_cache(folder)
                _, data, image = load_cache(folder, "spatial")
                np.testing.assert_array_equal(image, original)
                _, _, speed = load_cache(folder, "speed")
                np.testing.assert_array_equal(speed, data["velocity"])

        def test_changed_shard_and_row_overlap_rejected(self):
            with tempfile.TemporaryDirectory() as temporary, patch(__name__ + ".source_hashes", return_value={"fixture": "sha"}):
                folder = Path(temporary)
                self.fixture_cache(folder)
                path = folder / "complete.json"
                receipt = json.loads(path.read_text())
                receipt["spatial_shards"][1]["start"] = 3
                save(path, receipt)
                with self.assertRaises(ValueError):
                    load_cache(folder, "speed")
                receipt["spatial_shards"][1]["start"] = 4
                save(path, receipt)
                (folder / "spatial-0-4.npz").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    load_cache(folder, "spatial")

        def test_guard_includes_reserved_write(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                guard = BudgetGuard(root, root / "new-cache")
                with patch(__name__ + ".storage_state", return_value={"free_bytes": MIN_FREE_BYTES,
                                                                          "budget_root_bytes": MAX_NEW_BYTES}):
                    guard.check(force=True)
                    with self.assertRaises(RuntimeError):
                        guard.check(force=True, reserve=1)

        def test_metrics_inherit_five_class_contract(self):
            result = metrics(np.arange(5), np.eye(5) * 20)
            self.assertEqual(result["accuracy"], 1)
            self.assertEqual(result["macro_f1"], 1)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"complete": True, "component": "G35 teacher/cache synthetic checks",
                      "checks": result.testsRun, "gpu_training_tested": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "self-test"))
    parser.add_argument("--view", choices=("spatial", "speed"))
    parser.add_argument("--data", type=Path, default=Path("g35-data/direct-legs"))
    parser.add_argument("--encoding", type=Path, default=Path("../data/processed/original-hilbert64grid224/dataset.json"))
    parser.add_argument("--cache", type=Path, default=Path("g35-data/gaf-cache"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-root", type=Path, default=Path("g35-data"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", type=Path, default=Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
    elif args.action == "prepare":
        if args.workers < 1:
            parser.error("--workers must be positive")
        prepare(args)
    else:
        if args.view is None:
            parser.error("train requires --view")
        args.output = args.output or Path("g35-data/teachers")
        train(args)

