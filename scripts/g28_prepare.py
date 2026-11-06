"""Cache paired native spatial images and speed series on training partitions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from g19_gaf import images


def read_native(root, split):
    identifiers, labels, tracks = [], [], []
    files = sorted((root / split).glob("*/quarter.npz"))
    if not files:
        raise FileNotFoundError(f"No native observations in {root / split}")
    for file in files:
        with np.load(file) as native, np.load(file.parent / "trips.npz") as trips:
            if not np.array_equal(native["ids"], trips["ids"]):
                raise ValueError(f"Native ID mismatch in {file}")
            identifiers.extend(native["ids"])
            labels.extend(trips["labels"])
            offsets = native["offset"]
            observations = native["txy"]
            for start, end in zip(offsets[:-1], offsets[1:]):
                track = observations[start:end]
                if not np.isfinite(track).all() or np.any(np.diff(track[:, 0]) <= 0):
                    raise ValueError(f"Invalid native observation times in {file}")
                # Preserve all recorded XY observations, with no synthetic endpoint.
                tracks.append(track[:, 1:].copy())
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Repeated complete-trip identifier")
    return np.asarray(identifiers), np.asarray(labels), tracks


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "prepared.json").exists():
        return
    manifest = {"spatial_protocol": "native quarter-second observations, no endpoint append",
                "speed_protocol": "G23 raw-observation speed series, 128 samples",
                "splits": {}}
    split_ids = {}
    for split in ("train", "monitor"):
        identifiers, labels, tracks = read_native(args.trips, split)
        with np.load(args.speed / f"{split}.npz") as speed:
            lookup = {ident: i for i, ident in enumerate(speed["ids"])}
            index = np.asarray([lookup[ident] for ident in identifiers])
            if not np.array_equal(labels, speed["participant"][index]):
                raise ValueError("Paired-view participant labels disagree")
            velocity = speed["v"][index].copy()
        if not np.isfinite(velocity).all():
            raise ValueError("Invalid speed series")
        destination = args.output / f"{split}-images.npy"
        temporary = args.output / f"{split}-images.tmp.npy"
        encoded = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                          shape=(len(labels), 224, 224, 3))
        flags = []
        # Bound worker submissions to avoid copying the whole native corpus at once.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for base in range(0, len(tracks), 16 * args.workers):
                batches = [tracks[start:start + 16]
                           for start in range(base, min(base + 16 * args.workers, len(tracks)), 16)]
                for batch, (rgb, valid) in enumerate(pool.map(images, batches)):
                    start = base + 16 * batch
                    encoded[start:start + len(rgb)] = rgb
                    flags.extend(valid)
        encoded.flush()
        del encoded
        valid = np.asarray(flags)
        np.savez_compressed(args.output / f"{split}.npz", ids=identifiers,
                            labels=labels, velocity=velocity, valid=valid)
        temporary.replace(destination)
        split_ids[split] = set(identifiers)
        manifest["splits"][split] = {
            "rows": len(labels), "valid": int(valid.sum()),
            "counts": np.bincount(labels, minlength=8).tolist(),
            "ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest(),
            "image_sha256": hashlib.file_digest(destination.open("rb"), "sha256").hexdigest(),
        }
        print(json.dumps({"split": split, **manifest["splits"][split]}), flush=True)
        if not valid.all():
            raise ValueError(f"{split} spatial encoding failures require audit before matched fitting")
    if split_ids["train"] & split_ids["monitor"]:
        raise ValueError("Training/monitor identifier overlap")
    (args.output / "prepared.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=Path, default=Path("g22-data"))
    parser.add_argument("--speed", type=Path, default=Path("../g23-job/g23-data"))
    parser.add_argument("--output", type=Path, default=Path("g28-data/cache"))
    parser.add_argument("--workers", type=int, default=10)
    prepare(parser.parse_args())
