"""Matched-source encoding ladder using frozen inherited Hilbert metadata."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from cnr.original_gaf import encode_frozen, hilbert_index, raw_channels, squeeze, duplicate
from cnr.representations import hilbert_numpy

DOMAINS = ("cnr", "sumo")
SOURCES = (*DOMAINS, "sumo-duration")
SPLITS = ("train", "val")
STEPS = 224


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_raw(path):
    with np.load(path) as data:
        rows = {key: data[key] for key in data.files}
    required = {"ids", "labels", "txy", "offset"}
    if not required.issubset(rows):
        raise ValueError(f"Missing fields in {path}")
    if len(set(rows["ids"])) != len(rows["ids"]):
        raise ValueError(f"Duplicate IDs in {path}")
    if rows["offset"][0] != 0 or rows["offset"][-1] != len(rows["txy"]):
        raise ValueError(f"Invalid offsets in {path}")
    if len(rows["offset"]) != len(rows["ids"]) + 1:
        raise ValueError(f"ID/offset mismatch in {path}")
    if not np.isfinite(rows["txy"]).all() or not np.isin(rows["labels"], np.arange(6)).all():
        raise ValueError(f"Invalid raw values in {path}")
    return rows


def sampled(values):
    positions = np.linspace(0, len(values) - 1, STEPS)
    if values.ndim == 1:
        return np.interp(positions, np.arange(len(values)), values)
    return np.stack([np.interp(positions, np.arange(len(values)), axis) for axis in values.T], axis=1)


def sequence_job(job):
    ident, label, track, metadata = job
    times, xy = track[:, 0], track[:, 1:]
    if len(track) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid time order for {ident}")
    if not np.allclose(np.diff(times), .25, atol=1e-6, rtol=0):
        raise ValueError(f"Non-native cadence in selected continuous track {ident}")
    low, high = np.asarray(metadata["xy_min"]), np.asarray(metadata["xy_max"])
    unit = (xy - low) / (high - low)
    clipped = int(np.any((unit < 0) | (unit > 1), axis=1).sum())
    unit = unit.clip(0, 1)
    native = hilbert_numpy(unit, metadata["hilbert_eps"])
    # Validate the vectorized equivalent against inherited recursion on this trip.
    check = np.unique(np.linspace(0, len(unit) - 1, min(16, len(unit))).astype(int))
    exact = np.array([hilbert_index(float(unit[i, 0]), float(unit[i, 1]), metadata["hilbert_eps"])
                      for i in check])
    if not np.allclose(native[check], exact, atol=1e-12, rtol=0):
        raise ValueError("Vectorized Hilbert differs from inherited recursion")
    compressed = np.asarray(duplicate(squeeze(native.tolist(), STEPS), STEPS), dtype=np.float64)
    if len(compressed) != STEPS:
        return {"id": str(ident), "label": int(label), "failure": "squeeze did not yield 224 values"}
    span = float(np.ptp(compressed))
    normalized = (compressed - compressed.min()) / span if span > 1e-12 else np.zeros_like(compressed)
    features = {"xy": sampled(unit).T, "native_h": np.stack([sampled(native), np.zeros(STEPS)]),
                "squeezed_h": np.stack([compressed, np.zeros(STEPS)]),
                "unit_h": np.stack([normalized, np.zeros(STEPS)])}
    return {"id": str(ident), "label": int(label), "features": features, "sequence": compressed,
            "audit": {"native_points": len(track), "duration_s": float(times[-1] - times[0]),
                      "clipped_points": clipped, "native_unique_cells": int(len(np.unique(native))),
                      "squeezed_unique_cells": int(len(np.unique(compressed))),
                      "constant": bool(span <= 1e-12), "squeezed_min": float(compressed.min()),
                      "squeezed_range": span}}


def select(rows, view):
    lengths = np.diff(rows["offset"])
    return np.arange(len(rows["ids"])) if view == "full" else np.flatnonzero(lengths >= 512)


def matching_indices(data, split, view):
    selected = {domain: select(data[domain, split], view) for domain in DOMAINS}
    counts = {domain: np.bincount(data[domain, split]["labels"][indices], minlength=6)
              for domain, indices in selected.items()}
    if view == "full":
        if not np.array_equal(counts["cnr"], counts["sumo"]):
            raise ValueError(f"Raw domains must already have matching class counts for {split}")
        return selected
    # Windows use the same six-class counts and deterministic IDs in each source.
    limits = np.minimum(counts["cnr"], counts["sumo"])
    if np.any(limits == 0):
        raise ValueError(f"A class lacks a continuous 512-point window in {split}")
    output = {}
    for domain in DOMAINS:
        labels = data[domain, split]["labels"]
        groups = []
        for label, count in enumerate(limits):
            eligible = selected[domain][labels[selected[domain]] == label]
            ranked = sorted(eligible, key=lambda i: hashlib.sha256(
                f"G29-window:{data[domain, split]['ids'][i]}".encode()).digest())
            groups.extend(ranked[:count])
        output[domain] = np.array(sorted(groups))
    return output


def summarize(audit):
    output = {"rows": len(audit)}
    keys = audit[0].keys()
    for key in keys:
        values = np.asarray([row[key] for row in audit], dtype=float)
        output[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                       "p95": float(np.quantile(values, .95)), "max": float(values.max())}
    return output


def write_case(out, rows, indices, view, metadata, workers):
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in indices:
        track = rows["txy"][rows["offset"][i]:rows["offset"][i+1]]
        if view == "window512":
            begin = (len(track) - 512) // 2
            track = track[begin:begin+512]
        jobs.append((rows["ids"][i], rows["labels"][i], track, metadata))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        encoded = list(pool.map(sequence_job, jobs, chunksize=8))
    failed = [row for row in encoded if "failure" in row]
    if failed:
        (out / "encoding-failed.json").write_text(json.dumps(failed, indent=2))
        raise RuntimeError(f"{len(failed)} encoding failures in {out}; no arm silently excludes them")
    sequences = np.stack([row["sequence"] for row in encoded])
    ids = np.asarray([row["id"] for row in encoded])
    labels = np.asarray([row["label"] for row in encoded], dtype=np.int64)
    rgb = np.lib.format.open_memmap(out / "images.tmp.npy", mode="w+", dtype=np.uint8,
                                   shape=(len(encoded), STEPS, STEPS, 3))
    float_error, byte_error, channel_clipped = [], [], 0
    lower = np.asarray(metadata["channel_min"])
    upper = np.asarray(metadata["channel_max"])
    for start in range(0, len(sequences), 32):
        seq = sequences[start:start+32]
        raw = raw_channels(seq)
        image, count = encode_frozen(seq, lower, upper, batch=32)
        channel_clipped += count
        rgb[start:start+len(seq)] = image
        norm = np.stack([row["features"]["unit_h"][0] for row in encoded[start:start+len(seq)]])
        inverse_float = np.sqrt(np.clip((np.diagonal(raw[..., 0], axis1=1, axis2=2)+1)/2, 0, 1))
        gasf_recovered = image[..., 0].astype(float)/255*(upper[0]-lower[0])+lower[0]
        inverse_byte = np.sqrt(np.clip((np.diagonal(gasf_recovered, axis1=1, axis2=2)+1)/2, 0, 1))
        float_error.extend(np.abs(inverse_float-norm).mean(1))
        byte_error.extend(np.abs(inverse_byte-norm).mean(1))
    rgb.flush()
    del rgb
    (out / "images.tmp.npy").replace(out / "images.npy")
    features = {key: np.asarray([row["features"][key] for row in encoded], dtype=np.float32)
                for key in encoded[0]["features"]}
    np.savez_compressed(out / "features.tmp.npz", ids=ids, labels=labels, **features)
    (out / "features.tmp.npz").replace(out / "features.npz")
    for row, float_loss, byte_loss in zip(encoded, float_error, byte_error):
        row["audit"].update(gasf_diagonal_float_mae=float(float_loss), gasf_diagonal_uint8_mae=float(byte_loss))
    report = {"class_counts": np.bincount(labels, minlength=6).tolist(),
              "all": summarize([row["audit"] for row in encoded]),
              "byclass": {str(c): summarize([row["audit"] for row in encoded if row["label"] == c])
                          for c in range(6)},
              "channel_clipped_values": channel_clipped,
              "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    print(str(out), len(ids), flush=True)
    return report


def main(args):
    metadata = json.loads(args.meta.read_text())
    for key in ("xy_min", "xy_max", "hilbert_eps", "channel_min", "channel_max"):
        if key not in metadata:
            raise ValueError(f"Missing frozen metadata field {key}")
    if not np.isclose(metadata["hilbert_eps"], .0009):
        raise ValueError("G29 requires original E1 grid64 eps=.0009")
    data = {(domain, split): read_raw(args.raw / f"{domain}-{split}.npz")
            for domain in SOURCES for split in SPLITS}
    for domain in SOURCES:
        if set(data[domain, "train"]["ids"]) & set(data[domain, "val"]["ids"]):
            raise ValueError("Train/validation participant overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    input_hashes = {f"{domain}-{split}": digest(args.raw / f"{domain}-{split}.npz")
                    for domain, split in data}
    manifest_path = args.output / "complete.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["raw_sha256"] != input_hashes or manifest["metadata_sha256"] != digest(args.meta):
            raise ValueError("Existing encoded cohort uses different inputs")
        return
    reports = {}
    for view in ("full", "window512"):
        for split in SPLITS:
            selected = matching_indices(data, split, view)
            for domain in DOMAINS:
                key = f"{domain}/{view}/{split}"
                reports[key] = write_case(args.output / key, data[domain, split], selected[domain],
                                          view, metadata, args.workers)
    for split in SPLITS:
        rows = data["sumo-duration", split]
        if not np.array_equal(np.bincount(rows["labels"], minlength=6),
                              np.bincount(data["cnr", split]["labels"], minlength=6)):
            raise ValueError("Duration-matched source must share CNR class counts")
        key = f"sumo-duration/full/{split}"
        reports[key] = write_case(args.output/key, rows, np.arange(len(rows["ids"])),
                                  "full", metadata, args.workers)
    manifest = {"raw_sha256": input_hashes, "metadata_sha256": digest(args.meta),
                "frozen_metadata": {key:metadata[key] for key in
                    ("xy_min", "xy_max", "hilbert_eps", "channel_min", "channel_max")},
                "hilbert_units": "inherited fractional cell index in [0,1), integer index divided by4096",
                "native_resampling": "linear interpolation in observation index to224; no explicit duration or speed",
                "one_dimensional_inputs": "all two channels; scalar Hilbert inputs have a zero second channel",
                "full_scope": "longest source continuous segment selected upstream, raw shared source counts",
                "window_scope": "centred512 native points, per-class matched source counts using fixed ID hash",
                "reports": reports}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("g29-data/raw"))
    parser.add_argument("--output", type=Path, default=Path("g29-data/encoded"))
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args())
