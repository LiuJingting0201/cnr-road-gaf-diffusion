"""Read-only G29 train/validation audit of inherited squeeze and mirror padding.

Reads the exact encoded cohort IDs and compares native Hilbert runs before and
after the inherited squeeze, then verifies the saved final 224-point sequence.
No model fitting, image changes, or held-out test access occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cnr.original_gaf import duplicate, squeeze
from cnr.representations import hilbert_numpy
from g29_encode import read_raw


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runs(values):
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
    return values[starts], np.diff(np.r_[starts, len(values)])


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def summarize(rows):
    result = {"rows": len(rows)}
    counts = ("native_below_224", "native_equal_224", "native_above_224",
              "mirror_triggered", "native_runs_above_224", "run_order_preserved",
              "unique_cell_support_preserved", "final_saved_sequence_exact")
    for key in counts:
        count = sum(bool(row[key]) for row in rows)
        result[key] = {"count": count, "fraction": count / len(rows)}
    for key in ("native_points", "squeezed_points", "native_runs", "squeezed_runs",
                "removed_point_fraction", "mirrored_output_fraction",
                "dwell_fraction_total_variation"):
        result[key] = stats([row[key] for row in rows])
    return result


def audit_case(raw, features_path, view, metadata):
    with np.load(features_path) as features:
        identifiers, labels = features["ids"], features["labels"]
        saved = features["squeezed_h"][:, 0]
    positions = {str(ident): i for i, ident in enumerate(raw["ids"])}
    low, high = np.asarray(metadata["xy_min"]), np.asarray(metadata["xy_max"])
    rows = []
    for ident, label, final_saved in zip(identifiers, labels, saved):
        index = positions[str(ident)]
        if int(raw["labels"][index]) != label:
            raise ValueError("Saved and raw cohort labels differ")
        track = raw["txy"][raw["offset"][index]:raw["offset"][index+1]]
        if view == "window512":
            if len(track) < 512:
                raise ValueError("Window cohort contains a short trajectory")
            start = (len(track) - 512) // 2
            track = track[start:start+512]
        native = hilbert_numpy(((track[:, 1:] - low) / (high-low)).clip(0, 1),
                               metadata["hilbert_eps"])
        compressed = np.asarray(squeeze(native.tolist(), 224))
        final = np.asarray(duplicate(compressed.tolist(), 224))
        if len(final) != 224:
            raise ValueError("Inherited squeeze failed for an already encoded trajectory")
        native_values, native_lengths = runs(native)
        compressed_values, compressed_lengths = runs(compressed)
        preserved = np.array_equal(native_values, compressed_values)
        if not preserved:
            raise ValueError("Inherited squeeze changed the sequence of visited runs")
        saved_exact = np.array_equal(final.astype(np.float32), final_saved)
        if not saved_exact:
            raise ValueError("Recomputed squeeze/mirror differs from saved G29 features")
        rows.append({"label": int(label), "native_points": len(native),
                     "squeezed_points": len(compressed),
                     "native_below_224": len(native) < 224,
                     "native_equal_224": len(native) == 224,
                     "native_above_224": len(native) > 224,
                     "mirror_triggered": len(compressed) < 224,
                     "native_runs_above_224": len(native_values) > 224,
                     "native_runs": len(native_values),
                     "squeezed_runs": len(compressed_values),
                     "run_order_preserved": preserved,
                     "unique_cell_support_preserved": np.array_equal(
                         np.unique(native), np.unique(compressed)),
                     "final_saved_sequence_exact": saved_exact,
                     "removed_point_fraction": 1-len(compressed)/len(native),
                     "mirrored_output_fraction": (224-len(compressed))/224,
                     "dwell_fraction_total_variation": float(.5*np.abs(
                         native_lengths/len(native)-compressed_lengths/len(compressed)).sum())})
    return {"all": summarize(rows),
            "byclass": {str(label): summarize([row for row in rows if row["label"] == label])
                        for label in range(6)},
            "ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest()}


def main(args):
    manifest_path = args.encoded / "complete.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest["frozen_metadata"]
    reports = {}
    for source in ("cnr", "sumo", "sumo-duration"):
        for split in ("train", "val"):
            raw_path = args.raw / f"{source}-{split}.npz"
            if digest(raw_path) != manifest["raw_sha256"][f"{source}-{split}"]:
                raise ValueError("Raw data differs from the encoded cohort manifest")
            raw = read_raw(raw_path)
            for view in (("full", "window512") if source != "sumo-duration" else ("full",)):
                key = f"{source}/{view}/{split}"
                report = audit_case(raw, args.encoded / key / "features.npz", view, metadata)
                if report["ids_sha256"] != manifest["reports"][key]["ids_sha256"]:
                    raise ValueError("Audit cohort differs from the encoding manifest")
                reports[key] = report
                print(key, report["all"]["rows"], flush=True)
    result = {"scope": "Existing matched G29 train/validation cohorts only; no training or test access",
              "encoding_manifest_sha256": digest(manifest_path),
              "script_sha256": digest(Path(__file__)),
              "algorithm": "For inputs longer than 224, inherited ceil_list rescales every identical-value run by its relative length, rounding up; squeeze initially orders runs by decreasing length, then cycles through this queue removing repeated positions until length 224 while retaining at least one point per run. Short sequences are then mirrored and truncated to 224.",
              "dwell_fraction_total_variation": "Half L1 distance between native and squeezed per-run occupancy fractions BEFORE mirror; zero means proportional dwell preserved exactly, not raw duration retained.",
              "interpretation": "This measures encoding transformations, not label mutual information or generative utility. Squeeze retains ordered cell runs on successful cases but changes their dwell allocation; mirror introduces reversed runs. No timestamps or original point counts are encoded in the final image.",
              "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("g29-data/raw"))
    parser.add_argument("--encoded", type=Path, default=Path("g29-data/encoded"))
    parser.add_argument("--output", type=Path, default=Path("g29-data/encoding-detail.json"))
    main(parser.parse_args())
