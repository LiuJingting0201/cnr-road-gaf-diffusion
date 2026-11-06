"""Lossless CSV export of already selected G33/G34 trajectories.

Only NumPy and the standard library are required. This module does not import
the generator, teachers or evaluator, and performs no selection or scoring.
Use separate new output roots for G33 and G34: duplicate seed/method slots in
one invocation are rejected instead of overwritten or silently renamed.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tempfile

import numpy as np


CLASSES = ["bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
METHODS = ("Uniform", "S", "V", "SV")
HEADER = ["time", "id", "x", "y", "type"]
FIELDS = {"ids", "request_ids", "labels", "offsets", "txy"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def integer(value, name, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError("Invalid integer in receipt: " + name)
    return value


def input_receipt(folder):
    path = folder / "complete.json"
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (receipt.get("complete") is not True or receipt.get("phase") != "terminal_gaf_selection"
            or receipt.get("classes") != CLASSES or receipt.get("selection_method") not in METHODS
            or receipt.get("family") != "diffusion" or receipt.get("arm") != "native_0_25s"
            or receipt.get("reference_data_opened") is not False):
        raise ValueError("Completed native five-class terminal selection receipt required: " + str(folder))
    seed = integer(receipt.get("seed"), "seed")
    selected = integer(receipt.get("generated_candidates"), "generated_candidates", 1)
    ready = integer(receipt.get("map_ready_conditions"), "map_ready_conditions", 1)
    requested = integer(receipt.get("requested_conditions"), "requested_conditions", 1)
    if not selected <= ready <= requested:
        raise ValueError("Inconsistent selected/request coverage")
    if receipt.get("requested_candidates_per_condition") != 8:
        raise ValueError("Expected the fixed K=8 selection receipt")
    artifact = receipt.get("artifacts", {}).get("samples.npz", {})
    if not isinstance(artifact.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
        raise ValueError("Missing or invalid samples.npz SHA256")
    integer(artifact.get("bytes"), "samples.npz bytes", 1)
    return {"folder": folder, "receipt": receipt, "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "seed": seed, "method": receipt["selection_method"],
            "filename": f"{receipt['arm']}-{seed}-{receipt['selection_method']}.csv.gz"}


def validate_arrays(data, expected_count):
    if set(data) != FIELDS:
        raise ValueError("Unexpected selected NPZ fields")
    ids, request_ids, labels, offsets, txy = (data[key] for key in ("ids", "request_ids", "labels", "offsets", "txy"))
    if (ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != expected_count
            or any(not str(ident) for ident in ids) or len(set(ids.tolist())) != len(ids)):
        raise ValueError("Trajectory IDs must be nonempty unique Unicode strings with the receipt count")
    if (request_ids.shape != ids.shape or request_ids.dtype.kind != "U"
            or any(not str(ident) for ident in request_ids) or len(set(request_ids.tolist())) != len(ids)):
        raise ValueError("Expected exactly one selected trajectory per unique request ID")
    if (labels.shape != ids.shape or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= len(CLASSES))):
        raise ValueError("Invalid participant labels")
    if txy.ndim != 2 or txy.shape[1] != 3 or txy.dtype.kind != "f" or txy.dtype.itemsize != 8:
        raise ValueError("Selected txy must be a float64 array with three columns")
    if not np.isfinite(txy).all():
        raise ValueError("Nonfinite selected time/coordinates")
    if offsets.ndim != 1 or offsets.shape != (len(ids) + 1,) or offsets.dtype.kind not in "iu":
        raise ValueError("Invalid trajectory offset shape or type")
    # Python integers avoid unsigned wraparound while validating interval bounds.
    positions = [int(value) for value in offsets]
    if positions[0] != 0 or positions[-1] != len(txy):
        raise ValueError("Offsets do not cover the entire txy array exactly")
    if any(a < 0 or b > len(txy) or b - a < 2 for a, b in zip(positions, positions[1:])):
        raise ValueError("Offset range crosses bounds, overlaps, or contains fewer than two points")
    for a, b in zip(positions, positions[1:]):
        times = txy[a:b, 0]
        if np.any(times[1:] <= times[:-1]):
            raise ValueError("Time must increase strictly within each trajectory")
    return positions


def load_verified(item):
    path = item["folder"] / "samples.npz"
    raw = path.read_bytes()
    spec = item["receipt"]["artifacts"]["samples.npz"]
    if len(raw) != spec["bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("Selected samples.npz size/hash does not match its receipt")
    # Hash and decode the same immutable byte snapshot, not two separate reads.
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate array names in selected NPZ archive")
        data = {key: archive[key].copy() for key in archive.files}
    positions = validate_arrays(data, item["receipt"]["generated_candidates"])
    return data, positions


def write_csv(path, data, positions):
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(HEADER)
                for i, (a, b) in enumerate(zip(positions, positions[1:])):
                    ident, kind = str(data["ids"][i]), CLASSES[int(data["labels"][i])]
                    for time_value, x_value, y_value in data["txy"][a:b]:
                        writer.writerow([format(float(time_value), ".17g"), ident,
                                         format(float(x_value), ".17g"), format(float(y_value), ".17g"), kind])


def verify_csv(path, data, positions):
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != HEADER:
            raise ValueError("CSV header differs from time,id,x,y,type")
        for i, (a, b) in enumerate(zip(positions, positions[1:])):
            ident, kind = str(data["ids"][i]), CLASSES[int(data["labels"][i])]
            for expected in data["txy"][a:b]:
                row = next(reader, None)
                if row is None or len(row) != 5 or row[1] != ident or row[4] != kind:
                    raise ValueError("CSV trajectory boundary, ID, type or row count changed")
                recovered = (float(row[0]), float(row[2]), float(row[3]))
                # IEEE-754 bit equality also checks the sign of zero.
                if struct.pack(">ddd", *recovered) != struct.pack(">ddd", *expected):
                    raise ValueError("CSV time/XY does not round-trip to identical float64 values")
                count += 1
        if next(reader, None) is not None:
            raise ValueError("Unexpected extra CSV rows")
    if count != len(data["txy"]):
        raise ValueError("CSV point count differs")
    return count


def export(inputs, output):
    folders = [Path(folder).resolve() for folder in inputs]
    output = Path(output).resolve()
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("At least one distinct selected input folder is required")
    if any(output == folder or folder in output.parents or output in folder.parents for folder in folders):
        raise ValueError("Output must be separate from every existing scientific input directory")
    # Failure to create a fresh directory must never write into an old export.
    output.mkdir(parents=True, exist_ok=False)
    exports, current = [], None
    try:
        items = []
        names = set()
        for folder in folders:
            current = str(folder)
            item = input_receipt(folder)
            if item["filename"] in names:
                raise ValueError("Duplicate seed/method output name; use separate output roots for G33 and G34")
            names.add(item["filename"])
            items.append(item)
        for item in items:
            current = str(item["folder"])
            data, positions = load_verified(item)
            partial = output / (item["filename"] + ".partial")
            final = output / item["filename"]
            write_csv(partial, data, positions)
            points = verify_csv(partial, data, positions)
            # Refuse completion if an input changed while this export was made.
            if (digest(item["folder"] / "complete.json") != item["receipt_sha256"]
                    or digest(item["folder"] / "samples.npz") != item["receipt"]["artifacts"]["samples.npz"]["sha256"]):
                raise ValueError("Scientific input changed during export")
            partial.replace(final)
            exports.append({"source_directory": current, "seed": item["seed"], "selection_method": item["method"],
                            "source_receipt_sha256": item["receipt_sha256"],
                            "source_samples_npz_sha256": item["receipt"]["artifacts"]["samples.npz"]["sha256"],
                            "requested_conditions": item["receipt"]["requested_conditions"],
                            "map_ready_conditions": item["receipt"]["map_ready_conditions"],
                            "trajectories": len(data["ids"]), "points": points,
                            "class_counts": dict(zip(CLASSES, np.bincount(data["labels"].astype(np.int64), minlength=5).tolist())),
                            "filename": final.name, "bytes": final.stat().st_size, "sha256": digest(final),
                            "streamed_readback": {"all_points_checked": points, "all_trajectories_checked": len(data["ids"]),
                                                  "float64_bitwise_equal": True, "ids_and_types_exact": True}})
        result = {"complete": True, "phase": "lossless_existing_selection_csv_export", "classes": CLASSES,
                  "schema": HEADER, "float_format": ".17g", "gzip_mtime": 0,
                  "selection_recomputed": False, "model_weights_opened": False, "reference_trajectories_opened": False,
                  "code_sha256": digest(__file__), "exports": exports,
                  "datasets": len(exports), "trajectories": sum(row["trajectories"] for row in exports),
                  "points": sum(row["points"] for row in exports)}
        save_json(output / "export-complete.json", result)
        return result
    except Exception as error:
        save_json(output / "export-failed.json", {"complete": False, "phase": "lossless_existing_selection_csv_export",
                  "input_being_processed": current, "error": repr(error), "completed_exports": exports,
                  "partial_files_retained": True, "code_sha256": digest(__file__)})
        raise


def self_test():
    checks = []
    with tempfile.TemporaryDirectory(prefix="g33-export-check-") as temporary:
        root = Path(temporary)
        arrays = {"ids": np.array(['轨迹,"quoted"\nA', "trip-B"]), "request_ids": np.array(["request-A", "request-B"]),
                  "labels": np.array([0, 4]), "offsets": np.array([0, 3, 6]),
                  "txy": np.array([[0., -0., np.nextafter(0., 1.)], [.25, np.pi, -1e200], [.5000000000000001, 1e-200, 3.],
                                   [0., 4., 5.], [.25, 5., 6.], [.37, 6., 7.]], dtype=np.float64)}

        def fixture(name, edit=None):
            folder = root / name
            folder.mkdir()
            value = {key: a.copy() for key, a in arrays.items()}
            if edit:
                edit(value)
            np.savez_compressed(folder / "samples.npz", **value)
            spec = {"sha256": digest(folder / "samples.npz"), "bytes": (folder / "samples.npz").stat().st_size}
            receipt = {"complete": True, "phase": "terminal_gaf_selection", "classes": CLASSES,
                       "selection_method": "Uniform", "family": "diffusion", "arm": "native_0_25s", "seed": 42,
                       "reference_data_opened": False, "generated_candidates": 2, "map_ready_conditions": 2,
                       "requested_conditions": 3, "requested_candidates_per_condition": 8, "artifacts": {"samples.npz": spec}}
            (folder / "complete.json").write_text(json.dumps(receipt))
            return folder

        valid = fixture("valid")
        result = export([valid], root / "ok")
        assert result["points"] == 6 and result["trajectories"] == 2
        checks.append("float64_extremes_negative_zero_quoted_unicode_csv_and_fractional_endpoint_roundtrip")
        try:
            export([valid], root / "ok")
        except FileExistsError:
            checks.append("existing_output_refused_without_overwrite")
        else:
            raise AssertionError("Existing output accepted")
        malformed = [
            ("time_not_increasing", lambda a: a["txy"].__setitem__((1, 0), 0.)),
            ("offset_crosses_time_reset", lambda a: a["offsets"].__setitem__(1, 4)),
            ("offset_out_of_bounds", lambda a: a["offsets"].__setitem__(-1, 7)),
            ("duplicate_ids", lambda a: a["ids"].__setitem__(1, a["ids"][0])),
            ("illegal_class", lambda a: a["labels"].__setitem__(1, 5)),
            ("nonfinite_xy", lambda a: a["txy"].__setitem__((0, 1), np.inf)),
        ]
        for name, edit in malformed:
            folder = fixture(name, edit)
            try:
                export([folder], root / (name + "-out"))
            except ValueError:
                assert (root / (name + "-out") / "export-failed.json").is_file()
                assert not (root / (name + "-out") / "export-complete.json").exists()
                checks.append(name + "_rejected_and_failure_recorded")
            else:
                raise AssertionError("Malformed data accepted: " + name)
        corrupt = fixture("bad_hash")
        with (corrupt / "samples.npz").open("ab") as stream:
            stream.write(b"corrupt")
        try:
            export([corrupt], root / "bad-hash-out")
        except ValueError:
            checks.append("tampered_npz_rejected")
        else:
            raise AssertionError("Tampered hash accepted")
        duplicate_slot = fixture("duplicate_slot")
        try:
            export([valid, duplicate_slot], root / "duplicate-slot-out")
        except ValueError:
            checks.append("duplicate_seed_method_rejected")
        else:
            raise AssertionError("Duplicate output slot accepted")
        changed = root / "changed.csv.gz"
        with gzip.open(root / "ok/native_0_25s-42-Uniform.csv.gz", "rt", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        rows[2][1] = "wrong-trajectory"
        with gzip.open(changed, "wt", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        try:
            verify_csv(changed, arrays, [0, 3, 6])
        except ValueError:
            checks.append("streamed_readback_detects_wrong_trajectory_id")
        else:
            raise AssertionError("Corrupt CSV accepted")
    return {"complete": True, "construct_checks": checks, "code_sha256": digest(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.inputs or args.output:
            parser.error("--self-test cannot be combined with export paths")
        print(json.dumps(self_test()))
    else:
        if not args.inputs or args.output is None:
            parser.error("--inputs and --output are required for export")
        result = export(args.inputs, args.output)
        print(json.dumps({key: result[key] for key in ("complete", "datasets", "trajectories", "points")}))


if __name__ == "__main__":
    main()
