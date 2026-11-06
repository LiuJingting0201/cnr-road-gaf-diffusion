"""Direct SUMO full-leg supervision for G33; no handmade-map admission gate.

Observed training paths are paired conditioning inputs for learning motion given
a route. Observed validation paths may only support explicitly labelled oracle
conditional diagnostics. They are NOT available to a prospective generation
benchmark. Native time, zero motion intervals and all accepted vehicle legs are
retained. No fitting, teacher scores or new simulation occur here.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


CLASSES = ["bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
SOURCE_CLASSES = ["pedestrian", *CLASSES]
SCHEMA = "g33-direct-sumo-full-legs-v1"
DT = .25
ROUTE_POINTS = 129


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def exact_decode(path_xy, path_arc, progress):
    """Interpolate on retained observed vertices, never a 2049-node shortcut."""
    p = np.asarray(progress, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Progress must be finite and in [0,1]")
    if path_arc[-1] == 0:
        return np.repeat(path_xy[:1], len(p), axis=0)
    return np.stack([np.interp(p * path_arc[-1], path_arc, v)
                     for v in path_xy.T], axis=1)


def encode(txy):
    txy = np.asarray(txy, dtype=np.float64)
    if txy.ndim != 2 or txy.shape[1] != 3 or len(txy) < 2 or not np.isfinite(txy).all():
        raise ValueError("Finite observed t,x,y sequence with at least two points required")
    time = txy[:, 0] - txy[0, 0]
    xy = txy[:, 1:]
    if not np.allclose(np.diff(time), DT, rtol=0, atol=1e-7):
        raise ValueError("Unexpected native time gaps")
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(distance / DT > 100.):
        raise ValueError("Input violates the existing G31 jump criterion")
    arc = np.r_[0., np.cumsum(distance)]
    length, duration = float(arc[-1]), float(time[-1])
    if length > 0:
        keep = np.r_[True, np.diff(arc) > 0]
        path_xy, path_arc = xy[keep], arc[keep]
        progress = arc / length
    else:
        # No arbitrary motion is added to a stationary coordinate sequence.
        path_xy, path_arc = xy[[0, -1]], np.array([0., 0.])
        progress = time / duration
    conditioning = exact_decode(path_xy, path_arc, np.linspace(0., 1., ROUTE_POINTS)).astype(np.float32)
    legacy = np.interp(np.linspace(0., duration, 257), time, progress)
    recovered = exact_decode(path_xy, path_arc, progress)
    error = float(np.max(np.linalg.norm(recovered - xy, axis=1)))
    if error > 1e-7:
        raise ValueError("Exact-vertex normalized-progress reconstruction failed")
    return {"time": time, "progress": progress, "path_xy": path_xy,
            "path_arc": path_arc, "paths": conditioning, "legacy_progress": legacy,
            "length": length, "duration": duration,
            "zero_motion_intervals": int(np.count_nonzero(distance == 0.)),
            "max_roundtrip_position_error_m": error}


def load_source(folder):
    receipt_path = folder / "complete.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("complete") is not True or receipt.get("classes") != SOURCE_CLASSES:
        raise ValueError("Completed known G31 source receipt required")
    for name in ("legs.npz", "leg-metadata.json.gz"):
        if digest(folder / name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("Immutable G31 source artifact changed: " + name)
    with np.load(folder / "legs.npz", allow_pickle=False) as source:
        required = ("ids", "parent_base_ids", "labels", "partitions", "offset", "txy",
                    "event_start", "event_end", "ordinals")
        arrays = {key: source[key] for key in required}
    with gzip.open(folder / "leg-metadata.json.gz", "rt") as stream:
        metadata = json.load(stream)["accepted"]
    n = len(metadata)
    if n != receipt["accepted_legs"] or len(set(arrays["ids"].tolist())) != n:
        raise ValueError("Source identity counts are inconsistent")
    if any(len(arrays[key]) != n for key in required if key not in ("offset", "txy")):
        raise ValueError("Source record arrays are misaligned")
    offset = arrays["offset"]
    if (offset.shape != (n+1,) or offset[0] != 0 or offset[-1] != len(arrays["txy"])
            or np.any(np.diff(offset) < 2)):
        raise ValueError("Invalid source trajectory offsets")
    parent_split = {}
    for i, meta in enumerate(metadata):
        parent, split, label = str(arrays["parent_base_ids"][i]), str(arrays["partitions"][i]), int(arrays["labels"][i])
        if (split not in ("train", "val") or parent_split.get(parent, split) != split
                or not 0 <= label < len(SOURCE_CLASSES)
                or meta["leg_id"] != str(arrays["ids"][i])
                or meta["parent_base_id"] != parent or meta["partition"] != split
                or int(meta["label"]) != label or int(meta["ordinal"]) != int(arrays["ordinals"][i])):
            raise ValueError("Source identities, labels or inherited parent partition mismatch")
        parent_split[parent] = split
    return receipt, arrays, metadata


def summary(a, indices, encoded):
    indices = list(indices)
    selected = [encoded[i] for i in indices]
    counts = Counter(int(a["labels"][i]) for i in indices)
    def quantiles(values):
        v = np.asarray(list(values), dtype=float)
        return {"n": len(v), **({"min": float(v.min()), "median": float(np.median(v)),
                "mean": float(v.mean()), "p95": float(np.quantile(v, .95)), "max": float(v.max())} if len(v) else {})}
    return {"legs": len(indices), "parents": len(set(str(a["parent_ids"][i]) for i in indices)),
            "class_counts": {name: counts[c] for c, name in enumerate(CLASSES)},
            "duration_s": quantiles(x["duration"] for x in selected),
            "arc_length_m": quantiles(x["length"] for x in selected),
            "native_points": quantiles(len(x["time"]) for x in selected),
            "zero_motion_intervals": sum(x["zero_motion_intervals"] for x in selected),
            "constant_legs": sum(x["length"] == 0 for x in selected),
            "max_exact_vertex_roundtrip_position_error_m": max((x["max_roundtrip_position_error_m"] for x in selected), default=0.)}


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("New prepared output must be separate from the immutable input")
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipt, src, metadata = load_source(source)
        selected = np.flatnonzero(src["labels"] > 0)
        encoded = []
        for source_index in selected:
            start, end = src["offset"][source_index:source_index+2]
            txy = src["txy"][start:end]
            if (abs(float(txy[0, 0]) - float(src["event_start"][source_index])) > DT+1e-7
                    or not -1e-7 <= float(src["event_end"][source_index] - txy[-1, 0]) <= DT+1e-7):
                raise ValueError("Accepted source event/sample support changed")
            encoded.append(encode(txy))
        arrays = {"ids": src["ids"][selected], "parent_ids": src["parent_base_ids"][selected],
                  "partition": src["partitions"][selected], "labels": (src["labels"][selected]-1).astype(np.int64),
                  "paths": np.stack([r["paths"] for r in encoded]),
                  "length": np.asarray([r["length"] for r in encoded], dtype=np.float64),
                  "duration": np.asarray([r["duration"] for r in encoded], dtype=np.float64),
                  "legacy_progress": np.stack([r["legacy_progress"] for r in encoded]),
                  "offset": np.r_[0, np.cumsum([len(r["time"]) for r in encoded])].astype(np.int64),
                  "path_offset": np.r_[0, np.cumsum([len(r["path_arc"]) for r in encoded])].astype(np.int64)}
        arrays.update({key: np.concatenate([r[key] for r in encoded])
                       for key in ("time", "progress", "path_xy", "path_arc")})
        np.savez_compressed(output / "dataset.npz", **arrays)
        groups = {split: summary(arrays, np.flatnonzero(arrays["partition"] == split), encoded)
                  for split in ("train", "val")}
        groups["all"] = summary(arrays, range(len(selected)), encoded)
        report = {"complete": True, "schema": SCHEMA, "classes": CLASSES,
                  "scope": "Direct SUMO complete observed vehicle legs for conditional motion learning; no map-constructor filtering",
                  "formal_generation_evaluation_ready": False,
                  "source_legs": len(src["ids"]), "vehicle_legs_retained": len(selected),
                  "pedestrian_legs_excluded_by_predeclared_scope": int(np.sum(src["labels"] == 0)),
                  "vehicle_legs_excluded_by_geometry_or_duration": 0,
                  "groups": groups,
                  "checks": {"source_artifact_hashes": True, "source_metadata_alignment": True,
                             "parent_train_val_disjoint": True, "source_test_absent": True,
                             "all_accepted_vehicle_legs_retained": True,
                             "native_time_and_zero_motion_intervals_retained": True,
                             "exact_vertex_roundtrip_below_1e_minus_7_m": True},
                  "representation": {"native_step_s": DT, "route_condition_points": ROUTE_POINTS,
                    "conditioning_geometry": "129 equal-arc observed points, float32, paired supervision only",
                    "native_target": "float64 exact normalized cumulative observed arc at original .25s observations",
                    "legacy_target": "257 equal-relative-time controls from the same complete observations",
                    "decoder": "all ordered nonduplicate observed vertices with cumulative arc, float64; stationary path repeats endpoint",
                    "timing": "Observed span, no invented coordinate at an unobserved event endpoint",
                    "normalization": "No fitted normalization or duration distribution here; model statistics must use train only"},
                  "sources": {"G31_receipt_sha256": digest(source / "complete.json"),
                              "legs_sha256": digest(source / "legs.npz"),
                              "metadata_sha256": digest(source / "leg-metadata.json.gz"),
                              "script_sha256": digest(Path(__file__))},
                  "artifacts": {"dataset.npz": {"sha256": digest(output / "dataset.npz"),
                                               "bytes": (output / "dataset.npz").stat().st_size}},
                  "limits": ["One existing pilot episode; same-map public demand and inherited parent partition, not unseen-city/line coverage.",
                             "Observed train route geometry is supervision. Observed val route and true duration may only be labelled oracle conditional diagnostics.",
                             "Prospective generation must receive only independently available map/route requests and sample duration from train-side model.",
                             "These are source simulator observations, not physical road measurements and not outputs of the diffusion model.",
                             "New five-class full-leg GAF readers require matched training; G30 six-class segment accuracy is not their accuracy."]}
        save(output / "complete.json", report)
        print(json.dumps({"complete": True, "classes": CLASSES, "split_counts": {s: groups[s]["legs"] for s in ("train", "val")}}), flush=True)
        return report
    except Exception as error:
        save(output / "failed.json", {"complete": False, "error": repr(error)})
        raise


def self_test():
    t = np.arange(41) * DT
    x = np.r_[np.zeros(9), np.arange(1, 33) * .5]
    txy = np.c_[100.+t, x, np.sin(x)]
    r = encode(txy)
    assert len(r["time"]) == len(txy) and r["zero_motion_intervals"] == 8
    assert np.all(r["progress"][:9] == 0) and r["progress"][-1] == 1
    assert np.array_equal(r["time"], t)
    assert r["legacy_progress"].shape == (257,) and r["paths"].shape == (129, 2)
    assert r["max_roundtrip_position_error_m"] < 1e-10
    curved = np.c_[t, 100.*np.sin(t/5), 100.*(1.-np.cos(t/5))]
    r = encode(curved)
    assert len(r["path_xy"]) == len(t) and r["max_roundtrip_position_error_m"] < 1e-10
    constant = np.c_[t[:4], np.ones(4)*7, np.ones(4)*9]
    r = encode(constant)
    assert r["length"] == 0 and r["zero_motion_intervals"] == 3
    assert r["max_roundtrip_position_error_m"] == 0
    for invalid in (curved[[0, 2, 3]], np.array([[0., 0., 0.], [.25, 30., 0.]])):
        try:
            encode(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid observation accepted")
    print("g33_prepare_construct_checks_passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../g31-job/g31-data/legs/pilot-300001"))
    parser.add_argument("--output", type=Path, default=Path("g33-data/direct-legs"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        prepare(args.source, args.output)
