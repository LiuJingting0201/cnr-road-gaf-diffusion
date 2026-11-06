"""Merge the three predeclared full-day MoST episodes without counting reruns twice.

This creates source-simulator supervision, not diffusion-generated trajectories.
The numeric layout is compatible with G33; provenance and readiness are G35.
No fitting, test-set access or automatic additional simulation occurs here.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from g33_prepare import CLASSES, DT, ROUTE_POINTS, SCHEMA, digest, save


EPISODES = {350101: "train", 350102: "train", 350201: "val"}
TRAIN_SEEDS = (350101, 350102)
MIN_TRAIN_PARENTS = 10000
ROW_KEYS = ("ids", "parent_ids", "partition", "labels", "paths", "length",
            "duration", "legacy_progress", "episodes", "ordinals", "event_start",
            "event_end", "lines", "source_route_hashes", "source_parent_route_hashes",
            "realized_route_hashes")
POINT_KEYS = ("time", "progress", "xy", "native_speed")
PATH_KEYS = ("path_xy", "path_arc")
ARRAY_KEYS = set(ROW_KEYS + POINT_KEYS + PATH_KEYS + ("offset", "path_offset"))


def assigned_seed(parent):
    raw = hashlib.sha256(("g35-realization-v1:" + str(parent)).encode()).digest()
    return TRAIN_SEEDS[int.from_bytes(raw, "big") % len(TRAIN_SEEDS)]


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), **({"min": float(values.min()),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}
            if values.size else {})}


def validate_arrays(arrays, seed, split):
    """Check the persisted ragged layout before rebasing any offset."""
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Unexpected prepared array inventory")
    if EPISODES.get(seed) != split:
        raise ValueError("Unregistered episode seed or split")
    n = len(arrays["ids"])
    if any(len(arrays[key]) != n for key in ROW_KEYS):
        raise ValueError("Misaligned per-leg arrays")
    if (arrays["paths"].shape != (n, ROUTE_POINTS, 2)
            or arrays["legacy_progress"].shape != (n, 257)
            or arrays["path_xy"].shape != (len(arrays["path_arc"]), 2)
            or arrays["time"].shape != arrays["progress"].shape
            or arrays["xy"].shape != (len(arrays["time"]), 2)
            or arrays["native_speed"].shape != arrays["time"].shape):
        raise ValueError("Unexpected prepared geometry or time shape")
    for name, total in (("offset", len(arrays["time"])),
                        ("path_offset", len(arrays["path_arc"]))):
        offset = arrays[name]
        if (offset.dtype.kind not in "iu" or offset.shape != (n + 1,)
                or offset[0] != 0 or offset[-1] != total
                or np.any(np.diff(offset) < 2)):
            raise ValueError("Invalid ragged boundary: " + name)
    if (len(set(arrays["ids"].astype(str))) != n
            or any(str(x) != split for x in arrays["partition"])
            or arrays["episodes"].dtype.kind not in "iu" or np.any(arrays["episodes"] != seed)
            or arrays["ordinals"].dtype.kind not in "iu" or np.any(arrays["ordinals"] < 0)
            or arrays["labels"].dtype.kind not in "iu"
            or np.any(arrays["labels"] < 0) or np.any(arrays["labels"] >= len(CLASSES))):
        raise ValueError("Invalid identities, partition or five-class labels")
    if any(not np.isfinite(arrays[key]).all() for key in
           ("paths", "length", "duration", "legacy_progress", "event_start", "event_end",
            "time", "progress", "xy", *PATH_KEYS)):
        raise ValueError("Nonfinite prepared target")
    event_span = arrays["event_end"] - arrays["event_start"]
    if (np.any(arrays["length"] < 0) or np.any(arrays["duration"] <= 0)
            or np.any(event_span - arrays["duration"] < -1e-7)
            or np.any(event_span - arrays["duration"] > 2 * DT + 1e-7)):
        raise ValueError("Invalid route length or observed duration")
    labels_by_parent = {}
    for i, raw_parent in enumerate(arrays["parent_ids"]):
        parent, label = str(raw_parent), int(arrays["labels"][i])
        if not parent or labels_by_parent.get(parent, label) != label:
            raise ValueError("Empty parent identity or inconsistent parent class")
        labels_by_parent[parent] = label
        if split == "train" and assigned_seed(parent) != seed:
            raise ValueError("Parent did not use its preassigned realization")
        a, b = arrays["offset"][i:i + 2]
        c, d = arrays["path_offset"][i:i + 2]
        time, progress = arrays["time"][a:b], arrays["progress"][a:b]
        arc = arrays["path_arc"][c:d]
        if (abs(float(time[0])) > 1e-7
                or not np.allclose(np.diff(time), DT, atol=1e-7, rtol=0)
                or abs(float(time[-1] - arrays["duration"][i])) > 1e-7
                or np.any(np.diff(progress) < -1e-12)
                or np.any(progress < -1e-12) or np.any(progress > 1 + 1e-12)
                or abs(float(progress[0])) > 1e-12 or abs(float(progress[-1] - 1)) > 1e-12
                or abs(float(arc[0])) > 1e-7 or np.any(np.diff(arc) < 0)
                or not np.isclose(arc[-1], arrays["length"][i], rtol=1e-12, atol=1e-7)):
            raise ValueError("Native clock, progress or path arc contract failed")


def combine_arrays(episodes):
    """Concatenate whole legs; an episode's first offset is omitted after rebasing."""
    if set(episodes) != set(EPISODES):
        raise ValueError("Exactly two fixed train episodes and one val episode are required")
    ordered = [episodes[seed] for seed in EPISODES]
    parents, all_ids = {}, set()
    for seed, arrays in zip(EPISODES, ordered):
        split = EPISODES[seed]
        validate_arrays(arrays, seed, split)
        ids = set(arrays["ids"].astype(str))
        if ids & all_ids:
            raise ValueError("A leg identifier appears in multiple episodes")
        all_ids.update(ids)
        for parent in set(arrays["parent_ids"].astype(str)):
            if parent in parents:
                raise ValueError("A parent appears in multiple episodes or partitions")
            parents[parent] = (seed, split)
    merged = {key: np.concatenate([a[key] for a in ordered], axis=0)
              for key in (*ROW_KEYS, *POINT_KEYS, *PATH_KEYS)}
    for key in ("offset", "path_offset"):
        pieces, total = [np.asarray([0], dtype=np.int64)], 0
        for arrays in ordered:
            pieces.append(arrays[key][1:] + total)
            total += int(arrays[key][-1])
        merged[key] = np.concatenate(pieces).astype(np.int64, copy=False)
    return merged


def _route_info(row):
    source = row.get("source_plan", {})
    route_hash = row.get("source_parent_route_hash") or source.get("parent_edges_hash") or source.get("edges_hash")
    if not route_hash and source.get("edges"):
        route_hash = hashlib.sha256(" ".join(source["edges"]).encode()).hexdigest()
    line = row.get("line") or source.get("source_attributes", {}).get("line")
    return route_hash, line


def summarize(arrays, indices, metadata):
    indices = np.asarray(indices, dtype=np.int64)
    counts = Counter(int(x) for x in arrays["labels"][indices])
    parents_by_class = {name: len(set(arrays["parent_ids"][indices[
        arrays["labels"][indices] == c]].astype(str))) for c, name in enumerate(CLASSES)}
    routes = [_route_info(metadata[int(i)]) for i in indices]
    stops = sum(int(b - a - 1) if arrays["length"][i] == 0
                else int(np.count_nonzero(np.diff(arrays["progress"][a:b]) == 0))
                for i in indices for a, b in [arrays["offset"][i:i + 2]])
    return {"legs": int(len(indices)),
            "parents": len(set(arrays["parent_ids"][indices].astype(str))),
            "class_counts": {name: counts[c] for c, name in enumerate(CLASSES)},
            "class_parent_counts": parents_by_class,
            "duration_s": quantiles(arrays["duration"][indices]),
            "arc_length_m": quantiles(arrays["length"][indices]),
            "native_points": quantiles(np.diff(arrays["offset"])[indices]),
            "total_native_points": int(np.diff(arrays["offset"])[indices].sum()),
            "zero_motion_intervals": stops,
            "constant_legs": int(np.count_nonzero(arrays["length"][indices] == 0)),
            "source_parent_route_hashes": len({r for r, _ in routes if r}),
            "legs_with_source_parent_route_hash": sum(bool(r) for r, _ in routes),
            "source_bus_lines": len({line for i, (_, line) in zip(indices, routes)
                                     if int(arrays["labels"][i]) == 0 and line}),
            "route_count_scope": "Source parent route hashes; not necessarily individual activity-leg routes or unseen-route coverage"}


def load_episode(folder):
    folder = Path(folder).resolve()
    receipt = json.loads((folder / "complete.json").read_text())
    if ((folder / "failed.json").exists() or receipt.get("complete") is not True
            or receipt.get("schema") != SCHEMA or receipt.get("classes") != CLASSES
            or receipt.get("provenance_schema") != "g35-native-full-day-extraction-v1"):
        raise ValueError("Completed G35 prepared episode required")
    seed, split = receipt.get("episode"), receipt.get("split")
    if EPISODES.get(seed) != split:
        raise ValueError("Unexpected prepared episode or partition")
    expected_checks = ("old_test_templates_excluded_by_split_metadata_only",
                       "parent_episode_preassignment_before_FCD", "source_and_simulation_hashes_verified",
                       "native_time_and_zero_motion_intervals_retained",
                       "all_event_and_sampling_accepted_legs_retained", "exact_vertex_roundtrip_below_1e_minus_7_m")
    if any(receipt.get("checks", {}).get(name) is not True for name in expected_checks):
        raise ValueError("Missing extraction isolation checks")
    for name in ("dataset.npz", "leg-metadata.json.gz"):
        path = folder / name
        item = receipt.get("artifacts", {}).get(name, {})
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != digest(path):
            raise ValueError("Prepared artifact changed: " + name)
    with np.load(folder / "dataset.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_arrays(arrays, seed, split)
    with gzip.open(folder / "leg-metadata.json.gz", "rt") as stream:
        metadata = json.load(stream)["accepted"]
    if len(metadata) != len(arrays["ids"]):
        raise ValueError("Metadata count does not match prepared legs")
    for i, row in enumerate(metadata):
        if (row.get("leg_id") != str(arrays["ids"][i])
                or row.get("parent_base_id") != str(arrays["parent_ids"][i])
                or row.get("partition") != split
                or row.get("label") != int(arrays["labels"][i])
                or row.get("episode") != seed or row.get("ordinal") != int(arrays["ordinals"][i])
                or row.get("event_start") != float(arrays["event_start"][i])
                or row.get("event_end") != float(arrays["event_end"][i])
                or any(row.get(meta_key, "") != str(arrays[array_key][i]) for meta_key, array_key in
                       (("line", "lines"), ("source_route_hash", "source_route_hashes"),
                        ("source_parent_route_hash", "source_parent_route_hashes"),
                        ("realized_route_hash", "realized_route_hashes")))):
            raise ValueError("Prepared metadata and target arrays are misaligned")
    return seed, arrays, metadata, receipt


def merge(inputs, output):
    inputs, output = [Path(x).resolve() for x in inputs], Path(output).resolve()
    if len(inputs) != len(EPISODES) or len(set(inputs)) != len(inputs):
        raise ValueError("Exactly three distinct input directories are required")
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError("Merge output must be separate from each immutable source")
    output.mkdir(parents=True, exist_ok=False)
    try:
        episodes, metadata_by_seed, sources = {}, {}, {}
        for folder in inputs:
            seed, arrays, metadata, receipt = load_episode(folder)
            if seed in episodes:
                raise ValueError("Repeated seed with a different directory")
            episodes[seed], metadata_by_seed[seed] = arrays, metadata
            sources[str(seed)] = {"complete_sha256": digest(folder / "complete.json"),
                                 "artifacts": receipt["artifacts"], "source_provenance": receipt.get("sources", {}),
                                 "split": receipt["split"]}
        for field in ("source_manifest_sha256", "original_split_manifest_sha256"):
            values = {item["source_provenance"].get(field) for item in sources.values()}
            if len(values) != 1 or None in values:
                raise ValueError("Episodes do not share the same source and parent split: " + field)
        arrays = combine_arrays(episodes)
        metadata = [row for seed in EPISODES for row in metadata_by_seed[seed]]
        groups = {split: summarize(arrays, np.flatnonzero(arrays["partition"] == split), metadata)
                  for split in ("train", "val")}
        groups["all"] = summarize(arrays, range(len(arrays["ids"])), metadata)
        reasons = []
        if groups["train"]["parents"] < MIN_TRAIN_PARENTS:
            reasons.append("fewer_than_10000_unique_training_parents")
        for split in ("train", "val"):
            if any(value == 0 for value in groups[split]["class_counts"].values()):
                reasons.append(split + "_does_not_cover_all_five_classes")
        if groups["all"]["constant_legs"]:
            reasons.append("constant_legs_preserved_but_current_g33_model_requires_positive_route_length")
        np.savez_compressed(output / "dataset.npz", **arrays)
        with gzip.open(output / "leg-metadata.json.gz", "wt", encoding="utf-8") as stream:
            json.dump({"accepted": metadata}, stream, ensure_ascii=False, allow_nan=False)
        report = {"complete": True, "schema": SCHEMA, "provenance_schema": "g35-full-day-merged-v1",
                  "classes": CLASSES, "training_ready": not reasons,
                  "training_readiness_reasons": reasons, "training_started": False,
                  "formal_generation_evaluation_ready": False,
                  "scope": "Two train and one development full-day native MoST episodes; one preassigned realization per training parent, all eligible complete legs retained",
                  "groups": groups, "episode_partitions": {str(k): v for k, v in EPISODES.items()},
                  "checks": {"source_artifact_hashes": True, "source_metadata_alignment": True,
                             "parent_train_val_disjoint": True, "parent_episode_disjoint": True,
                             "source_test_absent": True, "preassigned_train_realization": True,
                             "native_time_and_zero_motion_intervals_retained": True,
                             "raw_xy_preserved_without_reconstruction": True},
                  "selection": {"realization": "SHA256('g35-realization-v1:' + parent_base_id), integer modulo 2 maps to [350101,350102]; no fallback to another successful realization",
                                "minimum_training_parent_target": MIN_TRAIN_PARENTS,
                                "legs": "All accepted complete legs from each designated parent realization; multiple legs are not independent parent identities",
                                "shortfall": "Preserve result and stop; no automatic reruns, relabeling or duplication"},
                  "sources": sources,
                  "script_sha256": digest(Path(__file__)),
                  "artifacts": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                                for name in ("dataset.npz", "leg-metadata.json.gz")},
                  "limits": ["Same public map and demand templates; three simulation seeds are not three independent new demand populations.",
                             "These are SUMO observations, not the supplied historical CNR CSV and not diffusion outputs.",
                             "Training and validation supervision uses observed routes. Prospective generation conditions must be independently specified.",
                             "Readiness means numeric preparation and size/support checks only; no new GAF accuracy or generation-quality result is claimed."]}
        save(output / "complete.json", report)
        print(json.dumps({"complete": True, "training_ready": report["training_ready"],
                          "groups": {s: {k: groups[s][k] for k in ("legs", "parents", "class_counts")}
                                     for s in ("train", "val")}, "reasons": reasons}), flush=True)
        return report
    except BaseException as error:
        save(output / "failed.json", {"complete": False, "error": repr(error)})
        raise


def self_test():
    from g33_prepare import encode

    def parent_for(seed, prefix):
        for i in range(100):
            parent = prefix + str(i)
            if assigned_seed(parent) == seed:
                return parent
        raise AssertionError("No fixture parent")

    def fixture(seed, split, parent, count):
        raw = [np.c_[100 + np.arange(3 + i) * DT,
                     np.arange(3 + i) * (i + 1), np.zeros(3 + i)] for i in range(count)]
        encoded = [encode(points) for points in raw]
        for row, points in zip(encoded, raw):
            row["xy"] = points[:, 1:].copy()
            row["native_speed"] = np.full(len(points), np.nan)
        arrays = {"ids": np.asarray([f"{seed}:vehicle:{parent}:leg{i}" for i in range(count)]),
                  "parent_ids": np.asarray([parent] * count), "partition": np.asarray([split] * count),
                  "labels": np.asarray([0] * count, dtype=np.int64),
                  "episodes": np.asarray([seed] * count, dtype=np.int64),
                  "ordinals": np.arange(count, dtype=np.int64),
                  "event_start": np.asarray([100.] * count),
                  "event_end": np.asarray([100. + x["duration"] for x in encoded]),
                  "lines": np.asarray([""] * count),
                  "source_route_hashes": np.asarray([""] * count),
                  "source_parent_route_hashes": np.asarray([""] * count),
                  "realized_route_hashes": np.asarray([""] * count),
                  "paths": np.stack([x["paths"] for x in encoded]),
                  "length": np.asarray([x["length"] for x in encoded]),
                  "duration": np.asarray([x["duration"] for x in encoded]),
                  "legacy_progress": np.stack([x["legacy_progress"] for x in encoded]),
                  "offset": np.r_[0, np.cumsum([len(x["time"]) for x in encoded])],
                  "path_offset": np.r_[0, np.cumsum([len(x["path_arc"]) for x in encoded])]}
        arrays.update({key: np.concatenate([x[key] for x in encoded]) for key in (*POINT_KEYS, *PATH_KEYS)})
        metadata = [{"leg_id": str(arrays["ids"][i]), "parent_base_id": parent, "partition": split, "label": 0,
                     "episode": seed, "ordinal": i, "event_start": 100., "event_end": float(arrays["event_end"][i])}
                    for i in range(count)]
        return arrays, metadata

    parents = {350101: parent_for(350101, "first"), 350102: parent_for(350102, "second"), 350201: "development"}
    fixtures = {seed: fixture(seed, split, parents[seed], 2 if seed == 350101 else 1)
                for seed, split in EPISODES.items()}
    merged = combine_arrays({seed: pair[0] for seed, pair in fixtures.items()})
    assert merged["offset"].tolist() == [0, 3, 7, 10, 13]
    assert merged["path_offset"].tolist() == [0, 3, 7, 10, 13]
    assert np.array_equal(merged["time"][7:10], fixtures[350102][0]["time"])
    assert np.array_equal(merged["xy"][7:10], fixtures[350102][0]["xy"])
    assert np.isnan(merged["native_speed"]).all()
    failures = 0
    for corrupt in ("cross_partition_parent", "bad_offset", "wrong_realization"):
        episodes = {seed: {k: v.copy() for k, v in pair[0].items()} for seed, pair in fixtures.items()}
        if corrupt == "cross_partition_parent":
            episodes[350201]["parent_ids"] = np.asarray([parents[350101]])
        elif corrupt == "bad_offset":
            episodes[350102]["offset"][-1] -= 1
        else:
            episodes[350102]["parent_ids"] = np.asarray([parents[350101]])
        try:
            combine_arrays(episodes)
        except ValueError:
            failures += 1
        else:
            raise AssertionError(corrupt + " was not rejected")
    with tempfile.TemporaryDirectory(prefix="g35-merge-check-") as root:
        root = Path(root)
        directories = []
        for seed, (arrays, metadata) in fixtures.items():
            folder = root / str(seed); folder.mkdir(); directories.append(folder)
            np.savez_compressed(folder / "dataset.npz", **arrays)
            with gzip.open(folder / "leg-metadata.json.gz", "wt") as stream:
                json.dump({"accepted": metadata}, stream)
            receipt = {"complete": True, "schema": SCHEMA, "classes": CLASSES,
                       "provenance_schema": "g35-native-full-day-extraction-v1", "episode": seed, "split": EPISODES[seed],
                       "checks": {name: True for name in ("old_test_templates_excluded_by_split_metadata_only", "parent_episode_preassignment_before_FCD", "source_and_simulation_hashes_verified", "native_time_and_zero_motion_intervals_retained", "all_event_and_sampling_accepted_legs_retained", "exact_vertex_roundtrip_below_1e_minus_7_m")},
                       "sources": {"source_manifest_sha256": "fixture-source", "original_split_manifest_sha256": "fixture-split"},
                       "artifacts": {name: {"sha256": digest(folder / name), "bytes": (folder / name).stat().st_size}
                                     for name in ("dataset.npz", "leg-metadata.json.gz")}}
            save(folder / "complete.json", receipt)
        report = merge(directories, root / "merged")
        assert report["groups"]["train"]["legs"] == 3 and report["groups"]["train"]["parents"] == 2
        assert report["training_ready"] is False and report["groups"]["val"]["parents"] == 1
        # Hash-valid but semantically misaligned metadata must still fail.
        folder = directories[0]
        metadata = fixtures[350101][1]
        metadata[0]["leg_id"] = "not-the-first-leg"
        with gzip.open(folder / "leg-metadata.json.gz", "wt") as stream:
            json.dump({"accepted": metadata}, stream)
        receipt = json.loads((folder / "complete.json").read_text())
        receipt["artifacts"]["leg-metadata.json.gz"] = {"sha256": digest(folder / "leg-metadata.json.gz"), "bytes": (folder / "leg-metadata.json.gz").stat().st_size}
        save(folder / "complete.json", receipt)
        try:
            load_episode(folder)
        except ValueError:
            failures += 1
        else:
            raise AssertionError("Misaligned metadata accepted")
        # Altered arrays cannot inherit an old completion receipt.
        with (directories[1] / "dataset.npz").open("ab") as stream:
            stream.write(b"changed")
        try:
            load_episode(directories[1])
        except ValueError:
            failures += 1
        else:
            raise AssertionError("Changed prepared artifact accepted")
    assert failures == 5
    print("g35_merge_construct_checks_passed: offsets, unique parents, preassignment, isolation, semantic metadata, hashes, readiness", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.inputs is None or args.output is None:
            parser.error("--inputs and --output are required outside --self-test")
        merge(args.inputs, args.output)
