"""Extract native MoST observations without treating riders as pedestrians.

This is an observation extractor, not a completed-trip reconstruction. Original
entity IDs and raw arrays stay in the isolated experiment; complete.json contains
aggregate counts only. Coordinates and native speeds retain SUMO units.
"""
import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from g29_prepare import record, sha256, span


CLASSES = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
VCLASS_LABEL = {"pedestrian": 0, "bus": 1, "motorcycle": 2,
                "passenger": 3, "bicycle": 4, "taxi": 5}
# An explicit source-type taxonomy exception, never a participant-ID heuristic:
# MoST basic.vType.xml defines uber as a passenger-class on-demand vehicle.
SOURCE_TYPE_LABEL = {"uber": 5}


def open_xml(path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else path.open("rb")


def children(path):
    """Yield one direct child at a time, releasing its XML tree afterwards."""
    with open_xml(path) as stream:
        stack = []
        for event, node in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                stack.append(node)
            else:
                if len(stack) == 2:
                    yield node
                    stack[0].remove(node)
                    node.clear()
                stack.pop()


def types_from_files(paths):
    types = {"DEFAULT_VEHTYPE": "passenger", "DEFAULT_PEDTYPE": "pedestrian",
             "DEFAULT_BIKETYPE": "bicycle"}
    distributions = set()
    for path in paths:
        with open_xml(path) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag == "vTypeDistribution":
                    distributions.add(element.get("id"))
                if element.tag == "vType" and element.get("id"):
                    name = element.get("id")
                    # vType references inside distributions need not redeclare
                    # physical properties. A bare reference cannot override an
                    # explicit definition loaded from another supplied file.
                    value = element.get("vClass")
                    if value is not None:
                        if name in types and types[name] != value:
                            raise ValueError(f"Conflicting vClass for vType {name}")
                        types[name] = value
                element.clear()
    return types, sorted(x for x in distributions if x)


def arrival_information(path):
    records = {}
    for node in children(path):
        if node.tag not in ("tripinfo", "personinfo"):
            continue
        ident = node.get("id")
        if not ident:
            continue
        depart = float(node.get("depart", "-1"))
        arrival = float(node.get("arrival", "-1"))
        if node.tag == "personinfo":
            stages = [child for child in node if child.tag in ("walk", "ride", "stop", "access")]
            if stages:
                # The final stage, not the maximum earlier arrival, determines
                # whether an interrupted multimodal person chain completed.
                arrival = float(stages[-1].get("arrival", "-1"))
                if depart < 0:
                    depart = float(stages[0].get("depart", "-1"))
        key = ("person" if node.tag == "personinfo" else "vehicle", ident)
        if key in records:
            raise ValueError("Duplicate tripinfo entity")
        records[key] = (depart, arrival)
    return records


def route_information(path):
    """Store observed route identity for grouped audits; never a model input."""
    if path is None:
        return {}
    result = {}
    for node in children(path):
        if node.tag != "vehicle" or not node.get("id"):
            continue
        routes = node.findall("route")
        distributions = node.findall("routeDistribution/route")
        # SUMO vehroute output may store the rerouting history; the final route
        # is used for an audit fingerprint, and reroute count remains explicit.
        candidates = distributions or routes
        if not candidates:
            continue
        edges = candidates[-1].get("edges", "").split()
        result[node.get("id")] = (hashlib.sha256(" ".join(edges).encode()).hexdigest(),
                                  node.get("line", ""), max(len(candidates)-1, 0))
    return result


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p05": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def extract(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists() or (output / "raw.npz").exists():
        raise FileExistsError("Refusing to overwrite an extracted observation cohort")
    types, distributions = types_from_files(args.types)
    arrivals = arrival_information(args.tripinfo)
    routes = route_information(args.routes)
    counts, unknown, unsupported, actual_types = Counter(), Counter(), Counter(), Counter()
    unknown_entities, unsupported_entities = {}, {}
    tracks, roads, road_index = {}, [], {}
    entity_types, changed_types = {}, set()
    time_min, time_max, previous_time = None, None, None
    timestep_counts = Counter()
    for step in children(args.fcd):
        if step.tag != "timestep":
            continue
        time = float(step.get("time"))
        if not np.isfinite(time) or (previous_time is not None and time <= previous_time):
            raise ValueError("FCD timesteps are not finite and strictly increasing")
        if previous_time is not None:
            dt = time-previous_time
            timestep_counts[f"{dt:.8f}"] += 1
            if not np.isclose(dt, .25, atol=1e-7, rtol=0):
                raise ValueError("FCD must contain native 0.25-second timesteps")
        previous_time = time
        time_min = time if time_min is None else time_min
        time_max = time
        for node in step:
            # Riders occur beneath a vehicle. Direct-child traversal is required;
            # iter('person') would incorrectly turn vehicle motion into walking.
            counts["nested_rider_observations_excluded"] += sum(1 for child in node.iter("person")) if node.tag == "vehicle" else 0
            if node.tag not in ("person", "vehicle"):
                counts["other_element_observations_excluded"] += 1
                continue
            if node.tag == "person" and node.get("vehicle", "") not in ("", "NULL", "None"):
                counts["direct_rider_observations_excluded"] += 1
                continue
            ident = node.get("id")
            if not ident:
                raise ValueError("FCD entity has no ID")
            key = (node.tag, ident)
            name = node.get("type", "")
            if key in entity_types and entity_types[key] != name:
                changed_types.add(key)
            entity_types.setdefault(key, name)
            actual_types[name] += 1
            vclass = types.get(name)
            if vclass is None:
                unknown[name or "<missing>"] += 1
                unknown_entities.setdefault(name or "<missing>", set()).add(key)
                continue
            label = SOURCE_TYPE_LABEL.get(name, VCLASS_LABEL.get(vclass))
            if label is None:
                unsupported[vclass] += 1
                unsupported_entities.setdefault(vclass, set()).add(key)
                continue
            if (node.tag == "person") != (label == 0):
                counts["entity_tag_class_mismatch_excluded"] += 1
                continue
            if key not in tracks:
                tracks[key] = {"values": array("d"), "road": array("i"), "label": label,
                               "type": name, "last_time": None, "invalid": False}
            row = tracks[key]
            if row["type"] != name or row["label"] != label:
                row["invalid"] = True
                counts["inconsistent_type_observations"] += 1
            if row["last_time"] is not None and time <= row["last_time"]:
                row["invalid"] = True
                counts["duplicate_entity_timestamps"] += 1
            row["last_time"] = time
            try:
                x, y = float(node.get("x", "nan")), float(node.get("y", "nan"))
                speed = float(node.get("speed", "nan"))
            except ValueError:
                x, y, speed = float("nan"), float("nan"), float("nan")
            if not np.isfinite([x, y]).all():
                row["invalid"] = True
                counts["invalid_coordinate_observations"] += 1
            if not np.isfinite(speed):
                counts["missing_native_speed_observations"] += 1
            road = node.get("lane", "") or node.get("edge", "")
            if road not in road_index:
                road_index[road] = len(roads)
                roads.append(road)
            row["road"].append(road_index[road])
            row["values"].extend((time, x, y, speed))
            counts["mapped_observations"] += 1
        counts["timesteps"] += 1

    rows, extras, rejected = [], [], []
    class_counts, native_speed_mae = Counter(), {str(c): [] for c in range(6)}
    native_velocity_stats = {str(c): [] for c in range(6)}
    for key in sorted(tracks):
        raw = tracks.pop(key)
        tag, base_id = key
        reason = None
        values = np.frombuffer(raw["values"], dtype=np.float64).reshape(-1, 4)
        if raw["invalid"] or key in changed_types:
            reason = "invalid_or_inconsistent_observations"
        elif len(values) < 2:
            reason = "fewer_than_two_observations"
        if reason is None:
            row = record(f"{args.episode}:{tag}:{base_id}", raw["label"], values[:, :3])
            if span(row) < 5.:
                reason = "longest_continuous_span_below_5s"
        if reason is not None:
            rejected.append((base_id, tag, raw["type"], reason))
            counts[f"entities_excluded:{reason}"] += 1
            continue
        start, end = row["core_start"], row["core_end"]
        full_speed = values[:, 3].copy()
        core_speed = full_speed[start:end].copy()
        full_road = np.frombuffer(raw["road"], dtype=np.int32).copy()
        core_road = full_road[start:end].copy()
        depart, arrival = arrivals.get(key, (-1., -1.))
        arrival_complete = depart >= 0 and arrival >= depart
        observation_complete = (arrival_complete and row["break_count"] == 0
                                and abs(values[0, 0]-depart) <= .25+1e-7
                                and abs(values[-1, 0]-arrival) <= .25+1e-7)
        route_hash, line, reroutes = routes.get(base_id, ("", "", -1)) if tag == "vehicle" else ("", "", -1)
        rows.append(row)
        extras.append({"base_id": base_id, "tag": tag, "vtype": raw["type"],
                       "arrival_complete": arrival_complete,
                       "observation_complete": observation_complete,
                       "depart": depart, "arrival": arrival,
                       "native_speed": core_speed, "full_native_speed": full_speed,
                       "road_index": core_road, "full_road_index": full_road,
                       "route_hash": route_hash, "line": line, "reroutes": reroutes})
        c = str(row["label"])
        class_counts[c] += 1
        xy_speed = np.linalg.norm(np.diff(row["core"][:, 1:], axis=0), axis=1) / .25
        midpoint_speed = (core_speed[:-1]+core_speed[1:])/2
        valid = np.isfinite(midpoint_speed)
        if valid.any():
            native_speed_mae[c].append(float(np.abs(xy_speed[valid]-midpoint_speed[valid]).mean()))
        if np.isfinite(core_speed).any():
            native_velocity_stats[c].append(float(np.nanmean(core_speed)))

    offsets = np.r_[0, np.cumsum([len(row["core"]) for row in rows], dtype=np.int64)]
    full_offsets = np.r_[0, np.cumsum([len(row["full"]) for row in rows], dtype=np.int64)]
    def join(key, dtype=np.float64, shape=None):
        values = [row[key] for row in rows] if key in ("core", "full") else [extra[key] for extra in extras]
        return np.concatenate(values).astype(dtype, copy=False) if values else np.empty(shape or (0,), dtype=dtype)
    def values(key, dtype=None):
        return np.asarray([extra[key] for extra in extras], dtype=dtype)
    arrays = dict(ids=np.asarray([row["id"] for row in rows], dtype=str),
                  labels=np.asarray([row["label"] for row in rows], dtype=np.int64),
                  txy=join("core", shape=(0, 3)), offset=offsets,
                  full_txy=join("full", shape=(0, 3)), full_offset=full_offsets,
                  duration=np.asarray([span(row) for row in rows]),
                  core_start=np.asarray([row["core_start"] for row in rows], dtype=np.int64),
                  core_end=np.asarray([row["core_end"] for row in rows], dtype=np.int64),
                  break_count=np.asarray([row["break_count"] for row in rows], dtype=np.int64),
                  eligible_512=np.asarray([len(row["core"]) >= 512 for row in rows], dtype=bool),
                  base_ids=values("base_id", str), entity_tags=values("tag", str), vtypes=values("vtype", str),
                  arrival_complete_flags=values("arrival_complete", bool),
                  observation_complete_flags=values("observation_complete", bool),
                  depart=values("depart", float), arrival=values("arrival", float),
                  native_speed=join("native_speed", np.float32), full_native_speed=join("full_native_speed", np.float32),
                  road_index=join("road_index", np.int32), full_road_index=join("full_road_index", np.int32),
                  road_ids=np.asarray(roads, dtype=str), route_hashes=values("route_hash", str),
                  lines=values("line", str), reroute_counts=values("reroutes", np.int64))
    temporary = output / "raw.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output / "raw.npz")
    rejected_values = np.asarray(rejected, dtype=str).reshape(-1, 4)
    np.savez_compressed(output / "excluded.npz", base_ids=rejected_values[:, 0],
                        entity_tags=rejected_values[:, 1], vtypes=rejected_values[:, 2], reasons=rejected_values[:, 3])
    sources = {"fcd": {"sha256": sha256(args.fcd)}, "tripinfo": {"sha256": sha256(args.tripinfo)},
               "types": [{"filename": p.name, "sha256": sha256(p)} for p in args.types]}
    if args.routes:
        sources["routes"] = {"sha256": sha256(args.routes)}
    report = {"complete": True, "episode": args.episode, "classes": CLASSES,
              "scope": "native observations; longest continuous .25-second segment >=5s per entity; not necessarily a complete trip",
              "pedestrian_scope": "direct FCD persons outside vehicles: on-foot/unboarded walking or waiting observations, with riding gaps retained; not complete multimodal person chains",
              "sources": sources, "raw_sha256": sha256(output/"raw.npz"),
              "script_sha256": sha256(Path(__file__)), "rows": len(rows),
              "class_counts": [class_counts[str(c)] for c in range(6)],
              "observed_time_range_s": [time_min, time_max], "timestep_intervals": dict(timestep_counts),
              "counts": dict(counts), "entities_with_vtype_changes": len(changed_types),
              "unknown_vtype_observations_excluded": dict(unknown),
              "unknown_vtype_entity_counts": {k: len(v) for k, v in unknown_entities.items()},
              "unsupported_vclass_observations_excluded": dict(unsupported),
              "unsupported_vclass_entity_counts": {k: len(v) for k, v in unsupported_entities.items()},
              "actual_vtype_observations": dict(actual_types), "type_to_vclass": types,
              "source_type_label_overrides": SOURCE_TYPE_LABEL, "distribution_ids": distributions,
              "moped_policy": "excluded: vClass moped is not silently equated with original motorcycle class",
              "class_label_scope": "declared SUMO vClass with source type uber override; original CNR class2 correspondence is not proven",
              "arrival_complete_counts": np.bincount(arrays["labels"][arrays["arrival_complete_flags"]], minlength=6).tolist(),
              "observation_complete_counts": np.bincount(arrays["labels"][arrays["observation_complete_flags"]], minlength=6).tolist(),
              "eligible_512_counts": np.bincount(arrays["labels"][arrays["eligible_512"]], minlength=6).tolist(),
              "xy_speed_vs_native_midpoint_mae_mps_per_track": {c: summarize(x) for c, x in native_speed_mae.items()},
              "native_track_mean_speed_mps": {c: summarize(x) for c, x in native_velocity_stats.items()},
              "arrival_flag_definition": "tripinfo vehicle arrival or last person-stage arrival >= departure; does not imply all stages were observed",
              "observation_complete_definition": "arrival flag plus no retained-observation continuity breaks and observed endpoints within .25s of declared departure/arrival",
              "route_metadata_scope": "final emitted route fingerprint and line for grouping/audit only; not a predictive generation input"}
    temporary = output / "complete.tmp.json"
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(output / "complete.json")
    print(json.dumps({"complete": True, "rows": len(rows), "class_counts": report["class_counts"],
                      "unknown_vtypes": dict(unknown), "output": str(output)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcd", type=Path, required=True)
    parser.add_argument("--types", type=Path, nargs="+", required=True)
    parser.add_argument("--tripinfo", type=Path, required=True)
    parser.add_argument("--routes", type=Path)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for raw.npz, excluded.npz, complete.json")
    extract(parser.parse_args())
