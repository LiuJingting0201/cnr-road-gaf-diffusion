"""Extract one preassigned full-day G35 v2 recording into complete vehicle legs.

Only one simulation realization is allowed for each parent, decided from its
identity before any outcomes are read. This is new native-demand supervision,
not a replay-invariance check or a prospective generation benchmark. Original
test coordinates are never loaded. Per-parent metadata remains beside the raw
dataset; the completion receipt contains aggregates and hashes only.
"""
import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import g31_legs as legs
import g33_prepare as prepared
from g30_prepare import entity_partition, original_id_partitions


TRAIN_SEEDS = (350101, 350102)
VAL_SEED = 350201
SEED_SPLIT = {**dict.fromkeys(TRAIN_SEEDS, "train"), VAL_SEED: "val"}
REALIZATION_NAMESPACE = "g35-realization-v1:"
PROVENANCE_SCHEMA = "g35-native-full-day-extraction-v1"
CLASSES = prepared.CLASSES


def assigned_train_seed(parent_base_id):
    if not isinstance(parent_base_id, str) or not parent_base_id:
        raise ValueError("Nonempty source parent identity required")
    digest = hashlib.sha256((REALIZATION_NAMESPACE + parent_base_id).encode()).digest()
    return TRAIN_SEEDS[int.from_bytes(digest, "big") % len(TRAIN_SEEDS)]


def assignment_reason(parent_base_id, partition, seed):
    """The admission decision has no coordinate, duration, label or score input."""
    if seed not in SEED_SPLIT or partition not in ("train", "val", "test"):
        raise ValueError("Unknown episode or parent partition")
    if partition == "test":
        return "original_test_parent_excluded"
    if partition != SEED_SPLIT[seed]:
        return "other_parent_partition"
    if partition == "train" and assigned_train_seed(parent_base_id) != seed:
        return "other_preassigned_train_realization"
    return None


def validate_config(config):
    if (config.get("schema") != "g35-native-full-day-v2"
            or config.get("seed") not in SEED_SPLIT
            or config.get("split") != SEED_SPLIT[config["seed"]]
            or config.get("begin") != 14400 or config.get("end") != 50400
            or config.get("native_step_s") != .25):
        raise ValueError("Not a declared G35 full-day simulation")


def recording_ids(config, simulation, partitions):
    """Verify the pre-simulation private cohort, never choose from outcomes."""
    spec = config.get("recording", {})
    if (spec.get("attributes") != "x,y,type,lane"
            or spec.get("vehicle_ids_file") != "fcd-ids.txt"):
        raise ValueError("Unexpected v2 FCD attribute or explicit-ID file contract")
    path = simulation / spec["vehicle_ids_file"]
    if path.is_symlink() or legs.digest(path) != spec.get("vehicle_ids_sha256"):
        raise ValueError("Frozen explicit FCD cohort file changed")
    records = path.read_text(encoding="utf-8").splitlines()
    if (not records or any(not ident or ident != ident.strip() for ident in records)
            or len(set(records)) != len(records)
            or len(records) != spec.get("vehicle_ids_count")):
        raise ValueError("Explicit FCD cohort is empty, duplicated or miscounted")
    seed = int(config["seed"])
    for ident in records:
        if assignment_reason(ident, entity_partition(ident, partitions), seed) is not None:
            raise ValueError("Recorded ID violates frozen parent/episode assignment")
    return set(records)


def collect_explicit_fcd(path, wanted, recorded_ids, candidates=None):
    """Keep exact candidate slices plus each parent's original first record.

    Every FCD record is still checked/counted, and parent type consistency and
    lane dtype width are accumulated over the complete observation stream.
    Candidate intervals use precisely cover()'s inclusive EPS boundaries; an
    initial record preserves its distinction between no parent observations
    and a parent with no observations inside an event. None is only the legacy
    full-retention mode used by the existing focused checks.
    """
    wanted = set(wanted)
    if any(tag != "vehicle" or ident not in recorded_ids for tag, ident in wanted):
        raise ValueError("Candidate parent is outside the preselected vehicle FCD cohort")
    intervals = None
    if candidates is not None:
        intervals = {key: [] for key in wanted}
        seen = set()
        for candidate in candidates:
            key = candidate["entity_tag"], candidate["parent_base_id"]
            if key not in intervals:
                raise ValueError("Candidate interval is outside the requested parent set")
            seen.add(key)
            start, end = float(candidate["event_start"]), float(candidate["event_end"])
            if not (math.isfinite(start) and math.isfinite(end)):
                raise ValueError("Candidate interval boundaries must be finite")
            if end > start:
                intervals[key].append((start-legs.EPS, end+legs.EPS))
        if seen != wanted:
            raise ValueError("Requested parent lacks candidate interval metadata")
        for key, rows in intervals.items():
            merged = []
            for start, end in sorted(rows):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            intervals[key] = merged
    observations = {key: {"values": array("d"), "roads": [], "types": set()} for key in wanted}
    road_width = dict.fromkeys(wanted, 0)
    interval_index = dict.fromkeys(wanted, 0)
    counts = Counter(non_target_observations=0, observed_direct_vehicle_records=0,
                     candidate_parent_vehicle_records=0, speed_field_absent=0, pos_field_absent=0,
                     stored_candidate_parent_vehicle_records=0,
                     stored_first_records_outside_candidate_intervals=0,
                     omitted_outside_candidate_intervals_records=0)
    observed_ids, previous = set(), None
    for step in legs.children(path):
        if step.tag != "timestep":
            continue
        stamp = legs.number(step.get("time"))
        if stamp < 0 or (previous is not None and abs(stamp - previous - legs.DT) > legs.EPS):
            raise ValueError("V2 FCD does not retain every native quarter-second timestep")
        previous = stamp
        for node in step:
            ident = node.get("id")
            if node.tag != "vehicle" or ident not in recorded_ids:
                counts["non_target_observations"] += 1
                raise ValueError("V2 FCD contains an observation outside the explicit vehicle cohort")
            counts["observed_direct_vehicle_records"] += 1
            counts["speed_field_absent"] += "speed" not in node.attrib
            counts["pos_field_absent"] += "pos" not in node.attrib
            observed_ids.add(ident)
            key = ("vehicle", ident)
            if key not in observations:
                continue
            counts["candidate_parent_vehicle_records"] += 1
            row = observations[key]
            lane = node.get("lane") or ""
            road_width[key] = max(road_width[key], len(lane))
            row["types"].add(node.get("type", ""))
            inside = True
            if intervals is not None:
                spans, cursor = intervals[key], interval_index[key]
                while cursor < len(spans) and stamp > spans[cursor][1]:
                    cursor += 1
                interval_index[key] = cursor
                inside = cursor < len(spans) and spans[cursor][0] <= stamp <= spans[cursor][1]
            first = not len(row["values"])
            if not (inside or first):
                counts["omitted_outside_candidate_intervals_records"] += 1
                continue
            counts["stored_candidate_parent_vehicle_records"] += 1
            counts["stored_first_records_outside_candidate_intervals"] += first and not inside
            row["values"].extend((stamp, legs.number(node.get("x"), math.nan),
                legs.number(node.get("y"), math.nan), legs.number(node.get("speed"), math.nan),
                legs.number(node.get("pos"), math.nan)))
            row["roads"].append(lane)
    for key, row in observations.items():
        row["values"] = np.frombuffer(row["values"], dtype=np.float64).reshape(-1, 5)
        row["roads"] = np.asarray(row["roads"], dtype="U"+str(max(1, road_width[key])))
    counts["explicit_recording_IDs"] = len(recorded_ids)
    counts["observed_recording_IDs"] = len(observed_ids)
    counts["recording_IDs_without_observations"] = len(recorded_ids - observed_ids)
    if (counts["stored_candidate_parent_vehicle_records"] + counts["omitted_outside_candidate_intervals_records"]
            != counts["candidate_parent_vehicle_records"]):
        raise AssertionError("Full-stream and stored observation counts do not reconcile")
    return observations, counts


def arrays_from_rows(rows, encoded):
    """Same explicit representation contract as G33, with G35 provenance."""
    if len(rows) != len(encoded):
        raise ValueError("Metadata/encoding alignment failed")
    identifiers = [r["leg_id"] for r in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("Duplicate complete-leg identity")
    arrays = {
        "ids": np.asarray(identifiers, dtype=str),
        "parent_ids": np.asarray([r["parent_base_id"] for r in rows], dtype=str),
        "partition": np.asarray([r["partition"] for r in rows], dtype=str),
        "labels": np.asarray([r["label"] for r in rows], dtype=np.int64),
        "episodes": np.asarray([r["episode"] for r in rows], dtype=np.int64),
        "ordinals": np.asarray([r["ordinal"] for r in rows], dtype=np.int64),
        "event_start": np.asarray([r["event_start"] for r in rows], dtype=np.float64),
        "event_end": np.asarray([r["event_end"] for r in rows], dtype=np.float64),
        "lines": np.asarray([r["line"] for r in rows], dtype=str),
        "source_route_hashes": np.asarray([r["source_route_hash"] for r in rows], dtype=str),
        "source_parent_route_hashes": np.asarray([r["source_parent_route_hash"] for r in rows], dtype=str),
        "realized_route_hashes": np.asarray([r["realized_route_hash"] for r in rows], dtype=str),
        "paths": np.stack([r["paths"] for r in encoded]) if rows else np.empty((0, 129, 2), np.float32),
        "length": np.asarray([r["length"] for r in encoded], dtype=np.float64),
        "duration": np.asarray([r["duration"] for r in encoded], dtype=np.float64),
        "legacy_progress": np.stack([r["legacy_progress"] for r in encoded]) if rows else np.empty((0, 257), np.float64),
        "offset": np.r_[0, np.cumsum([len(r["time"]) for r in encoded])].astype(np.int64),
        "path_offset": np.r_[0, np.cumsum([len(r["path_arc"]) for r in encoded])].astype(np.int64),
    }
    for key in ("time", "progress", "path_xy", "path_arc", "xy", "native_speed"):
        arrays[key] = (np.concatenate([r[key] for r in encoded]) if rows else
                       np.empty((0, 2) if key in ("path_xy", "xy") else (0,), np.float64))
    return arrays


def per_class_summary(rows, encoded):
    result = {}
    for label, name in enumerate(CLASSES):
        indices = [i for i, row in enumerate(rows) if row["label"] == label]
        chosen = [rows[i] for i in indices]
        result[name] = {
            "accepted_legs": len(chosen),
            "parents": len({r["parent_base_id"] for r in chosen}),
            "source_leg_route_fingerprints_nonempty": len({r["source_route_hash"] for r in chosen if r["source_route_hash"]}),
            "source_parent_route_fingerprints_nonempty": len({r["source_parent_route_hash"] for r in chosen if r["source_parent_route_hash"]}),
            "realized_leg_route_fingerprints_nonempty": len({r["realized_route_hash"] for r in chosen if r["realized_route_hash"]}),
            "source_lines_nonempty": len({r["line"] for r in chosen if r["line"]}),
            "missing_source_leg_route": sum(not r["source_route_hash"] for r in chosen),
            "missing_source_parent_route": sum(not r["source_parent_route_hash"] for r in chosen),
            "kind_counts": dict(Counter(r["kind"] for r in chosen)),
            "duration_s": legs.summarize(encoded[i]["duration"] for i in indices),
            "native_points": legs.summarize(len(encoded[i]["time"]) for i in indices),
            "constant_legs": sum(encoded[i]["length"] == 0 for i in indices),
        }
    return result


def extract(args):
    source_path, simulation, output = [p.resolve() for p in (args.source, args.simulation, args.output)]
    for protected in (source_path, simulation):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Extraction must be separate from immutable simulation/source")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        # Deliberately no G31 comparison receipt: this is a new simulation.
        from g35c_adopt import verify_adopted_receipt, verify_source
        config, simulation_receipt = verify_adopted_receipt(args.adoption)
        if Path(simulation_receipt["simulation_path"]).resolve() != simulation:
            raise ValueError("Adopted receipt does not name the selected simulation folder")
        validate_config(config)
        verify_source(source_path)
        if config["source_manifest_sha256"] != legs.digest(source_path / "complete.json"):
            raise ValueError("Simulation and extractor use different source manifests")
        seed, split = int(config["seed"]), str(config["split"])
        partitions = original_id_partitions(args.cnr_splits)
        recorded_ids = recording_ids(config, simulation, partitions)
        parents, partition_exclusions = legs.parent_events(simulation, partitions)
        selection_exclusions = Counter()
        selected = {}
        for key, parent in parents.items():
            # Personal activity chains remain in SUMO dynamics, not this
            # predeclared five-vehicle-class training cohort.
            if parent["tag"] != "vehicle":
                selection_exclusions["person_outside_five_vehicle_scope"] += 1
                continue
            reason = assignment_reason(parent["base_id"], parent["partition"], seed)
            if reason:
                selection_exclusions[reason] += 1
            else:
                selected[key] = parent
        del parents
        source = legs.source_inventory(source_path, set(selected))
        candidates, parent_metadata, parent_rejections = [], {}, []
        parent_counts, type_exclusions, boundary_exclusions, source_links = (Counter() for _ in range(4))
        for key, parent in selected.items():
            typename = parent["tripinfo"].get("vType") or parent["tripinfo"].get("type", "")
            vclass = source["types"].get(typename)
            label = legs.SOURCE_TYPE_LABEL.get(typename, legs.VCLASS_LABEL.get(vclass))
            if label not in range(1, 6):
                type_exclusions[typename or "<missing>"] += 1
                continue
            if parent["base_id"] not in recorded_ids:
                raise ValueError("A taxonomy-eligible assigned parent is absent from the explicit FCD cohort")
            parent.update(label=label, vtype=typename)
            parent_counts[CLASSES[label - 1]] += 1
            found, rejected = legs.candidate_legs(parent, source)
            source_links[parent.get("source_link_status", "unknown")] += 1
            for reason, count in rejected.items():
                boundary_exclusions[f"{CLASSES[label - 1]}:{reason}"] += count
                parent_rejections.append({"parent_base_id": parent["base_id"], "label": label - 1,
                                          "episode": seed, "partition": split,
                                          "reason": reason, "count": count})
            candidates.extend(found)
            if found:
                parent_metadata[parent["base_id"]] = {
                    "parent_base_id": parent["base_id"], "partition": split, "episode": seed,
                    "label": label - 1, "vtype": typename,
                    "source_link_status": parent["source_link_status"],
                    "source_plan": parent["source_plan"],
                    "emitted_route_audit_only": parent["emitted"],
                    "tripinfo": parent["tripinfo"], "stop_events": parent["stop_events"],
                }
        prepared.save(output / "progress.json", {"stage": "collecting_native_FCD", "episode": seed,
                      "candidate_legs": len(candidates), "candidate_parents": len(parent_metadata),
                      "seconds": time.monotonic() - started})
        print(json.dumps({"stage": "collecting_native_FCD", "episode": seed,
                          "candidate_legs": len(candidates), "candidate_parents": len(parent_metadata)}), flush=True)
        wanted = {(row["entity_tag"], row["parent_base_id"]) for row in candidates}
        observed, fcd_counts = collect_explicit_fcd(simulation / "fcd.xml.gz", wanted, recorded_ids, candidates)
        accepted, rejected_rows, encoded = [], [], []
        coverage_exclusions, source_status, realized_status, agreement = (Counter() for _ in range(4))
        for candidate in candidates:
            key = candidate["entity_tag"], candidate["parent_base_id"]
            values, reason, diagnostics = legs.cover(candidate, observed[key])
            candidate["coverage"] = diagnostics
            candidate["episode"] = seed
            if reason:
                rejected_rows.append({**candidate, "label": candidate["label"] - 1, "rejection_reason": reason})
                coverage_exclusions[f"{CLASSES[candidate['label'] - 1]}:{reason}"] += 1
                continue
            row = dict(candidate)
            # Existing G31 helper expects its six-class label; convert only
            # after event/coverage checks and preserve the source-label field.
            row["source_label"] = row["label"]
            row["label"] -= 1
            row["leg_id"] = f"{seed}:vehicle:{row['parent_base_id']}:leg{row['ordinal']}"
            row["realized_route_audit_only"] = legs.realized_leg_audit(candidate, selected[key], values[0], values[1])
            planned_edges = row["source_plan"]["edges"]
            actual_edges = row["realized_route_audit_only"]["edges"]
            row["source_and_realized_edges_equal"] = planned_edges == actual_edges if planned_edges and actual_edges else None
            row["source_route_hash"] = legs.fingerprint(planned_edges) if planned_edges else ""
            row["source_parent_route_hash"] = row["source_plan"].get("parent_edges_hash", "")
            row["realized_route_hash"] = legs.fingerprint(actual_edges) if actual_edges else ""
            row["line"] = row["source_plan"]["source_attributes"].get("line", "")
            row["status"] = "complete_event_and_native_FCD"
            row["route_condition_scope"] = "Observed path is paired training supervision; never a prospective generation request"
            representation = prepared.encode(values[0][:, :3])
            # Preserve the exact source coordinates for GAF construction;
            # inverse progress decoding is only an audited representation.
            representation["xy"] = values[0][:, 1:3].copy()
            representation["native_speed"] = values[0][:, 3].copy()
            row["unsupported_by_current_motion_trainer"] = representation["length"] <= 0
            accepted.append(row)
            encoded.append(representation)
            source_status[row["source_plan"]["status"]] += 1
            realized_status[row["realized_route_audit_only"]["status"]] += 1
            agreement[str(row["source_and_realized_edges_equal"])] += 1
        del observed
        if any(assignment_reason(r["parent_base_id"], r["partition"], seed) is not None for r in accepted):
            raise ValueError("Accepted output violates preassigned parent/episode split")
        arrays = arrays_from_rows(accepted, encoded)
        unsupported = [r["leg_id"] for r in accepted if r["unsupported_by_current_motion_trainer"]]
        metadata = {
            "provenance_schema": PROVENANCE_SCHEMA, "episode": seed, "split": split,
            "classes": CLASSES, "accepted": accepted, "coverage_rejected": rejected_rows,
            "parent_boundary_rejections": parent_rejections, "candidate_parents": parent_metadata,
            "unsupported_zero_length_leg_ids": unsupported,
        }
        np.savez_compressed(output / "dataset.npz", **arrays)
        with gzip.open(output / "leg-metadata.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(metadata, stream, allow_nan=False, separators=(",", ":"))
        groups = {partition: prepared.summary(arrays, np.flatnonzero(arrays["partition"] == partition), encoded)
                  for partition in ("train", "val")}
        groups["all"] = prepared.summary(arrays, range(len(accepted)), encoded)
        own_dir = Path(__file__).resolve().parent
        helper_names = ("g35c_extract.py", "g35c_adopt.py", "g35b_extract.py", "g35b_simulate.py", "g35_simulate.py", "g31_legs.py", "g33_prepare.py",
                        "g30_extract.py", "g30_prepare.py", "g30_demand_audit.py", "g29_prepare.py")
        report = {
            "complete": True, "schema": prepared.SCHEMA, "provenance_schema": PROVENANCE_SCHEMA,
            "classes": CLASSES, "episode": seed, "split": split, "seconds": time.monotonic() - started,
            "scope": "Native full-day complete vehicle legs; one identity-preassigned realization per parent",
            "input_recording_schema": config["schema"],
            "input_recording": {**config["recording"],
                "geometry": "Unaltered native FCD x,y at .25s; no resampling or coordinate imputation",
                "omitted_fields": "SUMO speed/pos are not recorded; ancillary native_speed contains NaN, never fabricated physical speed"},
            "formal_generation_evaluation_ready": False,
            "train_ready": bool(accepted) and not unsupported and all(parent_counts[c] for c in CLASSES)
                           and set(arrays["labels"]) == set(range(5)),
            "candidate_legs": len(candidates), "accepted_legs": len(accepted), "vehicle_legs_retained": len(accepted),
            "groups": groups, "per_class": per_class_summary(accepted, encoded),
            "parent_counts_after_assignment_and_taxonomy": dict(parent_counts),
            "partition_exclusions": dict(partition_exclusions), "assignment_exclusions": dict(selection_exclusions),
            "type_exclusions": dict(type_exclusions), "boundary_exclusions": dict(boundary_exclusions),
            "coverage_exclusions": dict(coverage_exclusions), "fcd_counts": dict(fcd_counts),
            "source_link_counts": dict(source_links), "source_plan_status_counts": dict(source_status),
            "realized_route_status_counts": dict(realized_status), "source_realized_edges_equal_counts": dict(agreement),
            "unsupported_by_current_motion_trainer": {"zero_length_legs": len(unsupported),
                "retained_in_dataset": True, "ids_in_private_metadata": "unsupported_zero_length_leg_ids"},
            "checks": {"source_and_simulation_hashes_verified": True,
                "native_time_and_zero_motion_intervals_retained": True,
                "all_event_and_sampling_accepted_legs_retained": True,
                "exact_vertex_roundtrip_below_1e_minus_7_m": True,
                "parent_episode_preassignment_before_FCD": True,
                "all_accepted_parents_in_explicit_FCD_cohort": True,
                "all_recorded_IDs_follow_original_parent_and_episode_assignment": True,
                "no_unselected_FCD_observations": True,
                "old_test_templates_excluded_by_split_metadata_only": True},
            "assignment": {"namespace": REALIZATION_NAMESPACE, "train_seeds": list(TRAIN_SEEDS),
                "rule": "int.from_bytes(SHA256(namespace+parent_base_id),big) modulo 2 selects seed; no fallback",
                "val_seed": VAL_SEED, "parent_partition_rule": "Inherited CNR split, else existing G30 identity hash",
                "multiple_complete_activity_legs": "All legs from the parent's one assigned simulation; not independent parents"},
            "representation": {"native_step_s": .25, "route_condition_points": 129,
                "target": "Untruncated physical-time normalized progress at native observations",
                "decoder": "All observed nonduplicate path vertices; exact ordered arc interpolation",
                "raw_coordinates": "xy is unmodified source FCD x,y aligned with time/offset; native_speed is optional ancillary SUMO output",
                "velocity_GAF_source": "Coordinate-derived finite differences, not the ancillary SUMO native_speed field",
                "stops": "Zero-motion samples retained; low speed does not define activity boundaries",
                "constant_paths": "Retained and flagged unsupported by current positive-length motion trainer",
                "normalization": "No fitted statistics; later fitting must use training rows only"},
            "sources": {"source_manifest_sha256": legs.digest(source_path / "complete.json"),
                "simulation_receipt_sha256": legs.digest(args.adoption / "complete.json"),
                "explicit_FCD_ids_sha256": config["recording"]["vehicle_ids_sha256"],
                "simulation_config_sha256": simulation_receipt["config_sha256"],
                "original_split_manifest_sha256": legs.digest(args.cnr_splits),
                "scripts_sha256": {name: legs.digest(own_dir / name) for name in helper_names}},
            "artifacts": {name: {"sha256": legs.digest(output / name), "bytes": (output / name).stat().st_size}
                          for name in ("dataset.npz", "leg-metadata.json.gz")},
            "limits": ["One episode slice of G35; merge by declared disjoint parent/episode assignment before training.",
                "Source plans and realized-route audit metadata are separate; observed paths are supervision only.",
                "Repeated public routes and bus lines do not establish unseen-road/city generalization.",
                "Simulation observations are neither new real-world CNR measurements nor diffusion outputs."]}
        prepared.save(output / "complete.json", report)
        prepared.save(output / "progress.json", {"stage": "complete", "episode": seed,
                      "accepted_legs": len(accepted), "seconds": time.monotonic() - started})
        print(json.dumps({"complete": True, "episode": seed, "accepted_legs": len(accepted),
                          "parents": groups["all"]["parents"], "train_ready": report["train_ready"]}), flush=True)
        return report
    except BaseException as error:
        prepared.save(output / "failed.json", {"complete": False, "error": repr(error),
                      "seconds": time.monotonic() - started})
        raise


def self_test():
    import tempfile
    import unittest

    class RecordingChecks(unittest.TestCase):
        def fcd(self, folder, changes=None, extras=None, times=(0., .25, .5)):
            path = Path(folder) / "fixture.xml.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("<fcd-export>")
                for j, stamp in enumerate(times):
                    attrs = {"id": "car", "type": "passenger1", "lane": "a_0",
                             "x": str(j), "y": "2"}
                    if changes and j == 1:
                        for key, value in changes.items():
                            if value is None:
                                attrs.pop(key, None)
                            else:
                                attrs[key] = value
                    stream.write(f'<timestep time="{stamp}"><vehicle ' +
                                 " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "/>")
                    if extras:
                        stream.write(extras)
                    stream.write("</timestep>")
                stream.write("</fcd-export>")
            return path

        def test_speed_and_lane_position_may_be_absent_without_changing_xy(self):
            with tempfile.TemporaryDirectory(prefix="g35b-fcd-") as folder:
                observations, counts = collect_explicit_fcd(self.fcd(folder), {("vehicle", "car")}, {"car"})
                row = observations["vehicle", "car"]
                np.testing.assert_array_equal(row["values"][:, :3], [[0., 0., 2.], [.25, 1., 2.], [.5, 2., 2.]])
                self.assertTrue(np.isnan(row["values"][:, 3:]).all())
                values, reason, diagnostics = legs.cover({"event_start": 0., "event_end": .5,
                                                          "vtype": "passenger1"}, row)
                self.assertIsNone(reason)
                self.assertEqual(diagnostics["missing_native_speed_points"], 3)
                self.assertEqual(diagnostics["missing_position_on_lane_points"], 3)
                encoded = prepared.encode(values[0][:, :3])
                self.assertEqual(encoded["length"], 2.)
                np.testing.assert_array_equal(encoded["time"], [0., .25, .5])
                self.assertEqual(counts["non_target_observations"], 0)
                self.assertEqual(counts["speed_field_absent"], 3)
                self.assertEqual(counts["pos_field_absent"], 3)

        def test_missing_or_nonfinite_coordinate_is_not_imputed(self):
            for alteration in ({"x": None}, {"y": "nan"}):
                with tempfile.TemporaryDirectory(prefix="g35b-invalid-xy-") as folder:
                    observations, _ = collect_explicit_fcd(self.fcd(folder, changes=alteration), {("vehicle", "car")}, {"car"})
                    row = observations["vehicle", "car"]
                    self.assertFalse(np.isfinite(row["values"][:, :3]).all())
                    _, reason, _ = legs.cover({"event_start": 0., "event_end": .5, "vtype": "passenger1"}, row)
                    self.assertEqual(reason, "invalid_FCD_values_or_inconsistent_vtype")

        def test_road_and_type_are_still_required(self):
            for alteration, expected in (({"lane": None}, "FCD_road_reference_missing"),
                                         ({"type": None}, "invalid_FCD_values_or_inconsistent_vtype")):
                with tempfile.TemporaryDirectory(prefix="g35b-required-") as folder:
                    observations, _ = collect_explicit_fcd(self.fcd(folder, changes=alteration), {("vehicle", "car")}, {"car"})
                    _, reason, _ = legs.cover({"event_start": 0., "event_end": .5, "vtype": "passenger1"}, observations["vehicle", "car"])
                    self.assertEqual(reason, expected)

        def test_explicit_recording_cohort_is_checked_before_candidate_filter(self):
            with tempfile.TemporaryDirectory(prefix="g35b-cohort-") as folder:
                extra = '<vehicle id="recorded-but-no-complete-leg" type="passenger1" lane="b_0" x="3" y="4"/>'
                path = self.fcd(folder, extras=extra)
                _, counts = collect_explicit_fcd(path, {("vehicle", "car")}, {"car", "recorded-but-no-complete-leg"})
                self.assertEqual(counts["observed_direct_vehicle_records"], 6)
                self.assertEqual(counts["candidate_parent_vehicle_records"], 3)
                self.assertEqual(counts["non_target_observations"], 0)
                with self.assertRaises(ValueError):
                    collect_explicit_fcd(path, {("vehicle", "car")}, {"car"})
                with self.assertRaises(ValueError):
                    collect_explicit_fcd(path, {("vehicle", "not-recorded")}, {"car"})

        def test_recorded_ids_are_hash_and_assignment_verified(self):
            parent = next(f"private-parent-{i}" for i in range(100)
                          if assigned_train_seed(f"private-parent-{i}") == 350101)
            with tempfile.TemporaryDirectory(prefix="g35b-ids-") as folder:
                folder = Path(folder)
                path = folder / "fcd-ids.txt"
                path.write_text(parent + "\n")
                config = {"seed": 350101, "recording": {"vehicle_ids_file": "fcd-ids.txt",
                    "vehicle_ids_sha256": legs.digest(path), "vehicle_ids_count": 1, "attributes": "x,y,type,lane"}}
                self.assertEqual(recording_ids(config, folder, {parent: "train"}), {parent})
                for partition in ("val", "test"):
                    with self.assertRaises(ValueError):
                        recording_ids(config, folder, {parent: partition})
                path.write_text(parent + "\n" + parent + "\n")
                with self.assertRaises(ValueError):
                    recording_ids(config, folder, {parent: "train"})
                config["recording"].update(vehicle_ids_sha256=legs.digest(path), vehicle_ids_count=2)
                with self.assertRaises(ValueError):
                    recording_ids(config, folder, {parent: "train"})

        def test_native_time_gaps_and_old_v1_config_are_rejected(self):
            with tempfile.TemporaryDirectory(prefix="g35b-clock-") as folder:
                with self.assertRaises(ValueError):
                    collect_explicit_fcd(self.fcd(folder, times=(0., .25, .75)), {("vehicle", "car")}, {"car"})
            config = {"schema": "g35-native-full-day-v2", "seed": 350101, "split": "train",
                      "begin": 14400, "end": 50400, "native_step_s": .25}
            validate_config(config)
            with self.assertRaises(ValueError):
                validate_config({**config, "schema": "g35-native-full-day-v1"})

        def assert_filtered_equivalent(self, intervals, changes=None, missing_steps=(),
                                       empty_parent=False, expected_reasons=None):
            from g35b_extract import collect_explicit_fcd as full_collect
            with tempfile.TemporaryDirectory(prefix="g35c-equivalent-") as folder:
                path = Path(folder)/"full.xml.gz"
                # A stationary red-light segment stays inside the driving leg.
                xs = [0., 1., 1., 1., 2., 3., 4., 5., 5., 5., 6., 7., 8.]
                with gzip.open(path, "wt") as stream:
                    stream.write("<fcd-export>")
                    for j, x in enumerate(xs):
                        stream.write(f'<timestep time="{j*.25}">')
                        if j not in missing_steps:
                            attrs = {"id": "car", "type": "passenger1", "lane": "a_0",
                                     "x": str(x), "y": "2"}
                            for key, value in (changes or {}).get(j, {}).items():
                                if value is None:
                                    attrs.pop(key, None)
                                else:
                                    attrs[key] = value
                            stream.write('<vehicle '+" ".join(f'{k}="{v}"' for k, v in attrs.items())+'/>')
                        # Check full-stream counts include a recorded noncandidate.
                        stream.write('<vehicle id="other" type="passenger1" lane="b_0" x="2" y="3"/>')
                        stream.write("</timestep>")
                    stream.write("</fcd-export>")
                candidates = [{"entity_tag": "vehicle", "parent_base_id": "car",
                    "event_start": start, "event_end": end, "vtype": "passenger1"}
                    for start, end in intervals]
                if empty_parent:
                    candidates.append({"entity_tag": "vehicle", "parent_base_id": "unobserved",
                                       "event_start": .25, "event_end": .75, "vtype": "passenger1"})
                wanted = {(c["entity_tag"], c["parent_base_id"]) for c in candidates}
                ids = {"car", "other", "unobserved"}
                full, full_counts = full_collect(path, wanted, ids)
                slim, slim_counts = collect_explicit_fcd(path, wanted, ids, candidates)
                self.assertEqual(dict(full_counts), {key: slim_counts[key] for key in full_counts})
                for key in wanted:
                    self.assertEqual(full[key]["types"], slim[key]["types"])
                    self.assertEqual(full[key]["roads"].dtype, slim[key]["roads"].dtype)
                encoded_full, encoded_slim, accepted = [], [], []
                reasons = []
                for i, candidate in enumerate(candidates):
                    key = candidate["entity_tag"], candidate["parent_base_id"]
                    a, ar, ad = legs.cover(candidate, full[key])
                    b, br, bd = legs.cover(candidate, slim[key])
                    self.assertEqual(ar, br)
                    self.assertEqual(ad, bd)
                    reasons.append(br)
                    if ar is not None:
                        self.assertIsNone(a); self.assertIsNone(b)
                        continue
                    np.testing.assert_array_equal(a[0], b[0])
                    np.testing.assert_array_equal(a[1], b[1])
                    self.assertEqual(a[1].dtype, b[1].dtype)
                    for points, target in ((a[0], encoded_full), (b[0], encoded_slim)):
                        value = prepared.encode(points[:, :3])
                        value["xy"] = points[:, 1:3].copy()
                        value["native_speed"] = points[:, 3].copy()
                        target.append(value)
                    accepted.append({**candidate, "leg_id": "fixture-leg-"+str(i),
                        "partition": "train", "label": 2, "episode": 350101,
                        "ordinal": i, "line": "", "source_route_hash": "a",
                        "source_parent_route_hash": "b", "realized_route_hash": "c"})
                if expected_reasons is not None:
                    self.assertEqual(reasons, expected_reasons)
                left, right = arrays_from_rows(accepted, encoded_full), arrays_from_rows(accepted, encoded_slim)
                self.assertEqual(set(left), set(right))
                for key in left:
                    self.assertEqual(left[key].dtype, right[key].dtype)
                    np.testing.assert_array_equal(left[key], right[key])
                self.assertEqual(slim_counts["stored_candidate_parent_vehicle_records"]+
                    slim_counts["omitted_outside_candidate_intervals_records"],
                    slim_counts["candidate_parent_vehicle_records"])
                return full, slim, slim_counts

        def test_candidate_first_record_outside_keeps_full_lane_width(self):
            full, slim, counts = self.assert_filtered_equivalent([(.5, 1.)],
                changes={0: {"lane": "an_unused_but_longer_lane_name_0"}}, expected_reasons=[None])
            self.assertEqual(slim["vehicle", "car"]["values"][:, 0].tolist(), [0., .5, .75, 1.])
            self.assertEqual(counts["stored_first_records_outside_candidate_intervals"], 1)
            self.assertGreater(counts["omitted_outside_candidate_intervals_records"], 0)

        def test_candidate_no_observations_distinguishes_unobserved_parent(self):
            self.assert_filtered_equivalent([(4., 4.5)], empty_parent=True,
                expected_reasons=["fewer_than_two_FCD_observations", "parent_has_no_direct_FCD"])

        def test_candidate_type_change_outside_interval_is_still_rejected(self):
            self.assert_filtered_equivalent([(.25, .75)], changes={10: {"type": "other-type"}},
                expected_reasons=["invalid_FCD_values_or_inconsistent_vtype"])

        def test_candidate_multiple_activity_legs_and_redlight_stops_retained(self):
            _, slim, _ = self.assert_filtered_equivalent([(.25, .75), (2., 2.5)],
                                                        expected_reasons=[None, None])
            value = slim["vehicle", "car"]["values"]
            self.assertEqual(value[:, 0].tolist(), [0., .25, .5, .75, 2., 2.25, 2.5])
            self.assertEqual(value[1:4, 1].tolist(), [1., 1., 1.])

        def test_candidate_eps_boundaries_match_original_searchsorted(self):
            for intervals in ([((.25+legs.EPS), (.75-legs.EPS))],
                              [((.25+2*legs.EPS), (.75+2*legs.EPS))]):
                with self.subTest(intervals=intervals):
                    self.assert_filtered_equivalent(intervals)

        def test_candidate_overlapping_intervals_and_initial_point_not_duplicated(self):
            _, slim, counts = self.assert_filtered_equivalent([(0., .5), (.25, 1.)],
                                                              expected_reasons=[None, None])
            self.assertEqual(slim["vehicle", "car"]["values"][:, 0].tolist(), [0., .25, .5, .75, 1.])
            self.assertEqual(counts["stored_first_records_outside_candidate_intervals"], 0)

        def test_candidate_invalid_inside_values_and_gaps_preserve_diagnostics(self):
            self.assert_filtered_equivalent([(.25, 1.)], changes={2: {"x": "nan"}},
                                            expected_reasons=["invalid_FCD_values_or_inconsistent_vtype"])
            self.assert_filtered_equivalent([(.25, 1.)], missing_steps=(2,),
                                            expected_reasons=["FCD_gap_inside_leg"])

        def test_candidate_nonpositive_event_keeps_existing_rejection(self):
            self.assert_filtered_equivalent([(.5, .5)], expected_reasons=["nonpositive_event_duration"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecordingChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"complete": True, "component": "G35c extraction synthetic checks", "checks": result.testsRun,
                      "input_geometry_or_event_rules_changed": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--cnr-splits", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if any(getattr(args, field) is None for field in ("adoption", "source", "simulation", "cnr_splits", "output")):
            parser.error("--adoption, --source, --simulation, --cnr-splits and --output are required")
        extract(args)
