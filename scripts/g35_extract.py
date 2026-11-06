"""Extract one preassigned full-day G35 episode into complete vehicle legs.

Only one simulation realization is allowed for each parent, decided from its
identity before any outcomes are read. This is new native-demand supervision,
not a replay-invariance check or a prospective generation benchmark. Original
test coordinates are never loaded. Per-parent metadata remains beside the raw
dataset; the completion receipt contains aggregates and hashes only.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
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
    if (config.get("schema") != "g35-native-full-day-v1"
            or config.get("seed") not in SEED_SPLIT
            or config.get("split") != SEED_SPLIT[config["seed"]]
            or config.get("begin") != 14400 or config.get("end") != 50400
            or config.get("native_step_s") != .25):
        raise ValueError("Not a declared G35 full-day simulation")


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
        from g35_simulate import verify_receipt, verify_source
        config, simulation_receipt = verify_receipt(simulation)
        validate_config(config)
        verify_source(source_path)
        if config["source_manifest_sha256"] != legs.digest(source_path / "complete.json"):
            raise ValueError("Simulation and extractor use different source manifests")
        seed, split = int(config["seed"]), str(config["split"])
        partitions = original_id_partitions(args.cnr_splits)
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
        observed, fcd_counts = legs.collect_fcd(simulation / "fcd.xml.gz", wanted)
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
        helper_names = ("g35_extract.py", "g35_simulate.py", "g31_legs.py", "g33_prepare.py",
                        "g30_extract.py", "g30_prepare.py", "g30_demand_audit.py", "g29_prepare.py")
        report = {
            "complete": True, "schema": prepared.SCHEMA, "provenance_schema": PROVENANCE_SCHEMA,
            "classes": CLASSES, "episode": seed, "split": split, "seconds": time.monotonic() - started,
            "scope": "Native full-day complete vehicle legs; one identity-preassigned realization per parent",
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
                "simulation_receipt_sha256": legs.digest(simulation / "simulation-complete.json"),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--cnr-splits", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        from g35_extract_checks import main
        main()
    else:
        if any(getattr(args, field) is None for field in ("source", "simulation", "cnr_splits", "output")):
            parser.error("--source, --simulation, --cnr-splits and --output are required")
        extract(args)
