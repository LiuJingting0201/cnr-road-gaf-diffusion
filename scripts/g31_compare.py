"""Stream G30/G31 FCD comparison and output-schema audit; no trajectory export."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from g31_replay import (build_command, digest, json_hash, save, validate_pilot,
                        verify_receipt)


COMMON = ("x", "y", "type", "speed", "lane", "edge", "vehicle")
ENTITY_TAGS = {"vehicle", "person", "container"}


def children(path):
    """Retain at most one root child (one FCD timestep), including nested riders."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
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


class FcdStream:
    def __init__(self, path):
        self.path = path
        self.hash = hashlib.sha256()
        self.counts = Counter()
        self.fields = defaultdict(Counter)
        self.dt = Counter()
        self.first = self.last = None
        self.maximum_records_per_step = 0

    def __iter__(self):
        for step in children(self.path):
            if step.tag != "timestep":
                raise ValueError("Unexpected top-level FCD XML element")
            current = float(step.get("time", "nan"))
            if not math.isfinite(current) or (self.last is not None and current <= self.last):
                raise ValueError("FCD times must be finite and strictly increasing")
            if self.last is None:
                self.first = current
            else:
                self.dt[f"{current - self.last:.8f}"] += 1
            self.last = current
            self.counts["timesteps"] += 1
            records = {}

            def visit(node, ancestry=()):
                if node.tag not in ENTITY_TAGS:
                    raise ValueError("Unexpected FCD entity XML element")
                ident = node.get("id")
                if not ident:
                    raise ValueError("FCD entity has no ID")
                key = (node.tag, ident)
                if key in records:
                    raise ValueError("Duplicate (tag, ID) inside a single FCD timestep")
                attrs = dict(node.attrib)
                for name in ("x", "y", "speed", "pos"):
                    if name in attrs and not math.isfinite(float(attrs[name])):
                        raise ValueError("FCD has a non-finite numeric attribute")
                for required in ("x", "y"):
                    if required not in attrs:
                        raise ValueError("FCD entity lacks coordinates")
                records[key] = (attrs, ancestry)
                self.counts[node.tag] += 1
                self.counts["nested_" + node.tag] += bool(ancestry)
                self.fields[node.tag].update(attrs.keys())
                for child in node:
                    visit(child, ancestry + (key,))

            for node in step:
                visit(node)
            self.maximum_records_per_step = max(self.maximum_records_per_step, len(records))
            for key in sorted(records):
                attrs, ancestry = records[key]
                payload = [current, key, ancestry, [attrs.get(field) for field in COMMON]]
                self.hash.update(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
            # Include empty timesteps and time support in the semantic digest.
            self.hash.update(json.dumps(["end_step", current], separators=(",", ":")).encode() + b"\n")
            yield current, records

    def summary(self):
        return {"counts": dict(self.counts), "attribute_counts": dict(self.fields),
                "time_min": self.first, "time_max": self.last, "dt_counts": dict(self.dt),
                "maximum_records_per_step": self.maximum_records_per_step,
                "common_fields_sha256": self.hash.hexdigest()}


def compare_fcd(baseline, replay):
    streams = [FcdStream(baseline), FcdStream(replay)]
    iterators = [iter(stream) for stream in streams]
    steps = [next(iterator, None) for iterator in iterators]
    counts, fields = Counter(), Counter()
    examples = []

    def note(kind, timestamp, tag=None, changed=None):
        if len(examples) < 20:
            # Never include entity IDs, road IDs, coordinates or attribute values.
            examples.append({"kind": kind, "time": timestamp, "tag": tag,
                             "changed_field_names": changed or []})

    while any(step is not None for step in steps):
        a, b = steps
        if b is None or (a is not None and a[0] < b[0]):
            counts["baseline_only_timesteps"] += 1
            counts["baseline_only_records"] += len(a[1])
            note("baseline_only_timestep", a[0])
            steps[0] = next(iterators[0], None)
            continue
        if a is None or b[0] < a[0]:
            counts["replay_only_timesteps"] += 1
            counts["replay_only_records"] += len(b[1])
            note("replay_only_timestep", b[0])
            steps[1] = next(iterators[1], None)
            continue
        counts["matched_timesteps"] += 1
        for key in sorted(a[1].keys() | b[1].keys()):
            if key not in b[1]:
                counts["baseline_only_records"] += 1
                note("baseline_only_record", a[0], key[0])
                continue
            if key not in a[1]:
                counts["replay_only_records"] += 1
                note("replay_only_record", a[0], key[0])
                continue
            counts["matched_records"] += 1
            left, parent_left = a[1][key]
            right, parent_right = b[1][key]
            changed = [field for field in COMMON if left.get(field) != right.get(field)]
            if parent_left != parent_right:
                changed.append("XML_parent_entity")
            if changed:
                counts["changed_records"] += 1
                fields.update(changed)
                note("changed_record", a[0], key[0], changed)
        steps = [next(iterator, None) for iterator in iterators]
    bad = ("baseline_only_timesteps", "replay_only_timesteps", "baseline_only_records",
           "replay_only_records", "changed_records")
    equal = counts["matched_timesteps"] > 0 and counts["matched_records"] > 0 and not any(counts[k] for k in bad)
    summaries = [stream.summary() for stream in streams]
    if equal and summaries[0]["common_fields_sha256"] != summaries[1]["common_fields_sha256"]:
        raise ValueError("Equality count and independent semantic digests disagree")
    return {"complete": True, "equal": equal, "counts": dict(counts),
            "changed_field_counts": dict(fields), "first_mismatch_summaries": examples,
            "baseline": summaries[0], "replay": summaries[1],
            "policy": "Exact XML attribute strings for common fields; order-independent within timestep; parent relation included",
            "memory": "One timestep per input; no whole-file trajectory or ID collection"}


def schema_audit(path):
    counts, attributes = Counter(), defaultdict(Counter)
    numeric = defaultdict(Counter)
    structural, parking = Counter(), Counter()
    numeric_fields = {"started", "ended", "depart", "arrival", "duration", "routeLength", "priorEdgesLength"}

    def visit(node, parents=()):
        key = "/".join(parents + (node.tag,))
        counts[key] += 1
        attributes[key].update(node.attrib.keys())
        for name in numeric_fields & node.attrib.keys():
            try:
                value = float(node.get(name))
            except ValueError:
                numeric[key + "/" + name]["non_numeric"] += 1
                continue
            state = "non_finite" if not math.isfinite(value) else ("negative" if value < 0 else "nonnegative")
            numeric[key + "/" + name][state] += 1
        if "edges" in node.attrib:
            edges = node.get("edges").split()
            structural["nodes_with_edges"] += 1
            structural["nodes_with_repeated_edge"] += len(set(edges)) != len(edges)
            if "exitTimes" in node.attrib:
                exits = [float(value) for value in node.get("exitTimes").split()]
                structural["nodes_with_exitTimes"] += 1
                structural["exitTimes_edge_count_mismatches"] += len(exits) != len(edges)
                structural["exitTimes_negative_values"] += sum(value < 0 for value in exits)
                structural["exitTimes_nonfinite_values"] += sum(not math.isfinite(value) for value in exits)
                observed = [value for value in exits if value >= 0]
                structural["exitTimes_decreasing_nodes"] += any(a > b for a, b in zip(observed, observed[1:]))
        if node.tag in ("stop", "stopinfo"):
            structural["stop_nodes_with_both_times"] += "started" in node.attrib and "ended" in node.attrib
            structural["stop_nodes_with_priorEdges"] += "priorEdges" in node.attrib
            if "parking" in node.attrib:
                value = node.get("parking").lower()
                parking[{"0": "false", "false": "false", "1": "true", "true": "true"}.get(value, "unrecognised")] += 1
        for child in node:
            visit(child, parents + (node.tag,))

    for node in children(path):
        visit(node)
    return {"element_counts": dict(counts), "attribute_counts": dict(attributes),
            "numeric_states": dict(numeric), "structure": dict(structural),
            "parking_boolean_counts": dict(parking)}


def run(args):
    baseline, replay, output = [path.resolve() for path in (args.baseline, args.replay, args.output)]
    for protected in (baseline, replay):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Comparison report must be outside both immutable simulation directories")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        original, original_receipt = verify_receipt(baseline)
        enhanced, enhanced_receipt = verify_receipt(replay)
        validate_pilot(original)
        if enhanced["baseline_config_sha256"] != digest(baseline / "config.json"):
            raise ValueError("Replay belongs to a different baseline configuration")
        if enhanced["baseline_receipt_sha256"] != digest(baseline / "simulation-complete.json"):
            raise ValueError("Baseline receipt changed after replay preflight")
        if enhanced["baseline_command_sha256"] != json_hash(original["command"]):
            raise ValueError("Baseline command changed after replay preflight")
        expected = build_command(original, replay, Path(enhanced["source"]), Path(enhanced["command"][0]))
        if enhanced["command"] != expected or enhanced["version"] != original["version"]:
            raise ValueError("Replay command differs beyond the allowed output-only changes")
        fcd = compare_fcd(baseline / "fcd.xml.gz", replay / "fcd.xml.gz")
        save(output / "fcd-comparison.json", fcd)
        field_audit = {
            name: {"baseline": schema_audit(baseline / name), "replay": schema_audit(replay / name)}
            for name in ("vehroute.xml.gz", "tripinfo.xml", "stops.xml")
        }
        save(output / "field-audit.json", field_audit)
        route = field_audit["vehroute.xml.gz"]["replay"]["structure"]
        vehicle_count = fcd["replay"]["counts"].get("vehicle", 0)
        pos_count = fcd["replay"]["attribute_counts"].get("vehicle", {}).get("pos", 0)
        checks = {
            "common_fcd_exactly_equal": fcd["equal"],
            "full_expected_time_support": all(
                fcd[which]["counts"].get("timesteps") == 20464
                and fcd[which]["time_min"] == 14400 and fcd[which]["time_max"] == 19515.75
                and set(fcd[which]["dt_counts"]) == {"0.25000000"}
                for which in ("baseline", "replay")),
            "vehicle_pos_present": vehicle_count > 0 and pos_count == vehicle_count,
            "exit_times_present_and_consistent": route.get("nodes_with_exitTimes", 0) > 0
                and not any(route.get(name, 0) for name in ("exitTimes_edge_count_mismatches", "exitTimes_nonfinite_values", "exitTimes_decreasing_nodes")),
            "stop_event_times_present": route.get("stop_nodes_with_both_times", 0) > 0,
            "stop_prior_edges_present": route.get("stop_nodes_with_priorEdges", 0) > 0,
        }
        report = {
            "complete": all(checks.values()), "checks": checks,
            "seconds": time.monotonic() - started,
            "script_sha256": digest(Path(__file__)),
            "baseline_receipt_sha256": digest(baseline / "simulation-complete.json"),
            "replay_receipt_sha256": digest(replay / "simulation-complete.json"),
            "report_files": {name: {"sha256": digest(output / name)}
                             for name in ("fcd-comparison.json", "field-audit.json")},
            "scope": "Output invariance and XML fields only; no leg completeness, generation or GAF efficacy claim",
        }
        if not report["complete"]:
            save(output / "comparison-failed.json", report)
            raise ValueError("G31 comparison/schema checks failed; aggregate diagnostics preserved")
        save(output / "comparison-complete.json", report)
        print("G31 FCD invariance and output-field checks passed", flush=True)
    except BaseException as error:
        if not (output / "comparison-failed.json").exists():
            save(output / "comparison-failed.json", {
                "complete": False, "seconds": time.monotonic() - started, "error": repr(error),
            })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("../g30-job/g30-data/native/pilot/300001"))
    parser.add_argument("--replay", type=Path, default=Path("g31-data/replay/pilot/300001"))
    parser.add_argument("--output", type=Path, default=Path("g31-data/compare/pilot-300001"))
    run(parser.parse_args())
