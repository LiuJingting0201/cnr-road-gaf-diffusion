"""Audit pinned public MoST demand, separating input plans from emitted routes.

The aggregate output contains no participant IDs. The optional compressed plan
index contains public input entity IDs and belongs only in the isolated run.
This program neither runs SUMO nor fits a model or opens a frozen test split.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


COMMIT = "b29b2f65f1096a9c69a601ec62a724815cb4a43f"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def children(path):
    path = Path(path)
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


def time_range(values):
    values = list(values)
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None}


def numerical(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def route_record(node):
    edges = node.get("edges", "").split()
    stops = [dict(x.attrib) for x in node.findall("stop")]
    return {"edges_hash": fingerprint(edges), "edge_count": len(edges),
            "origin_edge": edges[0] if edges else None,
            "destination_edge": edges[-1] if edges else None,
            "stop_count": len(stops), "stop_chain_hash": fingerprint(stops),
            "stops": stops}


def entity_record(node, filename):
    attributes = dict(node.attrib)
    records = [route_record(x) for x in node.findall("route")]
    alternatives = [route_record(x) for x in node.findall("routeDistribution/route")]
    stops = [dict(x.attrib) for x in node.findall("stop")]
    stages = []
    for child in node:
        if child.tag in ("walk", "ride", "personTrip", "stop", "access", "transport", "tranship"):
            stage = {"kind": child.tag, "attributes": dict(child.attrib)}
            if "edges" in stage["attributes"]:
                edges = stage["attributes"].pop("edges").split()
                stage["edges_hash"] = fingerprint(edges)
                stage["edge_count"] = len(edges)
                stage["origin_edge"] = edges[0] if edges else None
                stage["destination_edge"] = edges[-1] if edges else None
            stages.append(stage)
    return {"tag": node.tag, "source_file": filename, "attributes": attributes,
            "inline_routes": records, "route_alternatives": alternatives,
            "direct_stops": stops, "direct_stop_chain_hash": fingerprint(stops), "stages": stages}


def planned_route(record, routes):
    if len(record["inline_routes"]) == 1:
        return record["inline_routes"][0]
    return routes.get(record["attributes"].get("route", ""))


def realized_routes(path):
    if path is None:
        return {}
    result = {}
    for node in children(path):
        if node.tag != "vehicle" or not node.get("id"):
            continue
        candidates = node.findall("routeDistribution/route") or node.findall("route")
        if candidates:
            result[node.get("id")] = {"line": node.get("line", ""),
                                      "route": route_record(candidates[-1]),
                                      "reroutes": max(0, len(candidates)-1)}
    return result


def observed_mapping(fcd, vehroute, entities, flows, routes):
    if fcd is None:
        return {"status": "not_checked_no_observed_FCD", "scope": "Input plan inventory only; generated IDs have not been assumed to map."}
    realized = realized_routes(vehroute)
    observed = {}
    nested_riders = 0
    for step in children(fcd):
        if step.tag != "timestep":
            continue
        for node in step:
            if node.tag == "vehicle":
                nested_riders += sum(1 for _ in node.iter("person"))
            if node.tag not in ("vehicle", "person") or not node.get("id"):
                continue
            if node.tag == "person" and node.get("vehicle", "") not in ("", "NULL", "None"):
                continue
            key = node.tag + ":" + node.get("id")
            row = observed.setdefault(key, {"tag": node.tag, "id": node.get("id"), "types": set()})
            row["types"].add(node.get("type", ""))
    counts, by_type, line_counts = Counter(), defaultdict(Counter), Counter()
    planned_hashes, realized_hashes = set(), set()
    exact_plan_route_same = exact_plan_route_different = route_comparisons = 0
    for key, row in observed.items():
        record = entities.get(key)
        status = "exact_declared_entity" if record is not None else "unmapped"
        if record is None and row["tag"] == "vehicle":
            # A numerical suffix alone is only a candidate. Verify the candidate
            # against actual emitted line+route or origin/destination before
            # claiming a source-flow mapping; otherwise report it unverified.
            prefix, separator, suffix = row["id"].rpartition(".")
            candidate = flows.get(prefix) if separator and suffix.isdecimal() else None
            if candidate is not None:
                record = candidate
                status = "flow_namespace_candidate_unverified"
                actual = realized.get(row["id"])
                plan = planned_route(record, routes)
                if actual is not None:
                    if (plan is not None and plan["edges_hash"] == actual["route"]["edges_hash"]
                            and record["attributes"].get("line", "") == actual["line"]):
                        status = "flow_verified_by_emitted_line_and_route"
                    elif (record["attributes"].get("from") == actual["route"]["origin_edge"]
                          and record["attributes"].get("to") == actual["route"]["destination_edge"]
                          and record["attributes"].get("from") is not None):
                        status = "flow_verified_by_emitted_endpoints"
        counts[status] += 1
        for typename in row["types"]:
            by_type[typename][status] += 1
        if record is not None and status != "flow_namespace_candidate_unverified":
            line = record["attributes"].get("line")
            if line:
                line_counts[line] += 1
            plan = planned_route(record, routes)
            actual = realized.get(row["id"]) if row["tag"] == "vehicle" else None
            if plan is not None:
                planned_hashes.add(plan["edges_hash"])
            if plan is not None and actual is not None:
                route_comparisons += 1
                if plan["edges_hash"] == actual["route"]["edges_hash"]:
                    exact_plan_route_same += 1
                else:
                    exact_plan_route_different += 1
        actual = realized.get(row["id"]) if row["tag"] == "vehicle" else None
        if actual is not None:
            realized_hashes.add(actual["route"]["edges_hash"])
    return {"status": "checked", "observed_direct_entities": len(observed),
            "mapping_counts": dict(counts), "mapping_by_actual_type": {k: dict(v) for k, v in sorted(by_type.items())},
            "source_lines_observed_verified_mapping": dict(sorted(line_counts.items())),
            "unique_input_route_fingerprints_observed": len(planned_hashes),
            "unique_emitted_route_fingerprints_observed": len(realized_hashes),
            "input_vs_emitted_route_comparisons": route_comparisons,
            "same_input_and_emitted_route": exact_plan_route_same,
            "different_input_and_emitted_route": exact_plan_route_different,
            "nested_rider_observations_ignored": nested_riders,
            "scope": "Observed namespace matching is audited empirically. A changed emitted route is not a new input demand. Person activities are not single walking trips.",
            "fcd_sha256": sha256(fcd), "emitted_routes_sha256": sha256(vehroute) if vehroute else None}


def audit(args):
    root = args.source
    if args.output.exists() or (args.plans and args.plans.exists()):
        raise FileExistsError("Choose unused audit/plan output paths; existing results are preserved")
    manifest_path = root/"complete.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or manifest.get("commit") != COMMIT:
        raise ValueError("Pinned source manifest is incomplete or has an unexpected commit")
    for name, record in manifest["files"].items():
        if sha256(root/name) != record["sha256"]:
            raise ValueError(f"Source file changed: {name}")
    config = ET.parse(root/"scenario/most.sumocfg").getroot()
    settings = {section.tag: {element.tag: element.get("value") for element in section} for section in config}
    route_files = [root/"scenario"/name for name in settings["input"]["route-files"].split(",")]
    additional_files = [root/"scenario"/name for name in settings["input"]["additional-files"].split(",")]
    entities, flows, routes, types, distributions = {}, {}, {}, {}, {}
    file_summaries = {}
    bus_stops, train_stops, parking_areas = set(), set(), set()
    all_stops = []
    for path in additional_files + route_files:
        tags, entity_types, stage_types = Counter(), Counter(), Counter()
        numeric_departures, numeric_begins, numeric_ends = [], [], []
        nonnumeric_departures = Counter()
        for node in children(path):
            tags[node.tag] += 1
            if node.tag == "busStop":
                (train_stops if "trainstops" in path.name else bus_stops).add(node.get("id"))
            elif node.tag == "trainStop":
                train_stops.add(node.get("id"))
            elif node.tag == "parkingArea":
                parking_areas.add(node.get("id"))
            for typename in ([node] if node.tag == "vType" else node.findall("vType") if node.tag == "vTypeDistribution" else []):
                name, vclass = typename.get("id"), typename.get("vClass")
                if name and vclass:
                    if name in types and types[name]["vClass"] != vclass:
                        raise ValueError(f"Conflicting vType definition {name}")
                    types[name] = {"vClass": vclass, "attributes": dict(typename.attrib),
                                   "parameters": {x.get("key"): x.get("value") for x in typename.findall("param")},
                                   "source_file": path.name}
            if node.tag == "vTypeDistribution":
                distributions[node.get("id")] = [{"id": x.get("id"), "probability": x.get("probability")}
                                                  for x in node.findall("vType")]
            if node.tag == "route" and node.get("id"):
                if node.get("id") in routes:
                    raise ValueError("Duplicate standalone route ID")
                routes[node.get("id")] = route_record(node)
                all_stops.extend(routes[node.get("id")]["stops"])
            if node.tag not in ("vehicle", "trip", "person", "flow", "personFlow"):
                continue
            record = entity_record(node, path.name)
            entity_types[node.get("type", "<default>")] += 1
            stage_types.update(x["kind"] for x in record["stages"])
            for field, target in (("depart", numeric_departures), ("begin", numeric_begins), ("end", numeric_ends)):
                value = node.get(field)
                if value is not None:
                    number = numerical(value)
                    if number is not None:
                        target.append(number)
                    elif field == "depart":
                        nonnumeric_departures[value] += 1
            if node.tag in ("flow", "personFlow"):
                if node.get("id") in flows:
                    raise ValueError("Duplicate flow ID")
                flows[node.get("id")] = record
            else:
                tag = "vehicle" if node.tag in ("vehicle", "trip") else "person"
                key = tag+":"+node.get("id")
                if key in entities:
                    raise ValueError("Duplicate declared entity ID")
                entities[key] = record
        file_summaries[path.name] = {"top_level_tag_counts": dict(tags), "entity_type_or_distribution_counts": dict(entity_types),
                                    "activity_or_stop_stage_counts": dict(stage_types),
                                    "numeric_departure_times_s": time_range(numeric_departures),
                                    "numeric_flow_begin_times_s": time_range(numeric_begins),
                                    "numeric_flow_end_times_s": time_range(numeric_ends),
                                    "nonnumeric_departure_counts": dict(nonnumeric_departures)}
    line_summary = {}
    for record in flows.values():
        attributes = record["attributes"]
        if not attributes.get("line"):
            continue
        route = planned_route(record, routes)
        line_summary[attributes["line"]] = {"type_or_distribution": attributes.get("type"),
            "begin": attributes.get("begin"), "end": attributes.get("end"), "period": attributes.get("period"),
            "planned_edge_count": route["edge_count"] if route else None,
            "planned_stop_count": route["stop_count"] if route else None}
    input_routes = [planned_route(row, routes) for row in entities.values()]
    missing_route_refs = sum(bool(row["attributes"].get("route")) and planned_route(row, routes) is None
                             for row in list(entities.values())+list(flows.values()))
    report = {"complete": True, "commit": COMMIT, "source_manifest_sha256": sha256(manifest_path),
              "scope": "Aggregate inventory of public native input demand; not a replay identity claim for the historical CNR export",
              "script_sha256": sha256(Path(__file__)), "native_configuration": settings,
              "file_summaries": file_summaries, "vtype_taxonomy": types,
              "vtype_distributions": distributions, "standalone_route_count": len(routes),
              "declared_entity_count": len(entities), "flow_count": len(flows),
              "line_flows": line_summary, "bus_stop_locations": len(bus_stops),
              "train_stop_locations": len(train_stops), "parking_area_count": len(parking_areas),
              "standalone_route_stop_count": len(all_stops),
              "standalone_route_stop_duration_values": dict(Counter(x.get("duration", "<missing>") for x in all_stops)),
              "standalone_route_stop_until_values_s": time_range(numerical(x["until"]) for x in all_stops if numerical(x.get("until")) is not None),
              "explicit_entity_planned_route_count": sum(x is not None for x in input_routes),
              "explicit_entity_unique_route_fingerprints": len({x["edges_hash"] for x in input_routes if x is not None}),
              "unresolved_standalone_route_references": missing_route_refs,
              "observed_source_mapping": observed_mapping(args.fcd, args.routes, entities, flows, routes),
              "interpretation": ["Basic vehicle definitions alone omit highway-specific vTypes; all route files were scanned.",
                  "The same public base entity can recur under different simulation seeds; a different final rerouted path does not create an independent input demand.",
                  "Bus line coverage describes shared operating lines, not unseen-line validation.",
                  "An input person can contain multiple walk and ride stages; its endpoints are not one continuous walking trip.",
                  "Stops inside stand-alone referenced routes use until times relative to vehicle departure; the native schedules were not rewritten.",
                  "Native type uber has passenger vClass but denotes on-demand transport; taxonomy overrides must use explicit source types."]}
    if args.plans:
        args.plans.parent.mkdir(parents=True, exist_ok=True)
        if args.plans.exists():
            raise FileExistsError("Do not overwrite the isolated public-demand plan index")
        temporary = args.plans.with_suffix(".tmp.gz")
        with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output:
            output.write(json.dumps({"source_commit": COMMIT, "entities": entities, "flows": flows, "routes": routes},
                                    sort_keys=True, separators=(",", ":")).encode())
        temporary.replace(args.plans)
        report["isolated_plan_index"] = {"sha256": sha256(args.plans), "bytes": args.plans.stat().st_size,
                                        "scope": "Contains public source IDs; server/local only, not committed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing demand audit; choose a new output filename")
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(args.output)
    print(json.dumps({"complete": True, "declared_entities": len(entities), "flows": len(flows),
                      "bus_stops": len(bus_stops), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("g30-data/source"))
    parser.add_argument("--output", type=Path, default=Path("g30-data/demand-audit.json"))
    parser.add_argument("--plans", type=Path, help="Optional isolated public-entity plan index (.json.gz), never committed")
    parser.add_argument("--fcd", type=Path, help="Optional completed pilot FCD to verify actual input-to-output ID mapping")
    parser.add_argument("--routes", type=Path, help="Emitted pilot vehroute, used only for mapping checks")
    audit(parser.parse_args())
