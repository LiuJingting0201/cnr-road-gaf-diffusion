"""Event-defined native MoST legs: coverage audit, not generation readiness.

Coordinates, entity identifiers and per-leg route metadata remain in the isolated
output directory. The JSON completion report is aggregate-only. Public input
plans are kept separate from realized vehroute/FCD audit information throughout.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from g30_extract import children, types_from_files, VCLASS_LABEL, SOURCE_TYPE_LABEL
from g30_prepare import entity_partition, original_id_partitions
from g30_demand_audit import fingerprint


CLASSES = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
DT = .25
EPS = 1e-7


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value, default=-1.):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def is_true(value):
    return str(value).lower() in ("1", "true")


def edge_of_lane(lane):
    prefix, separator, suffix = str(lane or "").rpartition("_")
    return prefix if separator and suffix.isdecimal() else None


def location_key(stop):
    for field in ("parkingArea", "busStop", "trainStop"):
        if stop.get(field):
            return field, stop[field]
    if stop.get("lane"):
        return "lane", stop["lane"]
    return None


def activity_kind(stop):
    # This uses scheduled source semantics, never observed low-speed thresholds.
    if stop.get("parkingArea"):
        return "planned_parking_area_activity"
    if is_true(stop.get("parking")):
        return "planned_parking_activity"
    return None


def route_data(node):
    edges = node.get("edges", "").split()
    exits = [number(value) for value in node.get("exitTimes", "").split()]
    return {"attributes": {k: v for k, v in node.attrib.items() if k not in ("edges", "exitTimes")},
            "edges": edges, "edges_hash": fingerprint(edges), "exit_times": exits,
            "stops": [dict(stop.attrib) for stop in node.findall("stop")]}


def node_plan(node):
    return {"tag": node.tag, "attributes": dict(node.attrib),
            "routes": [route_data(x) for x in node.findall("route")],
            "alternatives": [route_data(x) for x in node.findall("routeDistribution/route")],
            "stops": [dict(x.attrib) for x in node.findall("stop")],
            "stages": [{"tag": x.tag, "attributes": dict(x.attrib), "route": route_data(x)}
                       for x in node if x.tag in ("walk", "ride", "stop", "access", "personTrip")]}


def source_inventory(source, wanted):
    import xml.etree.ElementTree as ET
    receipt = json.loads((source/"complete.json").read_text())
    if not receipt.get("complete"):
        raise ValueError("Public source cache incomplete")
    cfg = ET.parse(source/"scenario/most.sumocfg").getroot()
    inputs = cfg.find("input")
    route_files = [source/"scenario"/x for x in inputs.find("route-files").get("value").split(",")]
    additions = [source/"scenario"/x for x in inputs.find("additional-files").get("value").split(",")]
    for path in route_files + additions:
        name = str(path.relative_to(source))
        if digest(path) != receipt["files"][name]["sha256"]:
            raise ValueError("Pinned source content changed")
    types, distributions = types_from_files(route_files + additions)
    entities, flows, routes, locations = {}, {}, {}, {}
    for path in additions + route_files:
        for node in children(path):
            if node.tag in ("busStop", "trainStop", "parkingArea"):
                locations[(node.tag, node.get("id"))] = dict(node.attrib)
                if node.tag == "busStop" and "trainstops" in path.name:
                    locations[("trainStop", node.get("id"))] = dict(node.attrib)
            elif node.tag == "route" and node.get("id"):
                routes[node.get("id")] = route_data(node)
            elif node.tag in ("flow", "personFlow"):
                flows[node.get("id")] = node_plan(node)
            elif node.tag in ("vehicle", "trip", "person"):
                tag = "vehicle" if node.tag in ("vehicle", "trip") else "person"
                key = (tag, node.get("id"))
                if key in wanted:
                    entities[key] = node_plan(node)
    return {"entities": entities, "flows": flows, "routes": routes, "locations": locations,
            "types": types, "distributions": distributions, "manifest": receipt}


def parent_events(folder, partitions):
    parents, excluded = {}, Counter()
    for node in children(folder/"tripinfo.xml"):
        if node.tag not in ("tripinfo", "personinfo"):
            continue
        base = node.get("id")
        partition = entity_partition(base, partitions)
        if partition == "test":
            excluded["original_test_templates_excluded_before_leg_extraction"] += 1
            continue
        tag = "vehicle" if node.tag == "tripinfo" else "person"
        key = tag, base
        if key in parents:
            raise ValueError("Duplicate parent tripinfo")
        parents[key] = {"tag": tag, "base_id": base, "partition": partition,
                        "tripinfo": dict(node.attrib),
                        "stages": [{"tag": x.tag, "attributes": dict(x.attrib)} for x in node
                                   if x.tag in ("walk", "ride", "stop", "access")],
                        "stop_events": []}
    for node in children(folder/"vehroute.xml.gz"):
        key = node.tag, node.get("id")
        if key not in parents:
            continue
        if "emitted" in parents[key]:
            raise ValueError("Duplicate emitted parent route")
        parents[key]["emitted"] = node_plan(node)
    for node in children(folder/"stops.xml"):
        key = "vehicle", node.get("id")
        if node.tag == "stopinfo" and key in parents:
            parents[key]["stop_events"].append(dict(node.attrib))
    return parents, excluded


def primary_route(plan, source):
    if len(plan["routes"]) == 1:
        return plan["routes"][0]
    return source["routes"].get(plan["attributes"].get("route", ""))


def link_source(parent, source):
    key = parent["tag"], parent["base_id"]
    if key in source["entities"]:
        return source["entities"][key], "exact_source_entity"
    if parent["tag"] != "vehicle":
        return None, "no_exact_source_person"
    prefix, separator, suffix = parent["base_id"].rpartition(".")
    plan = source["flows"].get(prefix) if separator and suffix.isdecimal() else None
    if plan is None:
        return None, "no_source_flow_namespace"
    emitted = parent.get("emitted", {})
    realized = emitted.get("alternatives") or emitted.get("routes") or []
    candidate = primary_route(plan, source)
    attributes = plan["attributes"]
    if realized:
        actual = realized[-1]
        if (candidate and actual["edges_hash"] == candidate["edges_hash"]
                and attributes.get("line", "") == emitted["attributes"].get("line", "")):
            return plan, "source_flow_confirmed_by_line_and_route"
        edges = actual["edges"]
        if edges and attributes.get("from") == edges[0] and attributes.get("to") == edges[-1]:
            return plan, "source_flow_confirmed_by_endpoints"
    return None, "source_flow_namespace_unverified"


def stop_bound(stop, locations):
    location = locations.get(location_key(stop), {})
    lane = stop.get("lane") or location.get("lane")
    return {"edge": edge_of_lane(lane), "lane": lane,
            "position": stop.get("endPos") or stop.get("pos") or location.get("endPos"),
            "semantic": activity_kind(stop), "source_stop": stop}


def source_leg_plan(plan, source, ordinal, final_leg, kind, walk=None):
    result = {"provenance": "public_input_only", "kind": kind,
              "source_attributes": plan["attributes"], "condition_ready": False,
              "condition_scope": "Topology, permission and reconstruction not checked"}
    if walk is not None:
        result["edges"] = walk["route"]["edges"]
        result["stage_attributes"] = walk["attributes"]
        result["status"] = "explicit_source_walk" if result["edges"] else "source_walk_without_explicit_edges"
        return result
    route = primary_route(plan, source)
    if route is None:
        result.update(status="source_od_or_ambiguous_route_only", edges=[],
                      origin_edge=plan["attributes"].get("from"), destination_edge=plan["attributes"].get("to"))
        return result
    edges = route["edges"]
    source_stops = route["stops"] + plan["stops"]
    result["parent_edges_hash"] = route["edges_hash"]
    result["parent_route_edge_count"] = len(edges)
    if kind == "bus_service":
        result.update(status="complete_scheduled_source_service", edges=edges, planned_stops=source_stops)
        return result
    activities = [x for x in source_stops if activity_kind(x)]
    bounds = [stop_bound(x, source["locations"]) for x in activities]
    candidates = [[i for i, edge in enumerate(edges) if edge == bound["edge"]] for bound in bounds]
    # Source-only monotone constraints retain ambiguity on repeated edges. Never
    # use FCD or realized exitTimes to select the prediction's source path.
    lower, previous = [], 0
    for indices in candidates:
        possible = [i for i in indices if i >= previous]
        lower.append(possible[0] if possible else None)
        previous = possible[0] if possible else len(edges)
    upper, following = [], len(edges)-1
    for indices in reversed(candidates):
        possible = [i for i in indices if i <= following]
        upper.append(possible[-1] if possible else None)
        following = possible[-1] if possible else -1
    upper.reverse()
    feasible = [[i for i in indices if lo is not None and hi is not None and lo <= i <= hi]
                for indices, lo, hi in zip(candidates, lower, upper)]
    first = [0] if ordinal == 0 else feasible[ordinal-1] if ordinal-1 < len(feasible) else []
    last = [len(edges)-1] if final_leg else feasible[ordinal] if ordinal < len(feasible) else []
    result["source_start_index_candidates"] = first
    result["source_end_index_candidates"] = last
    result["planned_activity_boundaries"] = bounds
    if len(first) == len(last) == 1 and first[0] <= last[0]:
        result.update(status="unique_source_activity_leg_slice", edges=edges[first[0]:last[0]+1])
    else:
        result.update(status="ambiguous_source_activity_leg_slice", edges=[])
    return result


def leg(parent, ordinal, start, end, start_reason, end_reason, kind, **extra):
    return {"parent_base_id": parent["base_id"], "entity_tag": parent["tag"],
            "partition": parent["partition"], "label": parent["label"], "vtype": parent["vtype"],
            "ordinal": ordinal, "kind": kind, "event_start": start, "event_end": end,
            "start_reason": start_reason, "end_reason": end_reason, **extra}


def candidate_legs(parent, source):
    rejected, candidates = Counter(), []
    plan, link_status = link_source(parent, source)
    parent["source_link_status"] = link_status
    if plan is None:
        return candidates, Counter({"source_plan_unresolved": 1})
    parent["source_plan"] = plan
    info = parent["tripinfo"]
    depart, arrival = number(info.get("depart")), number(info.get("arrival"))
    emitted = parent.get("emitted")
    if not emitted:
        return candidates, Counter({"emitted_parent_route_missing": 1})
    if parent["tag"] == "person":
        times = [x["attributes"] for x in parent["stages"] if x["tag"] == "walk"]
        emitted_walks = [x for x in emitted["stages"] if x["tag"] == "walk"]
        planned_walks = [x for x in plan["stages"] if x["tag"] == "walk"]
        if len(times) > len(emitted_walks):
            return candidates, Counter({"walk_event_route_stage_count_mismatch": 1})
        for ordinal, event in enumerate(times):
            start, end = number(event.get("depart")), number(event.get("arrival"))
            if start < 0 or end < start:
                rejected["walk_stage_not_completed"] += 1
                continue
            actual = emitted_walks[ordinal]
            started, ended = number(actual["attributes"].get("started")), number(actual["attributes"].get("ended"))
            if started < 0 or ended < 0:
                rejected["enhanced_walk_event_times_missing"] += 1
                continue
            if abs(start-started) > DT+EPS or abs(end-ended) > DT+EPS:
                rejected["walk_tripinfo_route_event_disagreement"] += 1
                continue
            if len(planned_walks) != len(emitted_walks):
                rejected["source_walk_stage_alignment_ambiguous"] += 1
                continue
            source_path = source_leg_plan(plan, source, ordinal, True, "walk_stage", walk=planned_walks[ordinal])
            candidates.append(leg(parent, ordinal, start, end, "walk_depart", "walk_arrival", "walk_stage",
                                  source_plan=source_path, emitted_walk=actual,
                                  source_link_status=link_status))
        return candidates, rejected
    if depart < 0:
        return candidates, Counter({"vehicle_not_departed": 1})
    # SUMO marks still-active vehicles at the observation cutoff as "end".
    # That is not destruction: a previous source activity arrival may already
    # define a completed leg, even when the parent activity chain is unfinished.
    if info.get("vaporized", "") not in ("", "end"):
        return candidates, Counter({"vehicle_vaporized": 1})
    if parent["label"] == 1:
        if arrival < depart:
            return candidates, Counter({"bus_service_unfinished": 1})
        candidates.append(leg(parent, 0, depart, arrival, "vehicle_depart", "vehicle_arrival", "bus_service",
                              source_plan=source_leg_plan(plan, source, 0, True, "bus_service"),
                              source_link_status=link_status,
                              intermediate_stop_events=parent["stop_events"]))
        return candidates, rejected
    route = primary_route(plan, source)
    source_stops = (route["stops"] if route else []) + plan["stops"]
    if any(activity_kind(stop) is None for stop in source_stops):
        return candidates, Counter({"ordinary_source_stop_semantics_ambiguous": 1})
    activities = source_stops
    events = sorted(parent["stop_events"], key=lambda event: number(event.get("started")))
    if len(events) > len(activities):
        return candidates, Counter({"more_realized_than_planned_activity_stops": 1})
    start, start_reason = depart, "vehicle_depart"
    for ordinal, event in enumerate(events):
        begun, ended = number(event.get("started")), number(event.get("ended"))
        if begun < start or begun < 0 or (ended >= 0 and ended < begun):
            return [], Counter({"activity_stop_event_order_invalid": 1})
        source_stop = activities[ordinal]
        same_location = location_key(source_stop) == location_key(event)
        if not same_location:
            # Source-requested parking may be rerouted. Preserve the mismatch as
            # an exclusion, rather than pairing changed locations by ordinal.
            return [], Counter({"activity_stop_source_location_mismatch": 1})
        source_path = source_leg_plan(plan, source, ordinal, False, "vehicle_activity_leg")
        candidates.append(leg(parent, ordinal, start, begun, start_reason, "activity_stop_started", "vehicle_activity_leg",
                              source_plan=source_path, activity_stop_event=event,
                              source_activity_stop=source_stop,
                              source_stop_location_matches=same_location,
                              activity_association="ordered_exact_location",
                              source_link_status=link_status))
        if ended < 0:
            rejected["activity_stop_still_ongoing_no_following_departure"] += 1
            if ordinal != len(events)-1:
                return [], Counter({"ongoing_activity_followed_by_another_stop": 1})
            return candidates, rejected
        start, start_reason = ended, "activity_stop_ended"
    if arrival >= start:
        if len(events) != len(activities):
            rejected["arrived_vehicle_has_unobserved_planned_activity_stops"] += 1
        else:
            ordinal = len(events)
            candidates.append(leg(parent, ordinal, start, arrival, start_reason, "vehicle_arrival", "vehicle_activity_leg",
                                  source_plan=source_leg_plan(plan, source, ordinal, True, "vehicle_activity_leg"),
                                  source_link_status=link_status))
    else:
        rejected["final_activity_leg_not_completed"] += 1
    return candidates, rejected


def collect_fcd(path, wanted):
    observations = {key: {"values": array("d"), "roads": [], "types": set()} for key in wanted}
    counts = Counter()
    previous = None
    for step in children(path):
        if step.tag != "timestep":
            continue
        time = number(step.get("time"))
        if time < 0 or (previous is not None and abs(time-previous-DT) > EPS):
            raise ValueError("Enhanced FCD does not have a native quarter-second clock")
        previous = time
        for node in step:
            key = node.tag, node.get("id")
            if key not in observations:
                continue
            if node.tag == "person" and node.get("vehicle", "") not in ("", "NULL", "None"):
                counts["direct_rider_observations_ignored"] += 1
                continue
            row = observations[key]
            row["values"].extend((time, number(node.get("x"), math.nan), number(node.get("y"), math.nan),
                                  number(node.get("speed"), math.nan), number(node.get("pos"), math.nan)))
            row["roads"].append(node.get("lane") or node.get("edge") or "")
            row["types"].add(node.get("type", ""))
    for row in observations.values():
        row["values"] = np.frombuffer(row["values"], dtype=np.float64).reshape(-1, 5)
        row["roads"] = np.asarray(row["roads"], dtype=str)
    return observations, counts


def cover(candidate, observation):
    start, end = candidate["event_start"], candidate["event_end"]
    if end <= start:
        return None, "nonpositive_event_duration", {}
    points = observation["values"]
    if not len(points):
        return None, "parent_has_no_direct_FCD", {}
    a = int(np.searchsorted(points[:, 0], start-EPS, side="left"))
    b = int(np.searchsorted(points[:, 0], end+EPS, side="right"))
    rows = points[a:b]
    diagnostics = {"rows": len(rows), "event_duration_s": end-start,
                   "first_observation_delay_s": float(rows[0, 0]-start) if len(rows) else None,
                   "last_observation_early_s": float(end-rows[-1, 0]) if len(rows) else None,
                   "missing_native_speed_points": int((~np.isfinite(rows[:, 3])).sum()),
                   "missing_position_on_lane_points": int((~np.isfinite(rows[:, 4])).sum())}
    if len(rows) < 2:
        return None, "fewer_than_two_FCD_observations", diagnostics
    if (not np.isfinite(rows[:, :3]).all() or len(observation["types"]) != 1
            or candidate["vtype"] not in observation["types"]):
        return None, "invalid_FCD_values_or_inconsistent_vtype", diagnostics
    if (rows[0, 0]-start > DT+EPS or end-rows[-1, 0] > DT+EPS):
        return None, "FCD_does_not_cover_event_endpoints", diagnostics
    intervals = np.diff(rows[:, 0])
    if not np.allclose(intervals, DT, atol=EPS, rtol=0):
        return None, "FCD_gap_inside_leg", diagnostics
    rates = np.linalg.norm(np.diff(rows[:, 1:3], axis=0), axis=1)/intervals
    if np.any(rates > 100.):
        return None, "FCD_position_jump_above_100mps", diagnostics
    if any(not road for road in observation["roads"][a:b]):
        return None, "FCD_road_reference_missing", diagnostics
    diagnostics["observed_span_s"] = float(rows[-1, 0]-rows[0, 0])
    diagnostics["xy_arc_length_m"] = float(np.linalg.norm(np.diff(rows[:, 1:3], axis=0), axis=1).sum())
    return (rows, observation["roads"][a:b]), None, diagnostics


def realized_leg_audit(candidate, parent, points, roads):
    """Target-only edge-instance association; never changes a source plan."""
    if candidate["kind"] == "walk_stage":
        route = candidate["emitted_walk"]["route"]
        return {"provenance": "emitted_walk_target_only", "status": "complete_emitted_walk_stage",
                "edges": route["edges"], "exit_times": route["exit_times"],
                "parent_edge_instance_indices": list(range(len(route["edges"]))),
                "exit_time_count_matches_edges": len(route["exit_times"]) == len(route["edges"])}
    emitted = parent["emitted"]
    history = emitted["alternatives"] or emitted["routes"]
    if not history:
        return {"provenance": "emitted_route_target_only", "status": "missing", "edges": []}
    route = history[-1]
    edges, exits = route["edges"], route["exit_times"]
    result = {"provenance": "emitted_route_target_only", "parent_edges_hash": route["edges_hash"],
              "parent_route_edge_count": len(edges), "route_history_count": len(history),
              "reroute_count": max(0, len(history)-1)}
    if candidate["kind"] == "bus_service":
        return {**result, "status": "complete_emitted_service", "edges": edges, "exit_times": exits,
                "parent_edge_instance_indices": list(range(len(edges))),
                "exit_time_count_matches_edges": len(exits) == len(edges)}
    if len(exits) != len(edges):
        return {**result, "status": "exit_times_missing_or_length_mismatch", "edges": []}
    completed = [value for value in exits if value >= 0]
    if any(b < a-EPS for a, b in zip(completed, completed[1:])):
        return {**result, "status": "nonmonotone_exit_times", "edges": []}
    end_of_completed = len(completed)
    if any(value >= 0 for value in exits[end_of_completed:]):
        return {**result, "status": "nonprefix_completed_edge_instances", "edges": []}
    enters = [number(parent["tripinfo"].get("depart"))] + exits[:-1]
    normal = [(i, edge_of_lane(road)) for i, road in enumerate(roads)
              if edge_of_lane(road) is not None and not str(edge_of_lane(road)).startswith(":" )]
    if not normal:
        return {**result, "status": "no_normal_vehicle_edge_observation", "edges": []}
    def possible(observation):
        row, edge = observation
        time = points[row, 0]
        return [i for i, value in enumerate(edges) if value == edge and enters[i] >= 0
                and enters[i] <= time+DT+EPS and (exits[i] < 0 or exits[i] >= time-DT-EPS)]
    first, last = possible(normal[0]), possible(normal[-1])
    result["first_edge_instance_candidates"] = first
    result["last_edge_instance_candidates"] = last
    if len(first) == len(last) == 1 and first[0] <= last[0]:
        indices = list(range(first[0], last[0]+1))
        return {**result, "status": "event_and_FCD_edge_instance_slice", "edges": [edges[i] for i in indices],
                "exit_times": [exits[i] for i in indices], "parent_edge_instance_indices": indices,
                "first_normal_observation_time": float(points[normal[0][0], 0]),
                "last_normal_observation_time": float(points[normal[-1][0], 0])}
    return {**result, "status": "ambiguous_emitted_edge_instance_slice", "edges": []}


def summarize(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def verify_inputs(args):
    receipt_path = args.replay/"simulation-complete.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("complete"):
        raise ValueError("Enhanced replay incomplete")
    config_path = args.replay/"config.json"
    if digest(config_path) != receipt["config_sha256"]:
        raise ValueError("Replay configuration changed")
    for filename, expected in receipt["files"].items():
        if digest(args.replay/filename) != expected["sha256"]:
            raise ValueError("Replay output changed")
    comparison = json.loads(args.comparison.read_text())
    if (not comparison.get("complete")
            or comparison["replay_receipt_sha256"] != digest(receipt_path)):
        raise ValueError("Exact baseline/replay invariance check has not passed for these outputs")
    return receipt, json.loads(config_path.read_text()), comparison


def extract(args):
    if args.output.exists():
        raise FileExistsError("Choose a new isolated leg output directory")
    receipt, config, comparison = verify_inputs(args)
    partitions = original_id_partitions(args.cnr_splits)
    parents, exclusions = parent_events(args.replay, partitions)
    source = source_inventory(args.source, set(parents))
    candidates, parent_counts, type_exclusions, boundary_exclusions = [], Counter(), Counter(), Counter()
    source_links, parent_metadata = Counter(), {}
    for key, parent in parents.items():
        typename = parent["tripinfo"].get("vType") or parent["tripinfo"].get("type", "")
        vclass = source["types"].get(typename)
        label = SOURCE_TYPE_LABEL.get(typename, VCLASS_LABEL.get(vclass))
        if label is None or ((parent["tag"] == "person") != (label == 0)):
            type_exclusions[typename or "<missing>"] += 1
            continue
        parent.update(label=label, vtype=typename)
        parent_counts[CLASSES[label]] += 1
        found, rejected = candidate_legs(parent, source)
        source_links[parent.get("source_link_status", "unknown")] += 1
        boundary_exclusions.update(rejected)
        for reason, count in rejected.items():
            exclusions[f"{CLASSES[label]}:{reason}"] += count
        candidates.extend(found)
        if found:
            parent_metadata[parent["tag"]+":"+parent["base_id"]] = {
                "parent_base_id": parent["base_id"], "tag": parent["tag"], "partition": parent["partition"],
                "vtype": typename, "label": label, "source_link_status": parent["source_link_status"],
                "source_plan": parent["source_plan"], "emitted_route_audit_only": parent["emitted"],
                "tripinfo": parent["tripinfo"], "stop_events": parent["stop_events"]}
    wanted = {(row["entity_tag"], row["parent_base_id"]) for row in candidates}
    observed, fcd_counts = collect_fcd(args.replay/"fcd.xml.gz", wanted)
    accepted, rejected_rows, point_arrays, road_arrays = [], [], [], []
    coverage_exclusions, source_status, realized_status, agreement = Counter(), Counter(), Counter(), Counter()
    for row in candidates:
        key = row["entity_tag"], row["parent_base_id"]
        values, reason, diagnostics = cover(row, observed[key])
        row["coverage"] = diagnostics
        if reason:
            row["rejection_reason"] = reason
            rejected_rows.append(row)
            coverage_exclusions[f"{CLASSES[row['label']]}:{reason}"] += 1
            continue
        row["status"] = "event_and_native_sampling_complete_only"
        row["topology_status"] = "not_audited"
        row["generation_condition_status"] = "not_ready"
        row["realized_route_audit_only"] = realized_leg_audit(row, parents[key], values[0], values[1])
        actual_edges = row["realized_route_audit_only"]["edges"]
        planned_edges = row["source_plan"]["edges"]
        row["source_and_realized_edges_equal"] = planned_edges == actual_edges if planned_edges and actual_edges else None
        agreement[str(row["source_and_realized_edges_equal"])] += 1
        realized_status[row["realized_route_audit_only"]["status"]] += 1
        row["leg_id"] = f"{args.episode}:{row['entity_tag']}:{row['parent_base_id']}:leg{row['ordinal']}"
        accepted.append(row)
        point_arrays.append(values[0])
        road_arrays.append(values[1])
        source_status[row["source_plan"]["status"]] += 1
    identities = defaultdict(set)
    for row in accepted:
        identities[row["parent_base_id"]].add(row["partition"])
        if entity_partition(row["parent_base_id"], partitions) != row["partition"] or row["partition"] == "test":
            raise ValueError("A leg violates its original parent partition")
    if any(len(x) != 1 for x in identities.values()):
        raise ValueError("A parent crosses partitions")
    args.output.mkdir(parents=True)
    coordinates = np.concatenate(point_arrays) if point_arrays else np.empty((0, 5))
    roads = np.concatenate(road_arrays) if road_arrays else np.array([], dtype=str)
    unique_roads, road_indices = np.unique(roads, return_inverse=True)
    arrays = {"ids": np.asarray([x["leg_id"] for x in accepted], dtype=str),
              "parent_base_ids": np.asarray([x["parent_base_id"] for x in accepted], dtype=str),
              "entity_tags": np.asarray([x["entity_tag"] for x in accepted], dtype=str),
              "partitions": np.asarray([x["partition"] for x in accepted], dtype=str),
              "labels": np.asarray([x["label"] for x in accepted], dtype=np.int64),
              "ordinals": np.asarray([x["ordinal"] for x in accepted], dtype=np.int64),
              "event_start": np.asarray([x["event_start"] for x in accepted], dtype=float),
              "event_end": np.asarray([x["event_end"] for x in accepted], dtype=float),
              "offset": np.r_[0, np.cumsum([len(x) for x in point_arrays], dtype=np.int64)],
              "txy": coordinates[:, :3], "native_speed": coordinates[:, 3], "pos": coordinates[:, 4],
              "road_ids": unique_roads, "road_index": road_indices.astype(np.int32)}
    np.savez_compressed(args.output/"legs.npz", **arrays)
    with gzip.open(args.output/"leg-metadata.json.gz", "wt", encoding="utf-8") as output:
        json.dump({"scope": "Isolated identifiers, source plans and observed audit metadata; never committed",
                   "accepted": accepted, "coverage_rejected": rejected_rows, "parents": parent_metadata}, output)
    per_class = {}
    for label, name in enumerate(CLASSES):
        rows = [x for x in accepted if x["label"] == label]
        per_class[name] = {"accepted_legs": len(rows), "parents": len({x["parent_base_id"] for x in rows}),
                           "event_duration_s": summarize(x["event_end"]-x["event_start"] for x in rows),
                           "observed_span_s": summarize(x["coverage"]["observed_span_s"] for x in rows),
                           "xy_arc_length_m": summarize(x["coverage"]["xy_arc_length_m"] for x in rows),
                           "under_5_seconds": sum(x["event_end"]-x["event_start"] < 5 for x in rows),
                           "missing_native_speed_points": sum(x["coverage"]["missing_native_speed_points"] for x in rows),
                           "missing_position_on_lane_points": sum(x["coverage"]["missing_position_on_lane_points"] for x in rows),
                           "split_counts": dict(Counter(x["partition"] for x in rows)),
                           "kind_counts": dict(Counter(x["kind"] for x in rows))}
    report = {"complete": True, "episode": args.episode, "generation_ready": False,
              "scope": "Pilot event/coverage interface audit only; no fitting or held-out generation evaluation",
              "classes": CLASSES, "parent_counts_after_test_and_taxonomy_filter": dict(parent_counts),
              "type_exclusions": dict(type_exclusions), "exclusions": dict(exclusions),
              "boundary_exclusions": dict(boundary_exclusions), "coverage_exclusions": dict(coverage_exclusions),
              "source_link_counts": dict(source_links), "source_plan_status_counts": dict(source_status),
              "realized_route_status_counts": dict(realized_status), "source_realized_edges_equal_counts": dict(agreement),
              "candidate_legs": len(candidates), "accepted_legs": len(accepted), "per_class": per_class,
              "parent_partition_overlap": 0, "fcd_counts": dict(fcd_counts),
              "checks": {"source_hashes_verified": True, "replay_hashes_verified": True,
                         "baseline_output_invariance_passed": True, "original_test_templates_excluded": True,
                         "same_parent_same_partition": True, "topology_and_reconstruction_verified": False},
              "rules": {"ordinary_vehicle": "Actual depart/previous source activity parking end to next activity parking start or final arrival; low speed does not define activity boundaries",
                        "vaporized": "SUMO end means observation cutoff, not destruction; completed earlier activity legs remain eligible, while unfinished bus services remain excluded",
                        "bus": "One completed scheduled service, keeping all intermediate stops",
                        "pedestrian": "Completed walk stage with tripinfo and enhanced stage-time agreement; riders excluded",
                        "coverage": "Native .25-second points, >=2 observations, event endpoints within .25s, no internal gaps or >100m/s coordinate jump",
                        "source_routes": "Public input plans only; repeated-edge ambiguity retained; realized paths are separate audit-only metadata",
                        "flow_eligibility": "Flow namespace association is confirmed using emitted route or endpoints; this target-dependent audit cohort is not a future-independent generation request sampler",
                        "short_legs": "No minimum duration tuning; <5s legs counted separately, not silently removed"},
              "sources": {"public_source_manifest_sha256": digest(args.source/"complete.json"),
                          "replay_receipt_sha256": digest(args.replay/"simulation-complete.json"),
                          "comparison_receipt_sha256": digest(args.comparison),
                          "original_split_manifest_sha256": digest(args.cnr_splits),
                          "script_sha256": digest(Path(__file__))},
              "artifacts": {name: {"sha256": digest(args.output/name), "bytes": (args.output/name).stat().st_size}
                            for name in ("legs.npz", "leg-metadata.json.gz")}}
    temporary = args.output/"complete.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(args.output/"complete.json")
    print(json.dumps({"complete": True, "accepted_legs": len(accepted),
                      "per_class": {k: v["accepted_legs"] for k, v in per_class.items()},
                      "generation_ready": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../g30-job/g30-data/source"))
    parser.add_argument("--replay", type=Path, default=Path("g31-data/replay/pilot/300001"))
    parser.add_argument("--comparison", type=Path, default=Path("g31-data/compare/pilot-300001/comparison-complete.json"))
    parser.add_argument("--cnr-splits", type=Path, default=Path("../data/processed/most64/dataset.json"))
    parser.add_argument("--episode", default="300001")
    parser.add_argument("--output", type=Path, default=Path("g31-data/legs/pilot-300001"))
    extract(parser.parse_args())
