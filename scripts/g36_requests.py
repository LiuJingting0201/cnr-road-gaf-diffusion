"""Finite independent SUMO reference: public class-specific plans, native background.

Preparation reads public inputs and original ID/split metadata only. No G32
geometry success, future FCD, realized route or model score selects requests.
The full frozen manifest stays on the server; Git receives aggregate receipts.
"""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from g30_prepare import entity_partition, original_id_partitions
from g31_paths import SourceNetwork, source_inputs

CLASSES = ["bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
FAMILIES = {"ptw": 1, "passenger": 2, "bicycle": 3, "on-demand": 4}
SELECTION_SEED = 36001
SIMULATION_SEED = 360201
BEGIN, END, DT = 14400., 23000., .25
DEPARTURE = 18000.


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rank(*parts):
    return hashlib.sha256(json.dumps([SELECTION_SEED, *parts], separators=(",", ":")).encode()).digest()


def child_nodes(path):
    stream = gzip.open(path, "rb") if str(path).endswith(".gz") else path.open("rb")
    with stream:
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


def chosen_type(family, ident):
    # The source families remain intact; only their latent type realization is
    # fixed before SUMO. The ptw experiment specifically requests motorcycle.
    options = {"bus": ["bus"], "ptw": ["motorcycle"],
               "passenger": ["passenger1", "passenger2a", "passenger2b", "passenger3", "passenger4"],
               "bicycle": ["slowbicycle", "avgbicycle", "fastbicycle"],
               "on-demand": ["taxi", "uber"]}[family]
    return options[int.from_bytes(rank("type", ident), "big") % len(options)]


def endpoint(network, edge, vclass, departure):
    candidates = network.permitted(edge, vclass)
    if not candidates:
        raise ValueError("Source endpoint edge has no permitted lane")
    lane = min(candidates, key=lambda name: int(network.lanes[name]["attributes"]["index"]))
    length = float(network.lanes[lane]["attributes"]["length"])
    return {"lane": lane, "pos": min(20., length / 2.) if departure else length}


def stop_anchors(edges, stops, locations):
    anchors, cursor, last_pos = [], 0, -1.
    for index, stop in enumerate(stops):
        location = locations[stop["busStop"]]
        lane, pos = location["lane"], float(location["endPos"])
        edge = lane.rsplit("_", 1)[0]
        choices = [i for i in range(cursor, len(edges)) if edges[i] == edge
                   and (i > cursor or pos >= last_pos)]
        if not choices:
            raise ValueError("Public bus stop order cannot bind to route edge instances")
        cursor, last_pos = choices[0], pos
        anchors.append({"edge_index": cursor, "lane": lane, "pos": pos,
                        "source_stop_index": index, "busStop": stop["busStop"],
                        "startPos": float(location["startPos"]), "endPos": pos})
    return anchors


def prepare(source, split_path, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from immutable public input")
    output.mkdir(parents=True, exist_ok=False)
    try:
        net_path, types = source_inputs(source)
        network = SourceNetwork(net_path)
        network_hash = digest(net_path)
        splits = original_id_partitions(split_path)
        buses = ET.parse(source / "scenario/in/route/most.buses.flows.xml").getroot()
        flows = {n.get("route"): n.attrib for n in buses.findall("flow")}
        locations = {n.get("id"): n.attrib for n in ET.parse(source / "scenario/in/add/most.busstops.add.xml").getroot().findall("busStop")}
        bus_routes = sorted(buses.findall("route"), key=lambda n: n.get("id"))
        if len(bus_routes) != 24:
            raise ValueError("Pinned source must have 24 bus service plans")
        selected = []
        for node in bus_routes:
            selected.append({"source_parent": node.get("id"), "family": "bus", "label": 0,
                             "route": copy.deepcopy(node), "line": flows[node.get("id")].get("line", ""),
                             "source_partition": "public_service_template_shared_across_partitions"})
        eligible, counts = {family: [] for family in FAMILIES}, Counter()
        for node in child_nodes(source / "scenario/in/route/most.pedestrian.rou.xml"):
            family = node.get("type")
            if node.tag != "vehicle" or family not in FAMILIES:
                continue
            route = node.find("route")
            if route is None or node.findall("stop") or route.findall("stop"):
                counts[family + "/not_single_no_activity_stop_leg"] += 1
                continue
            partition = entity_partition(node.get("id"), splits)
            counts[family + "/" + partition] += 1
            if partition != "val":
                continue
            eligible[family].append((node.get("id"), copy.deepcopy(route)))
        for family, label in FAMILIES.items():
            pool = sorted(eligible[family], key=lambda row: rank("request", family, row[0]))
            if len(pool) < 19:
                raise ValueError("Too few public validation parents for " + family)
            for ident, route in pool[:19]:
                selected.append({"source_parent": ident, "family": family, "label": label,
                                 "route": route, "line": "", "source_partition": "val"})
        requests, xml_routes, vehicles = [], [], []
        for i, item in enumerate(selected):
            ident = f"g36_{SIMULATION_SEED}_{i:03d}"
            source_id, family = item["source_parent"], item["family"]
            vtype = chosen_type(family, source_id)
            vclass = types[vtype]
            edges = item["route"].get("edges", "").split()
            depart = DEPARTURE + (int.from_bytes(rank("departure", source_id), "big") % 2401) * DT
            row = {"id": ident, "source_parent": source_id, "source_family": family,
                   "schema": "g32-presimulation-route-v1", "network_sha256": network_hash,
                   "source_partition": item["source_partition"], "label": item["label"],
                   "class": CLASSES[item["label"]], "vtype": vtype, "vclass": vclass,
                   "depart": depart, "line": item["line"], "edges": edges,
                   "source_stops": [dict(n.attrib) for n in item["route"].findall("stop")],
                   "provenance": "public_network_and_presimulation_request",
                   "frozen_before_simulation": True}
            try:
                row["start"] = endpoint(network, edges[0], vclass, True)
                row["end"] = endpoint(network, edges[-1], vclass, False)
                row["stops"] = stop_anchors(edges, row["source_stops"], locations)
                row["status"] = "request_ready"
                route = copy.deepcopy(item["route"])
                route.set("id", ident + "_route")
                # Stand-alone route stops preserve their relative-until semantics.
                xml_routes.append(route)
                vehicle = ET.Element("vehicle", {"id": ident, "route": route.get("id"),
                    "type": vtype, "depart": str(depart),
                    "departLane": network.lanes[row["start"]["lane"]]["attributes"]["index"],
                    "departPos": str(row["start"]["pos"]),
                    "arrivalLane": network.lanes[row["end"]["lane"]]["attributes"]["index"],
                    "arrivalPos": str(row["end"]["pos"])})
                if row["line"]:
                    vehicle.set("line", row["line"])
                ET.SubElement(vehicle, "param", {"key": "has.fcd.device", "value": "true"})
                ET.SubElement(vehicle, "param", {"key": "has.rerouting.device", "value": "false"})
                vehicles.append(vehicle)
            except (KeyError, IndexError, ValueError) as error:
                row.update(status="source_request_unresolved", reason=str(error))
            requests.append(row)
        root = ET.Element("routes")
        root.extend(xml_routes)
        root.extend(sorted(vehicles, key=lambda node: (float(node.get("depart")), node.get("id"))))
        ET.indent(root)
        ET.ElementTree(root).write(output / "targets.rou.xml", encoding="utf-8", xml_declaration=True)
        manifest = {"schema": "g36-frozen-public-requests-v1", "classes": CLASSES,
                    "selection_seed": SELECTION_SEED, "simulation_seed": SIMULATION_SEED,
                    "begin": BEGIN, "end": END, "step_s": DT, "requests": requests,
                    "network_sha256": network_hash, "source_manifest_sha256": digest(source / "complete.json"),
                    "original_split_metadata_sha256": digest(split_path),
                    "targets_sha256": digest(output / "targets.rou.xml"),
                    "policy": {"nonbus": "19 hash-ranked val public single-leg no-activity-stop source parents per family",
                               "bus": "all 24 public service routes; shared route templates, not unseen-line generalization",
                               "types": "source-family hash draw; ptw requests motorcycle rather than moped",
                               "endpoints": "lowest-index class-permitted lanes; start min(20m,lane_length/2); end lane_length",
                               "departure": "episode-specific explicit hash departure within morning or late-morning600s window; replaces triggered departure",
                               "bus_stops": "unchanged stand-alone route relative until/duration and public line",
                               "rerouting": "disabled only for the 100 target requests; native background unchanged",
                               "geometry_gate": "none; G32 successes do not select or omit simulation requests",
                               "data_scope": "independent SUMO realization under controlled new departures, not exact CNR replay"}}
        save(output / "requests.json", manifest)
        receipt = {"complete": True, "phase": "requests_frozen", "denominator": 100,
                   "ready": len(vehicles), "class_counts": dict(Counter(r["class"] for r in requests)),
                   "status_counts": dict(Counter(r["status"] for r in requests)),
                   "eligible_source_counts": dict(counts), "simulation_started": False,
                   "requests_sha256": digest(output / "requests.json"), "targets_sha256": manifest["targets_sha256"],
                   "network_sha256": manifest["network_sha256"], "script_sha256": digest(Path(__file__))}
        if len(requests) != 100 or len({r["id"] for r in requests}) != 100:
            raise ValueError("Fixed request denominator changed")
        save(output / "prepare-complete.json", receipt)
        return receipt
    except Exception as error:
        save(output / "prepare-failed.json", {"complete": False, "error": repr(error)})
        raise


def run(source, prepared, output, sumo):
    source, prepared, output, sumo = [p.resolve() for p in (source, prepared, output, sumo)]
    receipt = json.loads((prepared / "prepare-complete.json").read_text())
    if not receipt["complete"] or digest(prepared / "requests.json") != receipt["requests_sha256"] or digest(prepared / "targets.rou.xml") != receipt["targets_sha256"]:
        raise ValueError("Frozen preparation changed")
    manifest = json.loads((prepared / "requests.json").read_text())
    if manifest["simulation_seed"] != SIMULATION_SEED or manifest["end"] != END:
        raise ValueError("Episode does not match frozen requests")
    if digest(source / "complete.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Public source cache changed")
    output.mkdir(parents=True, exist_ok=False)
    config = ET.parse(source / "scenario/most.sumocfg").getroot()
    routes = [str(source / "scenario" / x) for x in config.find("input/route-files").get("value").split(",")]
    routes.append(str(prepared / "targets.rou.xml"))
    cmd = [str(sumo), "-c", str(source / "scenario/most.sumocfg"),
           "--route-files", ",".join(routes), "--seed", str(SIMULATION_SEED),
           "--begin", str(BEGIN), "--end", str(END), "--step-length", str(DT),
           "--output-prefix", "", "--fcd-output", str(output / "fcd.xml.gz"),
           "--fcd-output.attributes", "x,y,type,speed,lane,edge,vehicle,pos",
           "--device.fcd.probability", "0", "--person-device.fcd.probability", "0",
           "--device.fcd.period", str(DT), "--tripinfo-output", str(output / "tripinfo.xml"),
           "--tripinfo-output.write-unfinished", "true", "--vehroute-output", str(output / "vehroute.xml.gz"),
           "--vehroute-output.write-unfinished", "true", "--vehroute-output.exit-times", "true",
           "--vehroute-output.last-route", "false", "--stop-output", str(output / "stops.xml"),
           "--summary-output", str(output / "summary.xml"), "--log", str(output / "sumo.log"), "--no-step-log", "true"]
    save(output / "config.json", {"command": cmd, "request_manifest_sha256": receipt["requests_sha256"],
                                  "version": subprocess.check_output([str(sumo), "--version"], text=True).splitlines()[0],
                                  "requests_frozen_before_simulation": True, "denominator": 100})
    started = time.monotonic()
    try:
        with (output / "console.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        files = {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                 for name in ("fcd.xml.gz", "tripinfo.xml", "vehroute.xml.gz", "stops.xml", "summary.xml")}
        if sum(x["bytes"] for x in files.values()) > 8 * 1024**3:
            raise ValueError("Output exceeds fixed 8 GiB experiment storage budget")
        answer = {"complete": True, "phase": "simulation", "seconds": time.monotonic() - started,
                  "seed": SIMULATION_SEED, "denominator": 100, "files": files,
                  "reference_extraction_complete": False, "config_sha256": digest(output / "config.json")}
        save(output / "simulation-complete.json", answer)
        return answer
    except Exception as error:
        save(output / "simulation-failed.json", {"complete": False, "error": repr(error), "seconds": time.monotonic() - started})
        raise


def extract(prepared, simulation, output):
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = json.loads((prepared / "requests.json").read_text())
        receipt = json.loads((simulation / "simulation-complete.json").read_text())
        if not receipt["complete"]:
            raise ValueError("Simulation is not complete")
        for name, info in receipt["files"].items():
            if digest(simulation / name) != info["sha256"]:
                raise ValueError("Simulation artifact changed")
        requests = {r["id"]: r for r in manifest["requests"]}
        trips, actual_routes, stops = {}, {}, Counter()
        for node in child_nodes(simulation / "tripinfo.xml"):
            if node.tag == "tripinfo" and node.get("id") in requests:
                trips[node.get("id")] = dict(node.attrib)
        for node in child_nodes(simulation / "vehroute.xml.gz"):
            if node.tag == "vehicle" and node.get("id") in requests:
                routes = node.findall("routeDistribution/route") or node.findall("route")
                actual_routes[node.get("id")] = [x.get("edges", "").split() for x in routes]
        for node in child_nodes(simulation / "stops.xml"):
            if node.tag == "stopinfo" and node.get("id") in requests and float(node.get("ended", "-1")) >= 0:
                stops[node.get("id")] += 1
        tracks = {ident: [] for ident in requests}
        unknown_fcd = 0
        for step in child_nodes(simulation / "fcd.xml.gz"):
            stamp = float(step.get("time"))
            for node in step:
                ident = node.get("id")
                if ident not in requests:
                    unknown_fcd += 1
                    continue
                tracks[ident].append((stamp, float(node.get("x")), float(node.get("y"))))
        if unknown_fcd:
            raise ValueError("Unexpected non-target FCD records: " + str(unknown_fcd))
        rows, kept, values, offsets = [], [], [], [0]
        for ident, request in requests.items():
            trip = trips.get(ident, {})
            points = np.asarray(tracks[ident], dtype=np.float64).reshape(-1, 3)
            row = {"id": ident, "class": request["class"], "label": request["label"],
                   "samples": len(points), "planned_stops": len(request["source_stops"]),
                   "completed_stops": stops[ident], "vaporized": trip.get("vaporized", ""),
                   "actual_route_equals_planned": bool(actual_routes.get(ident)) and actual_routes[ident][-1] == request["edges"],
                   "actual_route_history_count": len(actual_routes.get(ident, []))}
            if request["status"] != "request_ready":
                row["status"] = "request_unresolved"
            elif float(trip.get("depart", "-1")) < 0:
                row["status"] = "not_departed"
            elif float(trip.get("arrival", "-1")) < 0:
                row["status"] = "not_completed_by_cutoff"
            elif stops[ident] != len(request["source_stops"]):
                row["status"] = "planned_stop_count_mismatch"
            elif len(points) < 2:
                row["status"] = "insufficient_observations"
            elif not np.isfinite(points).all() or not np.allclose(np.diff(points[:, 0]), DT, rtol=0, atol=1e-7):
                row["status"] = "invalid_or_discontinuous"
            elif np.max(np.linalg.norm(np.diff(points[:, 1:], axis=0), axis=1) / DT) > 100:
                row["status"] = "coordinate_jump_over_100mps"
            else:
                row["status"] = "reference_complete"
                row.update(observed_span_s=float(points[-1, 0] - points[0, 0]),
                           event_duration_s=float(trip["arrival"]) - float(trip["depart"]),
                           first_observation_minus_depart_s=float(points[0, 0]) - float(trip["depart"]),
                           arrival_minus_last_observation_s=float(trip["arrival"]) - float(points[-1, 0]))
                values.append(points); offsets.append(offsets[-1] + len(points)); kept.append(ident)
            rows.append(row)
        if not kept:
            raise ValueError("No completed reference trajectories")
        np.savez_compressed(output / "references.npz", ids=np.asarray(kept),
                            labels=np.asarray([requests[i]["label"] for i in kept], dtype=np.int64),
                            offsets=np.asarray(offsets, dtype=np.int64), txy=np.concatenate(values))
        save(output / "reference-metadata.json", {"rows": rows, "requests_sha256": digest(prepared / "requests.json")})
        result = {"complete": True, "phase": "references_extracted", "denominator": len(requests),
                  "accepted": len(kept), "classes": CLASSES,
                  "status_counts": dict(Counter(r["status"] for r in rows)),
                  "by_class": {name: dict(Counter(r["status"] for r in rows if r["class"] == name)) for name in CLASSES},
                  "planned_actual_route_matches": sum(r["actual_route_equals_planned"] for r in rows),
                  "observed_target_records": sum(len(t) for t in tracks.values()), "unknown_fcd_records": unknown_fcd,
                  "source_sha256": {"script": digest(Path(__file__)), "requests": digest(prepared / "requests.json"),
                                    "simulation_receipt": digest(simulation / "simulation-complete.json")},
                  "artifacts": {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                                for name in ("references.npz", "reference-metadata.json")},
                  "limits": ["Reference SUMO trajectory completion is distinct from diffusion or map-decoder success.",
                             "Same map and public route templates; new departures/RNG, not unseen-city or unseen-line generalization.",
                             "Target rerouting disabled, native background and public bus stop schedules retained.",
                             "No G32 geometry filter; all 100 original requests remain in the denominator."]}
        save(output / "complete.json", result)
        return result
    except Exception as error:
        save(output / "failed.json", {"complete": False, "error": repr(error)})
        raise


def self_test():
    import tempfile
    checks = []
    edges = ["a", "b", "a"]
    locations = {"p": {"lane": "a_0", "startPos": "5", "endPos": "10"},
                 "q": {"lane": "b_0", "startPos": "5", "endPos": "15"}}
    anchors = stop_anchors(edges, [{"busStop": "p"}, {"busStop": "q"}, {"busStop": "p"}], locations)
    assert [x["edge_index"] for x in anchors] == [0, 1, 2]
    checks.append("repeated_edge_stops_bind_distinct_forward_instances")
    try:
        stop_anchors(["b", "a"], [{"busStop": "p"}, {"busStop": "q"}], locations)
    except ValueError:
        checks.append("contradictory_source_stop_order_rejected")
    else:
        raise AssertionError("Reversed source stop order accepted")
    draws = {chosen_type("on-demand", str(i)) for i in range(50)}
    assert draws == {"taxi", "uber"}
    assert {chosen_type("ptw", str(i)) for i in range(50)} == {"motorcycle"}
    checks.append("explicit_type_draw_preserves_on_demand_family_and_motorcycle_request")
    with tempfile.TemporaryDirectory(prefix="g33-request-checks-") as folder:
        folder = Path(folder); prepared = folder / "prepared"; simulation = folder / "simulation"
        prepared.mkdir(); simulation.mkdir()
        requests = [{"id": "a", "class": "private_car", "label": 2, "source_stops": [],
                     "edges": ["planned"], "status": "request_ready"},
                    {"id": "b", "class": "bus", "label": 0, "source_stops": [],
                     "edges": ["planned"], "status": "request_ready"}]
        save(prepared / "requests.json", {"requests": requests})
        (simulation / "tripinfo.xml").write_text('<tripinfos><tripinfo id="a" depart="0" arrival="0.5"/><tripinfo id="b" depart="0" arrival="-1" vaporized="end"/></tripinfos>')
        with gzip.open(simulation / "vehroute.xml.gz", "wt") as stream:
            stream.write('<routes><vehicle id="a"><route edges="different"/></vehicle><vehicle id="b"><route edges="planned"/></vehicle></routes>')
        with gzip.open(simulation / "fcd.xml.gz", "wt") as stream:
            stream.write('<fcd-export><timestep time="0"><vehicle id="a" x="0" y="0"/><vehicle id="b" x="0" y="1"/></timestep><timestep time="0.25"><vehicle id="a" x="1" y="0"/><vehicle id="b" x="1" y="1"/></timestep></fcd-export>')
        (simulation / "stops.xml").write_text('<stops/>')
        save(simulation / "simulation-complete.json", {"complete": True, "files": {name: {"sha256": digest(simulation / name)}
             for name in ("tripinfo.xml", "vehroute.xml.gz", "fcd.xml.gz", "stops.xml")}})
        result = extract(prepared, simulation, folder / "extracted")
        assert result["denominator"] == 2 and result["accepted"] == 1
        assert result["status_counts"] == {"reference_complete": 1, "not_completed_by_cutoff": 1}
        assert result["planned_actual_route_matches"] == 1
        rows = json.loads((folder / "extracted/reference-metadata.json").read_text())["rows"]
        assert rows[0]["arrival_minus_last_observation_s"] == .25
        assert rows[0]["actual_route_equals_planned"] is False
        checks.extend(["unfinished_reference_keeps_denominator", "route_mismatch_reported_without_future_condition_repair",
                       "missing_final_event_sample_not_fabricated"])
        try:
            extract(prepared, simulation, folder / "extracted")
        except FileExistsError:
            checks.append("finished_output_not_overwritten")
        else:
            raise AssertionError("Finished extraction was overwritten")
        (simulation / "tripinfo.xml").write_text('<changed/>')
        try:
            extract(prepared, simulation, folder / "tampered")
        except ValueError:
            assert (folder / "tampered/failed.json").exists()
            checks.append("changed_simulation_artifact_rejected_and_failure_saved")
        else:
            raise AssertionError("Changed simulation was accepted")
    return {"complete": True, "scope": "synthetic_requests_and_extraction_checks", "checks": checks,
            "checks_passed": len(checks), "real_simulation_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(360201,360202), required=True)
    parser.add_argument("action", choices=("prepare", "run", "extract", "checks"))
    parser.add_argument("--source", type=Path, default=Path("../g30-job/g30-data/source"))
    parser.add_argument("--splits", type=Path, default=Path("../data/processed/most64/dataset.json"))
    parser.add_argument("--prepared", type=Path, default=Path("g33-data/requests"))
    parser.add_argument("--simulation", type=Path, default=Path("g33-data/reference-330201"))
    parser.add_argument("--output", type=Path, default=Path("g33-data/reference-extracted"))
    parser.add_argument("--sumo", type=Path, default=Path("../g10-env/sumo/bin/sumo"))
    args = parser.parse_args()
    SIMULATION_SEED = args.episode
    DEPARTURE = 18000. if args.episode == 360201 else 32400.
    END = DEPARTURE + 5000.
    result = (self_test() if args.action == "checks" else prepare(args.source, args.splits, args.prepared) if args.action == "prepare" else
              run(args.source, args.prepared, args.simulation, args.sumo) if args.action == "run" else
              extract(args.prepared, args.simulation, args.output))
    print(json.dumps(result, indent=2))
