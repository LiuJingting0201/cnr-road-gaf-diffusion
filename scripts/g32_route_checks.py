"""Finite synthetic checks for the prospective vehicle-route geometry adapter."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

from g32_routes import (PROVENANCE, SCHEMA, UnsupportedRoute, VehicleRouteNetwork,
                        request_from_source_plan)
from g31_paths import source_inputs
from g31_replay import COMMIT, digest


def network_xml(lane_rules="", offset=3.2, internal_gap=0., declared_length=100.):
    edges = []
    for edge, begin, count in (("a", 0, 2), ("b", 110, 2), ("c", 220, 1)):
        lanes = []
        for i in range(count):
            y = offset * i if edge != "c" else offset
            rule = lane_rules if i == 0 and edge in {"a", "b"} else ""
            lanes.append(f'<lane id="{edge}_{i}" index="{i}" speed="15" length="{declared_length}" width="3.2" shape="{begin},{y} {begin+100},{y}" {rule}/>')
        edges.append(f'<edge id="{edge}">{"".join(lanes)}</edge>')
    for lane, begin, y in ((":i_0", 100, 0), (":i_1", 100, offset), (":j_0", 210, offset)):
        gap = internal_gap if lane == ":j_0" else 0
        edges.append(f'<edge id="{lane[:-2]}{lane[-1]}" function="internal"><lane id="{lane}" index="0" speed="15" length="10" width="3.2" shape="{begin+gap},{y} {begin+10},{y}"/></edge>')
    connections = []
    for index in (0, 1):
        connections.append(f'<connection from="a" to="b" fromLane="{index}" toLane="{index}" via=":i_{index}"/>')
        connections.append(f'<connection from=":i{index}" to="b" fromLane="0" toLane="{index}"/>')
    connections.extend(['<connection from="b" to="c" fromLane="1" toLane="0" via=":j_0"/>',
                        '<connection from=":j0" to="c" fromLane="0" toLane="0"/>'])
    return '<net>' + ''.join(edges + connections) + '</net>'


def run():
    checks = []
    with tempfile.TemporaryDirectory(prefix="g32-route-checks-") as directory:
        directory = Path(directory)

        def net(name="base", **kwargs):
            path = directory / (name + ".net.xml")
            path.write_text(network_xml(**kwargs))
            return VehicleRouteNetwork(path)

        base = net()

        def request(network=base, **updates):
            result = {"schema": SCHEMA, "provenance": PROVENANCE,
                      "network_sha256": network.network_sha256,
                      "frozen_before_simulation": True, "vclass": "passenger",
                      "edges": ["a", "b", "c"], "start": {"lane": "a_0", "pos": 5.},
                      "end": {"lane": "c_0", "pos": 95.}, "stops": []}
            result.update(updates)
            return result

        def rejected(name, call):
            try:
                call()
            except UnsupportedRoute:
                checks.append(name)
            else:
                raise AssertionError("Expected rejection: " + name)

        for vclass in ("passenger", "bus", "motorcycle", "bicycle", "taxi"):
            route = base.construct(request(vclass=vclass))
            assert route.lane_changes == 1
            assert route.vertices[0] == (5., 0.) and route.vertices[-1] == (315., 3.2)
        checks.append("five_vehicle_classes_with_required_lateral_change")
        assert any(":j_0" in piece["lanes"] for piece in route.pieces)
        assert math.isclose(route.decode([0, route.length_m])[-1][0], 315.)
        checks.append("internal_junction_geometry_retained_and_exact_decoder_endpoints")
        rejected("path_exhaustion_not_silently_clipped", lambda: route.decode([route.length_m + .1]))
        straight = base.construct(request(edges=["a", "b"], end={"lane": "b_0", "pos": 95.}))
        assert straight.lane_changes == 0
        checks.append("no_unnecessary_lane_change")
        stopped = base.construct(request(stops=[{"edge_index": 1, "lane": "b_0", "pos": 20., "source_stop_index": 0}]))
        assert any(piece.get("source_stop_index") == 0 and piece["lanes"][-1] == "b_0" for piece in stopped.pieces)
        checks.append("planned_stop_lane_is_mandatory")
        limited = net("restricted", lane_rules='changeLeft="bus"')
        rejected("changeleft_class_permission_enforced", lambda: limited.construct(request(limited)))
        assert limited.construct(request(limited, vclass="bus")).lane_changes == 1
        checks.append("changeleft_permitted_class_succeeds")
        right = net("right", lane_rules='changeRight="bus"')
        # Source lane 1 needs its own rightward permission; edit only this fixture.
        right.lanes["a_1"]["attributes"]["changeRight"] = "bus"
        rejected("changeright_class_permission_enforced", lambda: right.construct(request(right, edges=["a"], start={"lane": "a_1", "pos": 5.}, end={"lane": "a_0", "pos": 95.})))
        assert right.construct(request(right, vclass="bus", edges=["a"], start={"lane": "a_1", "pos": 5.}, end={"lane": "a_0", "pos": 95.})).lane_changes == 1
        checks.append("changeright_permitted_class_succeeds")
        forbidden = net("forbidden")
        forbidden.lanes["c_0"]["attributes"]["disallow"] = "passenger"
        rejected("lane_occupancy_permission_enforced", lambda: forbidden.construct(request(forbidden)))
        connection_denied = net("connection_denied")
        connection_denied.successors["b_1"][0][1]["disallow"] = "passenger"
        rejected("connection_permission_enforced", lambda: connection_denied.construct(request(connection_denied)))
        denied = net("denied", lane_rules='changeLeft=""')
        rejected("empty_change_permission_denies_all", lambda: denied.construct(request(denied)))
        separated = net("separated", offset=10.)
        rejected("lane_gap_outside_permitted_corridors", lambda: separated.construct(request(separated)))
        broken = net("gap", internal_gap=4.)
        rejected("junction_jump_over_gap_tolerance", lambda: broken.construct(request(broken)))
        rejected("insufficient_distance_for_lateral_change", lambda: base.construct(request(edges=["a"], start={"lane": "a_0", "pos": 90.}, end={"lane": "a_1", "pos": 95.})))
        rejected("pedestrian_is_explicitly_unsupported", lambda: base.construct(request(vclass="pedestrian")))
        rejected("unfrozen_request_rejected", lambda: base.construct(request(frozen_before_simulation=False)))
        rejected("future_fcd_provenance_rejected", lambda: base.construct(request(provenance="realized_fcd")))
        rejected("network_hash_mismatch_rejected", lambda: base.construct(request(network_sha256="wrong")))
        rejected("random_endpoint_policy_requires_instantiation", lambda: base.construct(request(end={"lane": "c_0", "pos": "random"})))
        rejected("stop_order_uses_edge_occurrence_and_position", lambda: base.construct(request(stops=[{"edge_index": 1, "lane": "b_0", "pos": 80.}, {"edge_index": 1, "lane": "b_0", "pos": 20.}])))
        stretched = net("declared_length", declared_length=200.)
        half = stretched.construct(request(stretched, edges=["a"], start={"lane": "a_0", "pos": 50.}, end={"lane": "a_0", "pos": 100.}))
        assert half.vertices[0] == (25., 0.) and half.vertices[-1] == (50., 0.)
        checks.append("declared_lane_position_differs_from_shape_arclength")
        rejected("ten_declared_metres_are_not_ten_geometry_metres", lambda: stretched.construct(request(stretched, edges=["a"], start={"lane": "a_0", "pos": 50.}, end={"lane": "a_1", "pos": 60.})))
        geometric = stretched.construct(request(stretched, edges=["a"], start={"lane": "a_0", "pos": 50.}, end={"lane": "a_1", "pos": 70.}))
        assert all(min(window["longitudinal_geometry_m"]) >= 10. - 1e-10
                   for piece in geometric.pieces for window in piece["lane_change_windows"])
        checks.append("local_change_has_ten_actual_geometry_metres_on_both_lanes")
        from shapely.geometry import LineString
        curved = LineString([(0, 0), (30, 0), (50, 20), (80, 20), (100, 0), (130, 0)])
        offset = curved.parallel_offset(3.2, "left", join_style=2)
        # The lanes touch along their initial straight, then separate slightly
        # around the middle curve. A change is legal early, not across that gap.
        offset = LineString([(x, y + .4 if y > 20 else y) for x, y in offset.coords])
        shapes = [curved, offset]
        lanes = ''.join(f'<lane id="curve_{i}" index="{i}" length="{shape.length}" width="3.2" shape="' +
                        ' '.join(f'{x},{y}' for x, y in shape.coords) + '"/>' for i, shape in enumerate(shapes))
        curve_path = directory / "curve.net.xml"
        curve_path.write_text('<net><edge id="curve">' + lanes + '</edge></net>')
        curve_net = VehicleRouteNetwork(curve_path)
        whole_blend = curve_net.blend("curve_0", "curve_1", 0., 1.)
        assert not curve_net.covered(whole_blend, ("curve_0", "curve_1")), "Fixture must expose whole-edge blend failure"
        curve_route = curve_net.construct(request(curve_net, edges=["curve"], start={"lane": "curve_0", "pos": 0.}, end={"lane": "curve_1", "pos": offset.length}))
        assert curve_net.covered(curve_route.vertices, ("curve_0", "curve_1"))
        checks.append("curved_lane_global_blend_fails_but_local_window_stays_in_corridor")
        plan = {"provenance": "public_input_only", "status": "complete_scheduled_source_service",
                "kind": "vehicle", "edges": ["a", "b", "c"], "planned_stops": [{"busStop": "public-stop"}]}
        locations = {"public-stop": {"lane": "b_0", "startPos": "10", "endPos": "20"}}
        start, end = {"lane": "a_0", "pos": 5.}, {"lane": "c_0", "pos": 95.}
        wrapper = lambda stops: request_from_source_plan(plan, "bus", start, end, stops, locations, network_sha256=base.network_sha256)
        rejected("source_bus_stop_cannot_be_omitted", lambda: wrapper([]))
        rejected("source_stop_lane_cannot_be_substituted", lambda: wrapper([{"edge_index": 1, "lane": "b_1", "pos": 20., "source_stop_index": 0}]))
        assert base.construct(wrapper([{"edge_index": 1, "lane": "b_0", "pos": 20., "source_stop_index": 0}])).length_m > 0
        checks.append("public_source_bus_stop_binding")
        activity = {"provenance": "public_input_only", "status": "unique_source_activity_leg_slice",
                    "kind": "vehicle", "edges": ["a", "b"], "planned_activity_boundaries": [{"lane": "b_0", "position": "20"}]}
        rejected("source_activity_endpoint_cannot_be_replaced", lambda: request_from_source_plan(activity, "passenger", start, {"lane": "b_0", "pos": 30.}, [], {}, 0, network_sha256=base.network_sha256))
        literal = request(source_attributes={"departLane": "1"}, source_endpoint_constraints={"depart": True})
        rejected("literal_source_depart_lane_enforced", lambda: base.construct(literal))
        literal = request(source_attributes={"arrivalPos": "94"}, source_endpoint_constraints={"arrival": True})
        rejected("literal_source_arrival_position_enforced", lambda: base.construct(literal))
        literal = request(source_attributes={"arrivalPos": "-5"}, source_endpoint_constraints={"arrival": True})
        assert base.construct(literal).vertices[-1] == (315., 3.2)
        checks.append("negative_literal_source_position_resolved_against_lane_length")
        path = directory / "repeated.net.xml"
        loop = ('<edge id=":loop" function="internal"><lane id=":loop_0" index="0" length="360" width="3.2" '
                'shape="320,3.2 320,20 0,20 0,0"/></edge>'
                '<connection from="c" to="a" fromLane="0" toLane="0" via=":loop_0"/>'
                '<connection from=":loop" to="a" fromLane="0" toLane="0"/>')
        path.write_text(network_xml().replace('</net>', loop + '</net>'))
        repeated = VehicleRouteNetwork(path)
        loop_request = request(repeated, edges=["a", "b", "c", "a"], end={"lane": "a_0", "pos": 95.},
                               stops=[{"edge_index": 0, "lane": "a_1", "pos": 20., "source_stop_index": 0},
                                      {"edge_index": 3, "lane": "a_0", "pos": 40., "source_stop_index": 1}])
        loop_route = repeated.construct(loop_request)
        assert loop_route.vertices[-1] == (95., 0.)
        assert [piece["edge_occurrence"] for piece in loop_route.pieces if piece.get("source_stop_index") is not None] == [0, 3]
        checks.append("repeated_edge_instances_and_stop_visits_remain_distinct")
    return {"complete": True, "checks_passed": len(checks), "checks": checks,
            "native_source_coverage_executed": False, "simulation_or_training_executed": False,
            "scope": "synthetic_vehicle_point_geometry_constructor_checks"}


def public_bus_proof(source, output):
    """Exactly the 24 pinned public services, once, without simulation/FCD."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Proof output must be separate from public input")
    output.mkdir(parents=True, exist_ok=False)
    net_path, types = source_inputs(source)
    if types.get("bus") != "bus":
        raise ValueError("Pinned bus vehicle class changed")
    network = VehicleRouteNetwork(net_path)
    source_routes = ET.parse(source / "scenario/in/route/most.buses.flows.xml").getroot()
    public_stops = ET.parse(source / "scenario/in/add/most.busstops.add.xml").getroot()
    locations = {node.get("id"): dict(node.attrib) for node in public_stops.findall("busStop")}
    flows = {node.get("route"): dict(node.attrib) for node in source_routes.findall("flow")}
    routes = sorted(source_routes.findall("route"), key=lambda node: node.get("id"))
    if len(routes) != 24 or len(flows) != 24:
        raise ValueError("Expected all 24 pinned public bus services")
    policy = {"source_commit": COMMIT, "source_scope": "all_24_public_bus_services",
              "departure_lane": "lowest_index_bus_permitted_lane_on_first_edge",
              "departure_position_m": 20.,
              "arrival_lane": "lowest_index_bus_permitted_lane_on_last_edge",
              "arrival_position": "declared_lane_length",
              "stop_position": "public_busStop_endPos",
              "stop_edge_occurrence": "earliest_forward_occurrence_after_previous_stop",
              "geometry_policy": {"min_change_length_m": 10., "max_join_gap_m": 2.,
                                  "change_sample_spacing_m": .25, "corridor_tolerance_m": 1e-6},
              "simulation_run": False, "fcd_or_g31_leg_metadata_read": False}
    requests, rows = [], []
    for route in routes:
        ident, edges = route.get("id"), route.get("edges").split()
        row = {"public_route_id": ident, "edge_occurrences": len(edges),
               "public_stop_count": len(route.findall("stop"))}
        try:
            first = min(network.permitted(edges[0], "bus"), key=lambda lane: int(network.lanes[lane]["attributes"]["index"]))
            last = min(network.permitted(edges[-1], "bus"), key=lambda lane: int(network.lanes[lane]["attributes"]["index"]))
            start, end = {"lane": first, "pos": 20.}, {"lane": last, "pos": network.length(last)}
            anchors, previous = [], (-1, -1.)
            for index, stop in enumerate(route.findall("stop")):
                location = locations[stop.get("busStop")]
                lane, pos = location["lane"], float(location["endPos"])
                edge = network.lanes[lane]["edge"]
                fraction = pos / network.length(lane)
                candidates = [occurrence for occurrence, item in enumerate(edges)
                              if item == edge and (occurrence, fraction) >= previous]
                if not candidates:
                    raise UnsupportedRoute("Public stop has no remaining ordered source edge occurrence")
                occurrence = min(candidates)
                anchors.append({"edge_index": occurrence, "lane": lane, "pos": pos, "source_stop_index": index})
                previous = (occurrence, fraction)
            plan = {"provenance": "public_input_only", "status": "complete_scheduled_source_service",
                    "kind": "vehicle", "edges": edges, "source_attributes": flows[ident],
                    "planned_stops": [dict(stop.attrib) for stop in route.findall("stop")]}
            request = request_from_source_plan(plan, "bus", start, end, anchors, locations,
                                              network_sha256=network.network_sha256)
            request["public_request_id"] = ident
            requests.append(request)
            row["status"] = "request_frozen"
        except (UnsupportedRoute, KeyError, ValueError) as error:
            row.update(status="source_request_unresolved", reason=str(error))
        rows.append(row)
    frozen = {"policy": policy, "requests": requests, "request_resolution": rows}
    request_path = output / "frozen-public-requests.json"
    request_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    request_hash = digest(request_path)
    construction_start = time.time()
    row_lookup = {row["public_route_id"]: row for row in rows}
    for request in requests:
        row = row_lookup[request["public_request_id"]]
        started = time.monotonic()
        try:
            geometry = network.construct(request)
            manifest = geometry.manifest()
            if len(manifest["stop_progress_m"]) != row["public_stop_count"]:
                raise AssertionError("Constructor lost a source stop")
            encoded = json.dumps(geometry.vertices, separators=(",", ":")).encode()
            row.update(status="static_geometry_constructed", length_m=geometry.length_m,
                       lane_changes=geometry.lane_changes, vertex_count=len(geometry.vertices),
                       retained_stops=len(manifest["stop_progress_m"]), geometry_sha256=hashlib.sha256(encoded).hexdigest())
        except UnsupportedRoute as error:
            row.update(status="static_geometry_unsupported", reason=str(error))
        row["elapsed_seconds"] = time.monotonic() - started
    if digest(request_path) != request_hash:
        raise AssertionError("Frozen request manifest changed during construction")
    result = {"complete": True, "scope": "static_geometry_for_explicit_prospective_public_bus_requests",
              "policy": policy, "denominator": 24, "requests_frozen_before_construction": True,
              "frozen_request_sha256": request_hash, "network_sha256": network.network_sha256,
              "source_manifest_sha256": digest(source / "complete.json"),
              "construction_started_unix": construction_start,
              "status_counts": dict(Counter(row["status"] for row in rows)), "routes": rows,
              "generation_ready": False, "simulation_dynamics_or_reference_alignment_validated": False}
    (output / "public-bus-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def public_bus_proof_frozen(source, frozen_path, output):
    """One v2 pass over the byte-identical v1 manifest; no endpoint refreezing."""
    source, frozen_path, output = map(lambda value: Path(value).resolve(), (source, frozen_path, output))
    if output.is_relative_to(source) or source.is_relative_to(output) or output == frozen_path.parent:
        raise ValueError("A new isolated proof output is required")
    output.mkdir(parents=True, exist_ok=False)
    before = digest(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    old = json.loads((frozen_path.parent / "public-bus-proof.json").read_text())
    if before != old["frozen_request_sha256"] or len(frozen["requests"]) != 24:
        raise ValueError("Frozen v1 request identity or denominator changed")
    net_path, _ = source_inputs(source)
    network = VehicleRouteNetwork(net_path)
    if network.network_sha256 != old["network_sha256"]:
        raise ValueError("Network changed between geometry implementations")
    rows = []
    for request in frozen["requests"]:
        started = time.monotonic()
        row = {"public_route_id": request["public_request_id"], "edge_occurrences": len(request["edges"]),
               "public_stop_count": len(request["stops"])}
        try:
            geometry = network.construct(request)
            manifest = geometry.manifest()
            if len(manifest["stop_progress_m"]) != row["public_stop_count"]:
                raise AssertionError("Constructor lost a source stop")
            windows = [window for piece in geometry.pieces for window in piece.get("lane_change_windows", [])]
            if any(min(window["longitudinal_geometry_m"]) < 10. - 1e-8 for window in windows):
                raise AssertionError("Local window violated geometric distance policy")
            encoded = json.dumps(geometry.vertices, separators=(",", ":")).encode()
            row.update(status="static_geometry_constructed", length_m=geometry.length_m,
                       lane_changes=geometry.lane_changes, vertex_count=len(geometry.vertices),
                       retained_stops=len(manifest["stop_progress_m"]), geometry_sha256=hashlib.sha256(encoded).hexdigest())
        except UnsupportedRoute as error:
            row.update(status="static_geometry_unsupported", reason=str(error))
        row["elapsed_seconds"] = time.monotonic() - started
        rows.append(row)
    if digest(frozen_path) != before:
        raise AssertionError("Frozen v1 manifest changed")
    result = {"complete": True, "scope": "static_geometry_same_24_frozen_public_bus_requests_v2",
              "frozen_request_sha256": before, "network_sha256": network.network_sha256,
              "source_manifest_sha256": digest(source / "complete.json"), "denominator": 24,
              "same_frozen_v1_requests": True, "fcd_or_g31_leg_metadata_read": False,
              "request_policy": frozen["policy"],
              "geometry_policy": {"min_change_length_m": 10., "change_length_units": "actual_geometric_arclength_on_both_lanes",
                                  "lane_change_placement": "earliest_covered_local_window",
                                  "max_join_gap_m": 2., "change_sample_spacing_m": .25, "corridor_tolerance_m": 1e-6},
              "status_counts": dict(Counter(row["status"] for row in rows)), "routes": rows,
              "generation_ready": False, "simulation_dynamics_or_reference_alignment_validated": False}
    (output / "public-bus-proof-v2.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen-requests", type=Path)
    args = parser.parse_args()
    if bool(args.public_source) != bool(args.output):
        parser.error("--public-source and --output must be supplied together")
    if args.frozen_requests and not args.public_source:
        parser.error("--frozen-requests requires --public-source and --output")
    if args.frozen_requests:
        result = public_bus_proof_frozen(args.public_source, args.frozen_requests, args.output)
    else:
        result = public_bus_proof(args.public_source, args.output) if args.public_source else run()
    print(json.dumps(result, indent=2))
