"""Audit public-input edge plans, without reading realized routes or FCD.

Static permission/connection checks are not a SUMO validity oracle. In particular
they neither resolve pedestrian walking areas nor model lateral lane changes,
stop-lane/end-position constraints, traffic, or signals. No generation occurs.
"""
import argparse
from collections import Counter, defaultdict, deque
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from g31_compare import children
from g31_replay import COMMIT, NETWORK_SHA, digest, save


CLASSES = ["pedestrian", "bus", "motorcycle", "private_car", "bycicle", "taxi_or_uber"]
EXPLICIT = {"explicit_source_walk", "complete_scheduled_source_service", "unique_source_activity_leg_slice"}


def allows(attributes, vclass):
    if vclass == "ignoring":
        return True
    if "allow" in attributes:
        values = attributes["allow"].split()
        return "all" in values or vclass in values
    values = attributes.get("disallow", "").split()
    return "all" not in values and vclass not in values


class SourceNetwork:
    def __init__(self, path):
        self.edges, self.lanes, self.by_index = {}, {}, {}
        self.successors = defaultdict(list)
        self.unresolved_connections = 0
        root = ET.parse(path).getroot()
        for edge in root.findall("edge"):
            edge_id = edge.get("id")
            self.edges[edge_id] = {"function": edge.get("function", "normal"), "lanes": []}
            for lane in edge.findall("lane"):
                ident = lane.get("id")
                points = [tuple(float(value) for value in point.split(",")[:2])
                          for point in lane.get("shape", "").split()]
                self.lanes[ident] = {"edge": edge_id, "attributes": dict(lane.attrib), "shape": points}
                self.edges[edge_id]["lanes"].append(ident)
                self.by_index[(edge_id, lane.get("index", "0"))] = ident
        for connection in root.findall("connection"):
            src = self.by_index.get((connection.get("from"), connection.get("fromLane")))
            dst = (connection.get("via") or
                   self.by_index.get((connection.get("to"), connection.get("toLane"))))
            if src not in self.lanes or dst not in self.lanes:
                self.unresolved_connections += 1
                continue
            self.successors[src].append((dst, dict(connection.attrib)))

    @lru_cache(maxsize=65536)
    def permitted(self, edge, vclass):
        return tuple(lane for lane in self.edges.get(edge, {}).get("lanes", [])
                     if allows(self.lanes[lane]["attributes"], vclass))

    def geometry_join(self, left, right):
        a, b = self.lanes[left]["shape"], self.lanes[right]["shape"]
        return (len(a) >= 2 and len(b) >= 2 and
                math.dist(a[-1], b[0]) <= 2.)

    @lru_cache(maxsize=65536)
    def transitions(self, source_edge, target_edge, vclass, geometry):
        """Lane relation through explicit permitted internal connections only."""
        relation = {}
        targets = set(self.permitted(target_edge, vclass))
        for source in self.permitted(source_edge, vclass):
            queue, seen, reached = deque([source]), {source}, set()
            while queue:
                current = queue.popleft()
                for target, attributes in self.successors[current]:
                    if current == source and attributes.get("to") != target_edge:
                        continue
                    if (not allows(attributes, vclass)
                            or not allows(self.lanes[target]["attributes"], vclass)
                            or (geometry and not self.geometry_join(current, target))):
                        continue
                    if target in targets:
                        reached.add(target)
                    elif (self.edges[self.lanes[target]["edge"]]["function"] == "internal"
                          and target not in seen):
                        seen.add(target)
                        queue.append(target)
            relation[source] = frozenset(reached)
        return relation

    def chain(self, edges, vclass, geometry=False):
        possible = set(self.permitted(edges[0], vclass))
        if geometry:
            possible = {lane for lane in possible if len(self.lanes[lane]["shape"]) >= 2}
        for left, right in zip(edges, edges[1:]):
            relation = self.transitions(left, right, vclass, geometry)
            possible = set().union(*(relation.get(lane, frozenset()) for lane in possible))
            if not possible:
                return False
        return bool(possible)

    def audit(self, plan, vclass):
        if plan.get("provenance") != "public_input_only":
            raise ValueError("Path audit accepts only public-input plans")
        edges = plan.get("edges", [])
        result = {"source_status": plan.get("status"), "edge_count": len(edges),
                  "repeated_edges": len(edges) != len(set(edges)),
                  "generation_ready": False, "vclass": vclass}
        if not edges:
            result["status"] = "requires_source_disambiguation_or_prior_planning"
            return result
        if plan.get("status") not in EXPLICIT:
            raise ValueError("Nonempty edge plan has an unrecognized public-input provenance status")
        if vclass is None:
            result["status"] = "source_vtype_unresolved"
            return result
        unknown = sum(edge not in self.edges for edge in edges)
        prohibited = sum(edge in self.edges and not self.permitted(edge, vclass) for edge in edges)
        result.update(unknown_edge_occurrences=unknown, edges_without_permitted_lane=prohibited)
        if unknown or prohibited:
            result["status"] = "source_edge_or_permission_unresolved"
            return result
        result["every_edge_has_permitted_lane"] = True
        if vclass == "pedestrian":
            result["status"] = "pedestrian_direction_walkingarea_and_endpoint_adapter_required"
            return result
        missing = sum(not any(self.transitions(left, right, vclass, False).values())
                      for left, right in zip(edges, edges[1:]))
        result["transitions_without_permitted_connection"] = missing
        result["pairwise_connections_exist"] = missing == 0
        result["continuous_lane_chain_exists"] = self.chain(edges, vclass)
        result["continuous_lane_chain_with_2m_joins_exists"] = self.chain(edges, vclass, True)
        if missing:
            result["status"] = "explicit_lane_connection_unresolved"
        elif not result["continuous_lane_chain_exists"]:
            result["status"] = "pairwise_valid_but_lateral_lane_change_adapter_required"
        elif not result["continuous_lane_chain_with_2m_joins_exists"]:
            result["status"] = "lane_chain_exists_but_geometry_join_adapter_required"
        else:
            result["status"] = "static_lane_chain_supported_endpoints_stops_and_reconstruction_unchecked"
        return result


def source_inputs(source):
    manifest = json.loads((source / "complete.json").read_text())
    if manifest.get("complete") is not True or manifest.get("commit") != COMMIT:
        raise ValueError("Pinned public source manifest is incomplete or changed")
    for name, expected in manifest["files"].items():
        if digest(source / name) != expected["sha256"]:
            raise ValueError("Public source checksum mismatch")
    net = source / "scenario/in/most.net.xml"
    if digest(net) != NETWORK_SHA:
        raise ValueError("Unexpected native MoST road network")
    config = ET.parse(source / "scenario/most.sumocfg").getroot()
    files = [source / "scenario" / name
             for field in ("route-files", "additional-files")
             for name in config.find("input/" + field).get("value").split(",")]
    types = {"DEFAULT_VEHTYPE": "passenger", "DEFAULT_PEDTYPE": "pedestrian", "DEFAULT_BIKETYPE": "bicycle"}
    for path in files:
        for node in children(path):
            for item in node.iter("vType"):
                if item.get("id") and item.get("vClass"):
                    ident, value = item.get("id"), item.get("vClass")
                    if ident in types and types[ident] != value:
                        raise ValueError("Conflicting public vType definitions")
                    types[ident] = value
    return net, types


def run(args):
    source, legs, output = [path.resolve() for path in (args.source, args.legs, args.output)]
    for protected in (source, legs):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Path-audit output must be separate from protected input trees")
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = json.loads((legs / "complete.json").read_text())
        if receipt.get("complete") is not True:
            raise ValueError("Completed leg-interface receipt is required")
        metadata_path = legs / "leg-metadata.json.gz"
        if digest(metadata_path) != receipt["artifacts"][metadata_path.name]["sha256"]:
            raise ValueError("Leg metadata changed after extraction")
        if digest(source / "complete.json") != receipt["sources"]["public_source_manifest_sha256"]:
            raise ValueError("Path audit and leg extraction use different public source manifests")
        net_path, types = source_inputs(source)
        network = SourceNetwork(net_path)
        with gzip.open(metadata_path, "rt", encoding="utf-8") as stream:
            accepted = json.load(stream)["accepted"]
        if len(accepted) != receipt["accepted_legs"]:
            raise ValueError("Leg metadata count disagrees with completion receipt")
        # Read only these fields. No emitted_route_audit_only, observed stops,
        # FCD, NPZ, realized edge sequence, exitTimes or priorEdges are consumed.
        results, per_class = [], {name: Counter() for name in CLASSES}
        source_counts, type_counts = Counter(), Counter()
        for leg in accepted:
            result = network.audit(leg["source_plan"], types.get(leg["vtype"]))
            label = int(leg["label"])
            if not 0 <= label < len(CLASSES) or leg["partition"] not in ("train", "val"):
                raise ValueError("Unexpected label or non-development parent partition")
            per_class[CLASSES[label]][result["status"]] += 1
            source_counts[result["source_status"]] += 1
            type_counts[result["vclass"] or "unknown"] += 1
            results.append({"leg_id": leg["leg_id"], "label": label,
                            "partition": leg["partition"], **result})
        with gzip.open(output / "path-audit.json.gz", "wt", encoding="utf-8") as stream:
            json.dump({"scope": "Server-only per-leg static source-plan audit; contains identifiers",
                       "legs": results}, stream)
        save(output / "complete.json", {
            "complete": True, "generation_ready": False, "legs": len(results),
            "scope": "Public-input static edge/lane graph audit only, not SUMO validity or trajectory reconstruction",
            "status_counts": dict(Counter(row["status"] for row in results)),
            "per_class": {name: dict(counts) for name, counts in per_class.items()},
            "source_status_counts": dict(source_counts), "actual_vclass_counts": dict(type_counts),
            "unresolved_network_connections": network.unresolved_connections,
            "sources": {"public_source_manifest_sha256": digest(source / "complete.json"),
                        "leg_receipt_sha256": digest(legs / "complete.json"),
                        "leg_metadata_sha256": digest(metadata_path), "network_sha256": digest(net_path),
                        "script_sha256": digest(Path(__file__))},
            "artifacts": {"path-audit.json.gz": {"sha256": digest(output / "path-audit.json.gz")}},
            "limitations": [
                "Only source_plan fields are used; no realized route or future FCD is a condition",
                "Actual vType selects SUMO vClass; uber remains passenger permission, not taxi by label",
                "Pairwise edge connectivity does not imply one lane chain without lateral changes",
                "Pedestrian bidirectionality/walking areas are not tested by vehicle connections",
                "Stop-lane order, source endpoint positions, lane changes and target reconstruction remain unchecked",
                "No all-day activity-chain or repeated-edge ambiguity is resolved from a future trajectory",
            ],
        })
        print(json.dumps({"complete": True, "generation_ready": False, "legs": len(results),
                          "status_counts": dict(Counter(row["status"] for row in results))}), flush=True)
    except BaseException as error:
        save(output / "failed.json", {"complete": False, "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../g30-job/g30-data/source"))
    parser.add_argument("--legs", type=Path, default=Path("g31-data/legs/pilot-300001"))
    parser.add_argument("--output", type=Path, default=Path("g31-data/paths/pilot-300001"))
    run(parser.parse_args())
