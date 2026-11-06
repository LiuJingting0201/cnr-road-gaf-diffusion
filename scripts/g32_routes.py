"""Decode a *pre-simulation* vehicle route request into permitted map geometry.

This is a point-trajectory corridor constructor, not a SUMO traffic simulator.
No realized route, FCD, target coordinates or target duration is accepted. The
caller must freeze the source edge occurrences, endpoint lane/positions and all
planned stop anchors before generating its reference simulation. Pedestrians,
OD-only plans and unresolved source policies need separate adapters.

SUMO lane positions use declared lane length; shape interpolation therefore uses
pos / declared_length, not pos as geometric arc length. Lane changes interpolate
adjacent lane shapes with a quintic blend. Every resulting line segment must lie
in the union of permitted lane-width corridors. This proves neither vehicle-body
clearance, feasibility in traffic, nor a dynamically feasible change duration.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import math

from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

from g31_paths import EXPLICIT, SourceNetwork, allows, digest


VEHICLE_CLASSES = {"bus", "motorcycle", "passenger", "bicycle", "taxi"}
SCHEMA = "g32-presimulation-route-v1"
PROVENANCE = "public_network_and_presimulation_request"


class UnsupportedRoute(ValueError):
    """A declared requirement cannot be resolved without changing the request."""


@dataclass
class RouteGeometry:
    vertices: list
    pieces: list
    lane_changes: int
    vclass: str
    edge_occurrences: list
    network_sha256: str
    geometry_policy: dict

    @property
    def length_m(self):
        return LineString(self.vertices).length

    def decode(self, progress_m):
        """Exact piecewise-linear decoding; no fixed-2049-node shortcut.

        Out-of-route progress is rejected so the generator cannot hide path
        exhaustion by clipping to the endpoint. The caller handles arrival.
        """
        line = LineString(self.vertices)
        values = [float(v) for v in progress_m]
        if any(not math.isfinite(v) or v < 0 or v > line.length for v in values):
            raise UnsupportedRoute("Progress outside the frozen route")
        return [tuple(line.interpolate(v).coords[0]) for v in values]

    def manifest(self):
        distance, stops = 0., []
        for piece in self.pieces:
            distance += piece["length_m"]
            if piece.get("source_stop_index") is not None:
                stops.append({"source_stop_index": piece["source_stop_index"], "progress_m": distance})
        return {"schema": SCHEMA, "scope": "vehicle_point_geometry_only",
                "vclass": self.vclass, "edges": self.edge_occurrences,
                "network_sha256": self.network_sha256,
                "geometry_policy": self.geometry_policy, "stop_progress_m": stops,
                "length_m": self.length_m, "lane_changes": self.lane_changes,
                "pieces": self.pieces, "vertices": self.vertices,
                "traffic_collision_signal_and_dynamics_validated": False,
                "fixed_route_condition_does_not_test_route_choice": True}


@dataclass
class _Node:
    cost: tuple
    previous: object
    piece: dict


def _coordinates(geometry):
    return [tuple(p) for p in geometry.coords]


def _append(target, points):
    for point in points:
        if not target or math.dist(target[-1], point) > 1e-10:
            target.append(tuple(point))


class VehicleRouteNetwork(SourceNetwork):
    def __init__(self, path, min_change_length_m=10., max_join_gap_m=2.,
                 change_sample_spacing_m=.25, corridor_tolerance_m=1e-6):
        super().__init__(path)
        self.network_sha256 = digest(path)
        self.min_change_length_m = float(min_change_length_m)
        self.max_join_gap_m = float(max_join_gap_m)
        self.change_sample_spacing_m = float(change_sample_spacing_m)
        self.corridor_tolerance_m = float(corridor_tolerance_m)
        if (self.min_change_length_m <= 0 or self.max_join_gap_m < 0
                or self.change_sample_spacing_m <= 0 or self.corridor_tolerance_m < 0):
            raise ValueError("Invalid fixed geometry policy")
        self.shapes = {key: LineString(value["shape"])
                       for key, value in self.lanes.items() if len(value["shape"]) >= 2}

    def length(self, lane):
        value = float(self.lanes[lane]["attributes"].get("length", "nan"))
        if not math.isfinite(value) or value <= 0 or lane not in self.shapes:
            raise UnsupportedRoute("Lane length or geometry missing: " + lane)
        return value

    def fraction(self, anchor, edge):
        lane, pos = anchor.get("lane"), anchor.get("pos")
        if lane not in self.lanes or self.lanes[lane]["edge"] != edge:
            raise UnsupportedRoute("Anchor lane does not belong to its edge occurrence")
        if isinstance(pos, bool) or not isinstance(pos, (int, float)):
            raise UnsupportedRoute("Anchor position must be explicitly frozen in metres")
        length = self.length(lane)
        if not math.isfinite(pos) or not 0 <= pos <= length:
            raise UnsupportedRoute("Anchor position outside declared lane length")
        return float(pos) / length

    @lru_cache(maxsize=32768)
    def corridor(self, lanes):
        # Width omitted in SUMO net XML means the standard 3.2 m lane width.
        polygons = []
        for lane in lanes:
            width = float(self.lanes[lane]["attributes"].get("width", "3.2"))
            if not math.isfinite(width) or width <= 0:
                raise UnsupportedRoute("Missing or invalid lane width")
            polygons.append(self.shapes[lane].buffer(width / 2))
        return unary_union(polygons).buffer(self.corridor_tolerance_m)

    def covered(self, coords, lanes):
        shape = LineString(coords) if len(coords) > 1 else Point(coords[0])
        return self.corridor(tuple(lanes)).covers(shape)

    @lru_cache(maxsize=32768)
    def lateral_lanes(self, source, target, vclass):
        edge = self.lanes[source]["edge"]
        if self.lanes[target]["edge"] != edge:
            return None
        a = int(self.lanes[source]["attributes"].get("index", "0"))
        b = int(self.lanes[target]["attributes"].get("index", "0"))
        step = 1 if b >= a else -1
        ids = [self.by_index.get((edge, str(i))) for i in range(a, b + step, step)]
        if any(lane is None or not allows(self.lanes[lane]["attributes"], vclass)
               for lane in ids):
            return None
        rule = "changeLeft" if step == 1 else "changeRight"
        for lane in ids[:-1]:
            attrs = self.lanes[lane]["attributes"]
            # Missing rule permits all classes. Empty or 'none' permits none.
            if rule in attrs and vclass not in attrs[rule].split() and "all" not in attrs[rule].split():
                return None
        return tuple(ids)

    def lane_slice(self, lane, start, end):
        shape = self.shapes[lane]
        return _coordinates(substring(shape, start * shape.length, end * shape.length))

    def blend(self, left, right, lo, hi):
        length = max(self.shapes[left].length, self.shapes[right].length) * (hi - lo)
        count = max(8, int(math.ceil(length / self.change_sample_spacing_m)))
        coords = []
        for sample in range(count + 1):
            t = sample / count
            blend = t * t * t * (10 + t * (-15 + 6 * t))
            u = lo + (hi - lo) * t
            a = self.shapes[left].interpolate(u, normalized=True)
            b = self.shapes[right].interpolate(u, normalized=True)
            _append(coords, [(a.x * (1 - blend) + b.x * blend,
                              a.y * (1 - blend) + b.y * blend)])
        return coords

    def on_edge(self, source, target, start, end, vclass, occurrence):
        if end < start - 1e-10:
            return None
        lanes = self.lateral_lanes(source, target, vclass)
        if not lanes:
            return None
        changes = len(lanes) - 1
        spans = [self.min_change_length_m / min(self.shapes[a].length, self.shapes[b].length)
                 for a, b in zip(lanes, lanes[1:])]
        if sum(spans) > end - start + 1e-12:
            return None
        coords, windows = [], []
        if not changes:
            coords = self.lane_slice(source, start, end)
        else:
            current = start
            for index, (left, right) in enumerate(zip(lanes, lanes[1:])):
                span = spans[index]
                latest = end - span - sum(spans[index + 1:])
                step = self.change_sample_spacing_m / max(self.shapes[left].length, self.shapes[right].length)
                count = max(0, int(math.floor((latest - current) / step)))
                starts = [current + item * step for item in range(count + 1)]
                if latest > starts[-1] + 1e-12:
                    starts.append(latest)
                for lo in starts:
                    hi = lo + span
                    candidate = self.blend(left, right, lo, hi)
                    if self.covered(candidate, (left, right)):
                        _append(coords, self.lane_slice(left, current, lo))
                        _append(coords, candidate)
                        windows.append({"lanes": [left, right], "start_fraction": lo,
                                        "end_fraction": hi,
                                        "longitudinal_geometry_m": [(hi - lo) * self.shapes[lane].length
                                                                    for lane in (left, right)]})
                        current = hi
                        break
                else:
                    return None
            _append(coords, self.lane_slice(lanes[-1], current, end))
        if not coords or not self.covered(coords, lanes):
            return None
        distance = LineString(coords).length if len(coords) > 1 else 0.
        return {"kind": "lane_change" if changes else "lane_follow",
                "edge_occurrence": occurrence, "lanes": list(lanes),
                "start_fraction": start, "end_fraction": end,
                "lane_change_windows": windows,
                "lane_changes": changes, "length_m": distance, "xy": coords}

    @lru_cache(maxsize=32768)
    def connections(self, source, target_edge, vclass):
        """Keep internal junction lanes; no straight-line intersection shortcut."""
        queue = deque([(source, (source,), ())])
        seen, found = {source}, {}
        targets = set(self.permitted(target_edge, vclass))
        while queue:
            current, chain, rules = queue.popleft()
            for target, attrs in self.successors[current]:
                if current == source and attrs.get("to") != target_edge:
                    continue
                if (not allows(attrs, vclass) or target not in self.shapes
                        or not allows(self.lanes[target]["attributes"], vclass)):
                    continue
                gap = math.dist(self.lanes[current]["shape"][-1], self.lanes[target]["shape"][0])
                if gap > self.max_join_gap_m:
                    continue
                next_chain = chain + (target,)
                next_rules = rules + ({key: attrs[key] for key in
                                      ("from", "to", "fromLane", "toLane", "via", "allow", "disallow")
                                      if key in attrs},)
                if target in targets and target not in found:
                    coords = [self.lanes[source]["shape"][-1]]
                    for internal in next_chain[1:-1]:
                        _append(coords, self.lanes[internal]["shape"])
                    _append(coords, [self.lanes[target]["shape"][0]])
                    if not self.covered(coords, next_chain):
                        continue
                    found[target] = {"kind": "connection", "lanes": list(next_chain),
                                     "connection_rules": list(next_rules), "lane_changes": 0,
                                     "length_m": LineString(coords).length if len(coords) > 1 else 0.,
                                     "xy": coords}
                elif (target not in seen and self.edges[self.lanes[target]["edge"]]["function"] == "internal"):
                    seen.add(target)
                    queue.append((target, next_chain, next_rules))
        return found

    @staticmethod
    def extend(node, piece):
        return _Node((node.cost[0] + piece["lane_changes"], node.cost[1] + piece["length_m"]), node, piece)

    def construct(self, request):
        if (request.get("schema") != SCHEMA or request.get("provenance") != PROVENANCE
                or request.get("frozen_before_simulation") is not True):
            raise UnsupportedRoute("A frozen prospective request is required")
        if request.get("network_sha256") != self.network_sha256:
            raise UnsupportedRoute("Request must bind the exact public network hash")
        vclass = request.get("vclass")
        if vclass not in VEHICLE_CLASSES:
            raise UnsupportedRoute("Pedestrian walking directions/areas and other classes are outside this adapter")
        edges = request.get("edges", [])
        if not edges or any(edge not in self.edges or self.edges[edge]["function"] != "normal" for edge in edges):
            raise UnsupportedRoute("An explicit ordered source edge plan is required")
        start, end = request.get("start", {}), request.get("end", {})
        first, last = self.fraction(start, edges[0]), self.fraction(end, edges[-1])
        source_attributes = request.get("source_attributes", {})
        for name, anchor in (("depart", start), ("arrival", end)):
            if not request.get("source_endpoint_constraints", {}).get(name, False):
                continue
            lane_rule = source_attributes.get(name + "Lane", "")
            if lane_rule.isdigit() and int(lane_rule) != int(self.lanes[anchor["lane"]]["attributes"].get("index", "0")):
                raise UnsupportedRoute("Endpoint lane contradicts a literal source constraint")
            try:
                source_pos = float(source_attributes.get(name + "Pos", ""))
            except ValueError:
                continue
            if source_pos < 0:
                source_pos += self.length(anchor["lane"])
            if not math.isfinite(source_pos) or abs(source_pos - anchor["pos"]) > 1e-6:
                raise UnsupportedRoute("Endpoint position contradicts a literal source constraint")
        anchors = [[] for _ in edges]
        previous = (-1, -1.)
        for stop in request.get("stops", []):
            index = stop.get("edge_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(edges):
                raise UnsupportedRoute("Each planned stop needs an explicit edge occurrence")
            pos = self.fraction(stop, edges[index])
            if (index, pos) < previous:
                raise UnsupportedRoute("Planned stop order contradicts the source route")
            previous = (index, pos)
            anchors[index].append((stop["lane"], pos, stop.get("source_stop_index")))
        root = _Node((0, 0.), None, None)
        states = {(start["lane"], first): root}
        for occurrence, edge in enumerate(edges):
            for target_lane, fraction, stop_index in anchors[occurrence]:
                states = self.advance(states, [(target_lane, fraction)], vclass, occurrence)
                if not states:
                    raise UnsupportedRoute("Planned stop cannot be reached on its permitted lane")
                for node in states.values():
                    node.piece["source_stop_index"] = stop_index
            goals = ([(end["lane"], last)] if occurrence == len(edges) - 1 else
                     [(lane, 1.) for lane in self.permitted(edge, vclass)])
            states = self.advance(states, goals, vclass, occurrence)
            if not states:
                raise UnsupportedRoute("No corridor route satisfies endpoints/stops/lateral permission at edge occurrence " + str(occurrence))
            if occurrence + 1 < len(edges):
                following = {}
                for (lane, _), node in states.items():
                    for target, piece in self.connections(lane, edges[occurrence + 1], vclass).items():
                        updated = self.extend(node, dict(piece, from_edge_occurrence=occurrence))
                        key = (target, 0.)
                        if key not in following or updated.cost < following[key].cost:
                            following[key] = updated
                states = following
                if not states:
                    raise UnsupportedRoute("No permitted internal connection at edge occurrence " + str(occurrence))
        node = min(states.values(), key=lambda value: value.cost)
        lane_changes, pieces = node.cost[0], []
        while node.previous is not None:
            pieces.append(node.piece)
            node = node.previous
        pieces.reverse()
        vertices = []
        for piece in pieces:
            _append(vertices, piece["xy"])
        if len(vertices) < 2 or LineString(vertices).length <= 0:
            raise UnsupportedRoute("Zero-length geometry has no motion decoder")
        policy = {"min_change_length_m": self.min_change_length_m,
                  "change_length_units": "actual_geometric_arclength_on_both_lanes",
                  "lane_change_placement": "earliest_covered_local_window",
                  "max_join_gap_m": self.max_join_gap_m,
                  "change_sample_spacing_m": self.change_sample_spacing_m,
                  "corridor_tolerance_m": self.corridor_tolerance_m,
                  "missing_lane_width_m": 3.2}
        return RouteGeometry(vertices, pieces, lane_changes, vclass, list(edges), self.network_sha256, policy)

    def advance(self, states, goals, vclass, occurrence):
        following = {}
        for (source, start), node in states.items():
            for target, end in goals:
                piece = self.on_edge(source, target, start, end, vclass, occurrence)
                if piece is None:
                    continue
                updated = self.extend(node, piece)
                key = (target, end)
                if key not in following or updated.cost < following[key].cost:
                    following[key] = updated
        return following


def request_from_source_plan(plan, vclass, start, end, stops, locations, ordinal=None, *, network_sha256):
    """Bind public source constraints to newly fixed request parameters.

    ``stops`` must include one explicit occurrence/lane/position per public bus
    stop, in source order. ``locations`` are public additional-file attributes
    indexed by stop/parking-area id, never realized stop events. Source random,
    best/current and omitted endpoint policies are instantiated prospectively by
    the caller, and must be written to the new SUMO demand as these same values.
    The resulting request does not recover original pilot endpoint realizations.
    """
    if (plan.get("provenance") != "public_input_only" or plan.get("status") not in EXPLICIT
            or plan.get("kind") == "walk" or not plan.get("edges")):
        raise UnsupportedRoute("OD-only/ambiguous/walking plans require a different planner")
    expected = plan.get("planned_stops", [])
    if len(stops) != len(expected):
        raise UnsupportedRoute("All source service stops must be explicitly anchored")
    for index, (source, anchor) in enumerate(zip(expected, stops)):
        ident = source.get("busStop") or source.get("parkingArea") or source.get("containerStop")
        location = locations.get(ident, {}) if ident else {}
        lane = source.get("lane") or location.get("lane")
        if not lane or anchor.get("source_stop_index") != index or anchor.get("lane") != lane:
            raise UnsupportedRoute("Planned stop source reference/lane not resolved")
        lower = source.get("startPos", location.get("startPos"))
        upper = source.get("endPos", source.get("pos", location.get("endPos")))
        pos = anchor.get("pos")
        if lower is None or upper is None or not isinstance(pos, (int, float)) or not float(lower) <= pos <= float(upper):
            raise UnsupportedRoute("Stop anchor must be fixed inside its public source interval")
    boundaries = plan.get("planned_activity_boundaries", [])
    if boundaries:
        if ordinal is None or not 0 <= ordinal <= len(boundaries):
            raise UnsupportedRoute("Activity leg ordinal must be known from the public plan")
        for anchor, boundary in ((start, boundaries[ordinal - 1] if ordinal else None),
                                 (end, boundaries[ordinal] if ordinal < len(boundaries) else None)):
            if boundary is None:
                continue
            if (anchor.get("lane") != boundary.get("lane") or boundary.get("position") is None
                    or abs(float(anchor.get("pos", math.inf)) - float(boundary["position"])) > 1e-6):
                raise UnsupportedRoute("Activity endpoint must match its public stop constraint")
    return {"schema": SCHEMA, "provenance": PROVENANCE, "frozen_before_simulation": True,
            "network_sha256": network_sha256,
            "edges": list(plan["edges"]), "vclass": vclass,
            "start": dict(start), "end": dict(end), "stops": [dict(stop) for stop in stops],
            "source_status": plan["status"], "source_attributes": plan.get("source_attributes", {}),
            "public_planned_stops": expected,
            "source_endpoint_constraints": {"depart": not boundaries or ordinal == 0,
                                            "arrival": not boundaries or ordinal == len(boundaries)},
            "endpoint_policy": "explicit_presimulation_instantiation_written_to_new_demand",
            "original_pilot_endpoint_reconstruction": False}
