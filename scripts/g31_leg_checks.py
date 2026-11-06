"""Synthetic G31 leg-boundary checks; no simulator or real/test trajectories.

Run with the same PYTHONPATH as g31_legs.py. Temporary XML fixtures are removed.
"""
from argparse import Namespace
import gzip
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

import g31_legs as legs


def plan(text):
    return legs.node_plan(ET.fromstring(text))


def fixture(source_xml=None, emitted_xml=None, tag="vehicle", label=3,
            depart="0", arrival="2", base="synthetic", vtype="passenger1"):
    source_xml = source_xml or '<vehicle id="synthetic"><route edges="a b c"/></vehicle>'
    emitted_xml = emitted_xml or '<vehicle id="synthetic"><route edges="a b c" exitTimes=".5 1 2"/></vehicle>'
    source_plan = plan(source_xml)
    parent = {"tag": tag, "base_id": base, "partition": "train", "label": label,
              "vtype": vtype, "tripinfo": {"depart": depart, "arrival": arrival},
              "stages": [], "stop_events": [], "emitted": plan(emitted_xml)}
    source = {"entities": {(tag, base): source_plan}, "flows": {}, "routes": {},
              "locations": {("parkingArea", "p"): {"lane": "b_0", "endPos": "20"}}}
    return parent, source


def observation(times=(0, .25, .5), speed=1., pos=2.):
    return {"values": np.asarray([[t, t, 0, speed, pos] for t in times], dtype=float),
            "roads": np.asarray(["a_0"]*len(times)), "types": {"passenger1"}}


class LegBoundaryChecks(unittest.TestCase):
    def test_unfinished_activity_keeps_completed_approach(self):
        parent, source = fixture('<vehicle id="synthetic"><route edges="a b c"/>'
                                 '<stop parkingArea="p" duration="100"/></vehicle>', arrival="-1")
        parent["stop_events"] = [{"parkingArea": "p", "started": "1", "ended": "-1"}]
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertEqual([(x["event_start"], x["event_end"]) for x in rows], [(0, 1)])
        self.assertEqual(rows[0]["source_plan"]["edges"], ["a", "b"])
        self.assertEqual(rejected["activity_stop_still_ongoing_no_following_departure"], 1)

    def test_completed_activity_splits_parent_without_cross_partition(self):
        parent, source = fixture('<vehicle id="synthetic"><route edges="a b c"/>'
                                 '<stop parkingArea="p" duration="100"/></vehicle>', arrival="3")
        parent["partition"] = "val"
        parent["stop_events"] = [{"parkingArea": "p", "started": "1", "ended": "2"}]
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertFalse(rejected)
        self.assertEqual([(x["event_start"], x["event_end"]) for x in rows], [(0, 1), (2, 3)])
        self.assertEqual({x["partition"] for x in rows}, {"val"})
        self.assertEqual({x["parent_base_id"] for x in rows}, {"synthetic"})
        self.assertEqual(rows[1]["source_plan"]["edges"], ["b", "c"])

    def test_unknown_stop_and_changed_parking_are_not_guessed(self):
        parent, source = fixture('<vehicle id="synthetic"><route edges="a b c"/>'
                                 '<stop lane="b_0" duration="10"/></vehicle>')
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertFalse(rows)
        self.assertEqual(rejected["ordinary_source_stop_semantics_ambiguous"], 1)
        parent, source = fixture('<vehicle id="synthetic"><route edges="a b c"/>'
                                 '<stop parkingArea="p" duration="10"/></vehicle>')
        parent["stop_events"] = [{"parkingArea": "other", "started": "1", "ended": "1.5"}]
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertFalse(rows)
        self.assertEqual(rejected["activity_stop_source_location_mismatch"], 1)

    def test_bus_retains_stops_as_one_completed_service(self):
        parent, source = fixture('<vehicle id="synthetic"><route edges="a b c"/>'
                                 '<stop busStop="station" duration="30"/></vehicle>', label=1, vtype="bus")
        parent["stop_events"] = [{"busStop": "station", "started": ".5", "ended": "1"}]
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "bus_service")
        self.assertEqual(rows[0]["intermediate_stop_events"], parent["stop_events"])
        self.assertEqual(rows[0]["source_plan"]["edges"], ["a", "b", "c"])
        parent["tripinfo"]["arrival"] = "-1"
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertFalse(rows)
        self.assertEqual(rejected["bus_service_unfinished"], 1)

    def test_only_complete_walk_stages_with_source_alignment(self):
        parent, source = fixture('<person id="synthetic"><walk edges="a b"/>'
                                 '<ride from="b" to="c" lines="bus"/><walk edges="c d"/></person>',
                                 '<person id="synthetic"><walk edges="a b" started="0" ended="1"/>'
                                 '<ride started="1" ended="2"/><walk edges="c d" started="2" ended="-1"/></person>',
                                 tag="person", label=0, vtype="avgpedestrian")
        parent["stages"] = [{"tag": "walk", "attributes": {"depart": "0", "arrival": "1"}},
                            {"tag": "ride", "attributes": {"depart": "1", "arrival": "2"}},
                            {"tag": "walk", "attributes": {"depart": "2", "arrival": "-1"}}]
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_plan"]["edges"], ["a", "b"])
        self.assertEqual(rejected["walk_stage_not_completed"], 1)
        source["entities"][("person", "synthetic")]["stages"].pop()
        rows, rejected = legs.candidate_legs(parent, source)
        self.assertFalse(rows)
        self.assertEqual(rejected["source_walk_stage_alignment_ambiguous"], 1)

    def test_source_repeat_ambiguity_never_uses_realized_exit_times(self):
        parent, source = fixture('<vehicle id="synthetic"><route edges="a b a c"/>'
                                 '<stop lane="a_0" parking="true"/></vehicle>',
                                 '<vehicle id="synthetic"><route edges="a b a c" exitTimes="1 2 3 4"/></vehicle>')
        source_path = legs.source_leg_plan(source["entities"][("vehicle", "synthetic")], source,
                                           0, False, "vehicle_activity_leg")
        self.assertEqual(source_path["status"], "ambiguous_source_activity_leg_slice")
        self.assertEqual(source_path["source_end_index_candidates"], [0, 2])
        self.assertFalse(source_path["edges"])
        points = np.asarray([[2.25, 0, 0, 1, 1], [2.5, 1, 0, 1, 2]])
        audit = legs.realized_leg_audit({"kind": "vehicle_activity_leg"}, parent, points, ["a_0", "a_0"])
        self.assertEqual(audit["parent_edge_instance_indices"], [2])
        self.assertFalse(source_path["edges"])
        self.assertFalse(source_path["condition_ready"])

    def test_optional_nonfinite_fields_retained_and_counted(self):
        candidate = {"event_start": 0, "event_end": .5, "vtype": "passenger1"}
        result, reason, info = legs.cover(candidate, observation(speed=np.nan, pos=np.inf))
        self.assertIsNone(reason)
        self.assertEqual(info["missing_native_speed_points"], 3)
        self.assertEqual(info["missing_position_on_lane_points"], 3)
        self.assertTrue(np.isnan(result[0][:, 3]).all())
        self.assertEqual(info["observed_span_s"], .5)

    def test_bad_coordinates_rejected(self):
        obs = observation()
        obs["values"][1, 1] = np.nan
        _, reason, _ = legs.cover({"event_start": 0, "event_end": .5, "vtype": "passenger1"}, obs)
        self.assertEqual(reason, "invalid_FCD_values_or_inconsistent_vtype")

    def test_internal_gap_and_endpoint_failure_are_distinct(self):
        candidate = {"event_start": 0, "event_end": .75, "vtype": "passenger1"}
        _, reason, _ = legs.cover(candidate, observation(times=(0, .25, .75)))
        self.assertEqual(reason, "FCD_gap_inside_leg")
        _, reason, _ = legs.cover(candidate, observation(times=(.5, .75)))
        self.assertEqual(reason, "FCD_does_not_cover_event_endpoints")

    def test_tolerance_does_not_interpolate_event_endpoints(self):
        candidate = {"event_start": 0, "event_end": 1, "vtype": "passenger1"}
        result, reason, info = legs.cover(candidate, observation(times=(.25, .5, .75)))
        self.assertIsNone(reason)
        self.assertEqual(result[0][:, 0].tolist(), [.25, .5, .75])
        self.assertEqual(info["event_duration_s"], 1)
        self.assertEqual(info["observed_span_s"], .5)

    def test_fcd_keeps_walking_not_nested_or_direct_riding(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"fcd.xml.gz"
            with gzip.open(path, "wt") as stream:
                stream.write('<fcd-export><timestep time="0"><vehicle id="bus" x="0" y="0">'
                             '<person id="walker" x="0" y="0"/></vehicle>'
                             '<person id="walker" vehicle="bus" x="0" y="0"/></timestep>'
                             '<timestep time=".25"><person id="walker" type="avgpedestrian" '
                             'x="1" y="0" edge="a"/></timestep></fcd-export>')
            observations, counts = legs.collect_fcd(path, {("person", "walker")})
            rows = observations[("person", "walker")]["values"]
            self.assertEqual(rows.shape, (1, 5))
            self.assertTrue(np.isnan(rows[:, 3:]).all())
            self.assertEqual(counts["direct_rider_observations_ignored"], 1)

    def test_original_test_templates_excluded_before_events(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder/"tripinfo.xml").write_text('<root><tripinfo id="test-parent"/><tripinfo id="train-parent"/></root>')
            with gzip.open(folder/"vehroute.xml.gz", "wt") as stream:
                stream.write('<routes><vehicle id="test-parent"><route edges="secret"/></vehicle>'
                             '<vehicle id="train-parent"><route edges="a b"/></vehicle></routes>')
            (folder/"stops.xml").write_text('<stops><stopinfo id="test-parent" started="0"/></stops>')
            parents, rejected = legs.parent_events(folder, {"test-parent": "test", "train-parent": "train"})
            self.assertEqual(set(parents), {("vehicle", "train-parent")})
            self.assertEqual(rejected["original_test_templates_excluded_before_leg_extraction"], 1)

    def test_existing_output_refused_before_real_input_access(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileExistsError):
                legs.extract(Namespace(output=Path(name)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
