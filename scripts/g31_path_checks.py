"""Five synthetic static-path checks; no real network or simulation required.

Run directly with ``python scripts/g31_path_checks.py``. Each synthetic network
is written inside an automatically removed temporary directory.
"""
from pathlib import Path
import tempfile
import unittest

from g31_paths import SourceNetwork


class G31PathChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cnr-g31-paths-check-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "net.xml"

    def network(self, connections):
        self.path.write_text(
            '<net><edge id="a"><lane id="a_0" index="0" allow="passenger bus" shape="0,0 1,0"/></edge>'
            '<edge id=":ab" function="internal">'
            '<lane id=":ab_0" index="0" allow="passenger bus" shape="1,0 2,0"/></edge>'
            '<edge id="b"><lane id="b_0" index="0" allow="passenger bus" shape="2,0 3,0"/>'
            '<lane id="b_1" index="1" allow="passenger bus" shape="2,1 3,1"/></edge>'
            '<edge id="c"><lane id="c_0" index="0" allow="passenger bus" shape="3,1 4,1"/></edge>'
            '<edge id="walk"><lane id="walk_0" index="0" allow="pedestrian" shape="0,2 4,2"/></edge>'
            + connections + '</net>'
        )
        return SourceNetwork(self.path)

    def plan(self, edges, status="unique_source_activity_leg_slice"):
        return {"provenance": "public_input_only", "status": status, "edges": edges}

    def test_internal_chain(self):
        network = self.network(
            '<connection from="a" to="b" fromLane="0" toLane="0" via=":ab_0"/>'
            '<connection from=":ab" to="b" fromLane="0" toLane="0"/>'
        )
        result = network.audit(self.plan(["a", "b"]), "passenger")
        self.assertTrue(result["continuous_lane_chain_with_2m_joins_exists"])
        self.assertFalse(result["generation_ready"])

    def test_pairwise_is_not_whole_chain(self):
        network = self.network(
            '<connection from="a" to="b" fromLane="0" toLane="0" via=":ab_0"/>'
            '<connection from=":ab" to="b" fromLane="0" toLane="0"/>'
            '<connection from="b" to="c" fromLane="1" toLane="0"/>'
        )
        result = network.audit(self.plan(["a", "b", "c"]), "passenger")
        self.assertTrue(result["pairwise_connections_exist"])
        self.assertFalse(result["continuous_lane_chain_exists"])
        self.assertEqual(result["status"], "pairwise_valid_but_lateral_lane_change_adapter_required")

    def test_connection_permission(self):
        network = self.network('<connection from="a" to="b" fromLane="0" toLane="0" disallow="bus"/>')
        self.assertFalse(network.audit(self.plan(["a", "b"]), "bus")["pairwise_connections_exist"])
        self.assertTrue(network.audit(self.plan(["a", "b"]), "passenger")["pairwise_connections_exist"])

    def test_pedestrian_and_unplanned(self):
        network = self.network("")
        result = network.audit(self.plan(["walk"], "explicit_source_walk"), "pedestrian")
        self.assertTrue(result["every_edge_has_permitted_lane"])
        self.assertNotIn("pairwise_connections_exist", result)
        self.assertEqual(network.audit(self.plan([]), "passenger")["status"],
                         "requires_source_disambiguation_or_prior_planning")
        with self.assertRaises(ValueError):
            network.audit({"provenance": "realized_route", "edges": ["a"]}, "passenger")

    def test_geometry_and_missing_edge(self):
        network = self.network('<connection from="a" to="b" fromLane="0" toLane="0"/>')
        network.lanes["b_0"]["shape"] = [(100, 0), (101, 0)]
        result = network.audit(self.plan(["a", "b"]), "passenger")
        self.assertTrue(result["continuous_lane_chain_exists"])
        self.assertFalse(result["continuous_lane_chain_with_2m_joins_exists"])
        self.assertEqual(network.audit(self.plan(["missing"]), "passenger")["unknown_edge_occurrences"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
