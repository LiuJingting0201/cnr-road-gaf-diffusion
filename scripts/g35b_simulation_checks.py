"""Focused stdlib checks for G35b source-only FCD selection and config safety."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import g35b_simulate as g


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / "source", self.root / "budget" / "attempt"
        self.output.mkdir(parents=True)
        scenario = self.source / "scenario"
        (scenario / "in/route").mkdir(parents=True)
        (scenario / "in/add").mkdir()
        (scenario / "in/most.net.xml").write_text("<net/>")
        (scenario / "in/add/basic.vType.xml").write_text('<additional><vType id="bus" vClass="bus"/></additional>')
        (scenario / "most.sumocfg").write_text('''<configuration><input>
            <net-file value="in/most.net.xml"/>
            <route-files value="in/route/most.pedestrian.rou.xml,in/route/most.buses.flows.xml,in/route/most.highway.flows.xml"/>
            <additional-files value="in/add/basic.vType.xml"/></input>
            <time><begin value="14400"/><end value="50400"/><step-length value=".25"/></time>
            <processing><lateral-resolution value="0.3"/><time-to-teleport value="120"/></processing>
            <routing><device.rerouting.probability value="1"/></routing></configuration>''')
        self.parent = next(f"parent{i}" for i in range(100) if g.assigned_train_seed(f"parent{i}") == 350101)
        (scenario / "in/route/most.pedestrian.rou.xml").write_text(
            f'<routes><vehicle id="{self.parent}" type="ptw" depart="triggered"/>'
            '<vehicle id="testParent" type="passenger" depart="triggered"/>'
            '<vehicle id="valParent" type="bicycle" depart="triggered"/>'
            '<person id="person" type="pedestrian" depart="18000"/></routes>')
        (scenario / "in/route/most.buses.flows.xml").write_text(
            '<routes><flow id="bus_line" type="bus" begin="14400" end="16200" period="900"/></routes>')
        (scenario / "in/route/most.highway.flows.xml").write_text(
            '<routes><flow id="highway" type="highway" begin="18000" end="18060" period="30"/></routes>')
        self.partitions = {self.parent: "train", "testParent": "test", "valParent": "val"}
        self.splits = self.root / "splits.json"
        self.splits.write_text(json.dumps({"split_by_id": self.partitions}))

    def inputs(self):
        ids, _ = g.plan_recording(self.source, self.splits, 350101, "train")
        (self.output / g.ID_FILE).write_text("\n".join(ids) + "\n")
        g.build_override(self.source, self.output, ids)
        (self.output / "sumo-parsed.sumocfg").write_text("fixture")
        (self.output / "config-parse.log").write_text("fixture")
        config = {"source": str(self.source), "baseline_source_config_sha256": g.digest(self.source / "scenario/most.sumocfg"),
            "recording": {"vehicle_ids_count": len(ids), "vehicle_ids_sha256": g.digest(self.output / g.ID_FILE)},
            "input_artifacts": {name: {"sha256": g.digest(self.output / name), "bytes": (self.output / name).stat().st_size}
                                for name in g.INPUT_ARTIFACTS}}
        return config

    def test_bus_zero_based_period_boundaries(self):
        row = {"id": "bus_M2:France", "begin": "14423", "end": "50423", "period": "900"}
        ids = g.flow_ids(row)
        self.assertEqual(len(ids), 40)
        self.assertEqual(ids[0], "bus_M2:France.0")
        self.assertEqual(ids[-1], "bus_M2:France.39")
        self.assertEqual(len(g.flow_ids(dict(row, id="highway", begin="18000", end="41400", period="30"))), 780)

    def test_reject_random_or_unsupported_flow(self):
        row = {"id": "bus", "begin": "14400", "end": "50400", "period": "900"}
        for changed in (dict(row, probability=".5"), dict(row, begin="0"), dict(row, period="exp(3)")):
            with self.assertRaises(Exception):
                g.flow_ids(changed)

    def test_g30_partition_bytes_and_inherited_test(self):
        for ident in ("a", "bus_M2:France.39", "France:Italy.12"):
            raw = json.dumps([30042, "entity-partition", ident], ensure_ascii=True, separators=(",", ":")).encode()
            expected = "val" if int.from_bytes(hashlib.sha256(raw).digest(), "big") < 2**256 // 5 else "train"
            self.assertEqual(g.entity_partition(ident, {}), expected)
        self.assertEqual(g.entity_partition("testParent", self.partitions), "test")

    def test_selection_preserves_potential_type_families(self):
        ids, report = g.plan_recording(self.source, self.splits, 350101, "train")
        self.assertIn(self.parent, ids)
        self.assertNotIn("testParent", ids)
        self.assertNotIn("valParent", ids)
        self.assertEqual(report["candidate_family_counts"]["highway"], 2)
        self.assertEqual(report["candidate_family_counts"]["bus"], 2)
        self.assertEqual(report["excluded_identity_counts"]["original_test_parent"], 1)

    def test_three_episode_identity_sets_disjoint(self):
        selected = [set(g.plan_recording(self.source, self.splits, seed, split)[0])
                    for seed, split in g.EPISODES.items()]
        for i in range(3):
            for j in range(i):
                self.assertFalse(selected[i] & selected[j])

    def test_clone_native_options_unchanged(self):
        config = self.inputs()
        g.verify_recording(self.output, config)
        options = g.config_options(ET.parse(self.output / g.OVERRIDE_FILE).getroot())
        self.assertEqual(options["time-to-teleport"], "120")
        self.assertEqual(options["device.rerouting.probability"], "1")
        self.assertTrue(Path(options["net-file"]).is_absolute())

    def test_changed_dynamics_rejected_even_if_artifact_rehashed(self):
        config = self.inputs()
        path = self.output / g.OVERRIDE_FILE
        path.write_text(path.read_text().replace('value="120"', 'value="0"'))
        config["input_artifacts"][g.OVERRIDE_FILE] = {"sha256": g.digest(path), "bytes": path.stat().st_size}
        with self.assertRaises(ValueError):
            g.verify_recording(self.output, config)

    def test_changed_private_ids_rejected(self):
        config = self.inputs()
        (self.output / g.ID_FILE).write_text("unexpected\n")
        with self.assertRaises(ValueError):
            g.verify_recording(self.output, config)

    def test_command_only_reduces_fcd(self):
        command = g.options(g.build_command(self.source, self.root / "sumo", self.output, 350101))
        self.assertEqual(command["--device.fcd.probability"], "0")
        self.assertEqual(command["--person-device.fcd.probability"], "0")
        self.assertEqual(command["--fcd-output.attributes"], "x,y,type,lane")
        self.assertEqual(command["--end"], "50400")
        self.assertNotIn("--device.fcd.explicit", command)
        self.assertNotIn("--route-files", command)

    def test_source_device_override_rejected(self):
        path = self.source / "scenario/in/add/basic.vType.xml"
        path.write_text('<additional><vType id="bus"><param key="has.fcd.device" value="true"/></vType></additional>')
        with self.assertRaises(ValueError):
            g.plan_recording(self.source, self.splits, 350101, "train")

    def test_existing_attempt_refused_before_any_mutation(self):
        args = argparse.Namespace(source=self.source, sumo=self.root / "sumo", output=self.output,
                                  budget_root=self.output.parent, cnr_splits=self.splits, seed=350101, split="train")
        (self.output / "evidence").write_text("preserve")
        with self.assertRaises(FileExistsError):
            g.run(args)
        self.assertEqual((self.output / "evidence").read_text(), "preserve")

    def test_disk_limits_still_fail_closed(self):
        self.assertEqual(g.MAX_NEW_BYTES, 15 * 1024**3)
        for state in ({"free_bytes": g.MIN_FREE_BYTES-1, "budget_root_bytes": 0},
                      {"free_bytes": g.MIN_FREE_BYTES, "budget_root_bytes": g.MAX_NEW_BYTES+1}):
            with self.assertRaises(RuntimeError):
                g.enforce_storage(state)

    def test_parallel_tempfile_rename_race_is_tolerated(self):
        transient = self.output / "progress.tmp.json"
        transient.write_text("temporary")
        original, calls = Path.stat, [0]
        def disappearing(path, *args, **kwargs):
            if path == transient and not kwargs.get("follow_symlinks") is False:
                calls[0] += 1
                if calls[0] >= 2:
                    raise FileNotFoundError("peer renamed its atomic progress file")
            return original(path, *args, **kwargs)
        with patch.object(Path, "stat", disappearing):
            self.assertEqual(g.storage_state(self.output)["budget_root_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
