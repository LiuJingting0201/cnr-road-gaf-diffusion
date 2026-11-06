"""Seven synthetic G31 interface checks; no real data, SUMO run or training.

Run directly with ``python scripts/g31_interface_checks.py``. Fixtures and
receipts live in automatically removed temporary directories. The full-clock
fixture contains synthetic constant positions, not a real pilot trajectory.
"""
import gzip
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import g31_compare as compare
import g31_replay as replay


def write_xml(path, text):
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as stream:
            stream.write(text)
    else:
        path.write_text(text)


def pilot(directory):
    command = [
        "/synthetic/sumo", "-c", "/synthetic/most.sumocfg",
        "--seed", "300001", "--begin", "14400", "--end", "19516",
        "--step-length", ".25", "--device.fcd.period", ".25",
        "--person-device.fcd.period", ".25", "--device.fcd.probability", "1",
        "--person-device.fcd.probability", "1",
        "--fcd-output.attributes", replay.OLD_ATTRIBUTES,
        "--tripinfo-output.write-unfinished", "true",
        "--vehroute-output.write-unfinished", "true",
        "--output-prefix", "", "--no-step-log", "true",
    ]
    for flag, name in replay.OUTPUT_FILES.items():
        command.extend((flag, str(directory / name)))
    return {
        "seed": 300001, "split": "pilot", "begin": 14400, "end": 19516,
        "native_step_s": .25, "source_commit": replay.COMMIT,
        "command": command, "source_manifest_sha256": "synthetic",
        "version": "Eclipse SUMO sumo 1.28.0",
    }


def receipt(directory, config):
    replay.save(directory / "config.json", config)
    record = {
        "complete": True,
        "config_sha256": replay.digest(directory / "config.json"),
        "files": {
            name: {"sha256": replay.digest(directory / name),
                   "bytes": (directory / name).stat().st_size}
            for name in replay.RECEIPT_FILES
        },
    }
    replay.save(directory / "simulation-complete.json", record)


class G31InterfaceChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cnr-g31-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def pair(self, left, right):
        a, b = self.root / "a.xml.gz", self.root / "b.xml.gz"
        write_xml(a, left)
        write_xml(b, right)
        return compare.compare_fcd(a, b)

    def test_reordering_and_output_addition(self):
        vehicle = (
            '<vehicle id="SECRET_CAR" x="1" y="2" type="car" speed="3" lane="SECRET_LANE">'
            '<person id="SECRET_RIDER" x="1" y="2" type="person" speed="3" vehicle="SECRET_CAR"/>'
            '</vehicle>'
        )
        person = '<person id="SECRET_PED" x="7" y="8" type="person" speed="1"/>'
        prefix = '<fcd-export><timestep time="14400">'
        suffix = '</timestep><timestep time="14400.25"/></fcd-export>'
        left = prefix + vehicle + person + suffix
        right = prefix + person + vehicle.replace('speed="3"', 'speed="3" pos="4"') + suffix
        result = self.pair(left, right)
        self.assertTrue(result["equal"])
        self.assertEqual(result["counts"]["matched_records"], 3)
        self.assertNotIn("SECRET_", json.dumps(result))

    def test_parent_and_field_change(self):
        left = (
            '<fcd-export><timestep time="0"><vehicle id="v" x="1" y="2">'
            '<person id="p" x="1" y="2"/></vehicle></timestep></fcd-export>'
        )
        right = (
            '<fcd-export><timestep time="0"><vehicle id="v" x="1" y="3"/>'
            '<person id="p" x="1" y="2"/></timestep></fcd-export>'
        )
        result = self.pair(left, right)
        self.assertFalse(result["equal"])
        self.assertEqual(result["counts"]["changed_records"], 2)
        self.assertEqual(result["changed_field_counts"], {"XML_parent_entity": 1, "y": 1})

    def test_missing_records_and_empty_timesteps(self):
        left = (
            '<fcd-export><timestep time="0"><vehicle id="v" x="1" y="2"/>'
            '</timestep><timestep time="1"/></fcd-export>'
        )
        right = '<fcd-export><timestep time="0"/><timestep time="2"/></fcd-export>'
        result = self.pair(left, right)
        self.assertFalse(result["equal"])
        self.assertEqual(result["counts"]["baseline_only_records"], 1)
        self.assertEqual(result["counts"]["baseline_only_timesteps"], 1)
        self.assertEqual(result["counts"]["replay_only_timesteps"], 1)

    def test_duplicate_and_bad_clock_rejected(self):
        fixtures = [
            '<fcd-export><timestep time="0"><vehicle id="v" x="1" y="2"/>'
            '<vehicle id="v" x="1" y="2"/></timestep></fcd-export>',
            '<fcd-export><timestep time="1"/><timestep time="0"/></fcd-export>',
        ]
        for xml in fixtures:
            with self.assertRaises(ValueError):
                self.pair(xml, xml)

    def test_command_whitelist(self):
        config = pilot(self.root / "old")
        command = replay.build_command(
            config, self.root / "new", self.root / "source", self.root / "sumo")
        old, new = replay.options(config["command"]), replay.options(command)
        allowed = set(replay.OUTPUT_FILES) | set(replay.ENHANCEMENTS) | {"-c", "--fcd-output.attributes"}
        self.assertEqual(
            {key: value for key, value in old.items() if key not in allowed},
            {key: value for key, value in new.items() if key not in allowed},
        )
        self.assertEqual(new["--fcd-output.attributes"], replay.NEW_ATTRIBUTES)
        config["command"][config["command"].index("--seed") + 1] = "300002"
        with self.assertRaises(ValueError):
            replay.build_command(config, self.root / "new", self.root / "source", self.root / "sumo")

    def test_schema_audit(self):
        path = self.root / "routes.xml"
        path.write_text(
            '<routes><vehicle id="SECRET"><route edges="a b a" exitTimes="1 2 -1"/>'
            '<stop started="2" ended="-1" priorEdges="a b" priorEdgesLength="123" parking="1"/>'
            '</vehicle></routes>'
        )
        result = compare.schema_audit(path)
        self.assertEqual(result["structure"]["nodes_with_repeated_edge"], 1)
        self.assertEqual(result["structure"]["exitTimes_negative_values"], 1)
        self.assertEqual(result["parking_boolean_counts"], {"true": 1})
        self.assertNotIn("SECRET", json.dumps(result))

    def test_complete_pipeline_and_no_overwrite(self):
        baseline, enhanced = self.root / "old", self.root / "new"
        baseline.mkdir()
        enhanced.mkdir()
        config = pilot(baseline)
        for destination, is_enhanced in ((baseline, False), (enhanced, True)):
            with gzip.open(destination / "fcd.xml.gz", "wt") as stream:
                stream.write("<fcd-export>")
                for index in range(20464):
                    pos = ' pos="0"' if is_enhanced else ""
                    stream.write(
                        f'<timestep time="{14400 + index * .25:.2f}">'
                        f'<vehicle id="v" x="1" y="2" type="car" speed="0"{pos}/></timestep>'
                    )
                stream.write("</fcd-export>")
            route = (
                '<route edges="a b" exitTimes="1 2"/>'
                '<stop started="1" ended="2" priorEdges="a" priorEdgesLength="3"/>'
                if is_enhanced else '<route edges="a b"/>'
            )
            write_xml(destination / "vehroute.xml.gz", '<routes><vehicle id="v">' + route + '</vehicle></routes>')
            write_xml(destination / "tripinfo.xml", '<tripinfos><tripinfo id="v" depart="14400" arrival="-1"/></tripinfos>')
            write_xml(destination / "stops.xml", '<stops><stopinfo id="v" started="14401" ended="-1" parking="1"/></stops>')
            write_xml(destination / "summary.xml", '<summary><step time="19515.75"/></summary>')
        receipt(baseline, config)
        enhanced_config = dict(
            config,
            command=replay.build_command(config, enhanced, self.root / "source", self.root / "sumo"),
            source=str(self.root / "source"),
            baseline_config_sha256=replay.digest(baseline / "config.json"),
            baseline_receipt_sha256=replay.digest(baseline / "simulation-complete.json"),
            baseline_command_sha256=replay.json_hash(config["command"]),
        )
        receipt(enhanced, enhanced_config)
        args = Namespace(baseline=baseline, replay=enhanced, output=self.root / "report")
        compare.run(args)
        self.assertTrue(json.loads((args.output / "comparison-complete.json").read_text())["complete"])
        with self.assertRaises(FileExistsError):
            compare.run(args)
        (enhanced / "stops.xml").write_text("<stops/>")
        with self.assertRaises(ValueError):
            compare.run(Namespace(baseline=baseline, replay=enhanced, output=self.root / "bad"))
        self.assertTrue((self.root / "bad/comparison-failed.json").exists())
        self.assertFalse((self.root / "bad/comparison-complete.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
