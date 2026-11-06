"""Synthetic G35 assignment, native representation and event extraction checks."""
from argparse import Namespace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import g35_extract as extraction
import g31_legs as legs
import g33_prepare as prepared


def metadata(ident="synthetic", episode=350101, label=2):
    return {"leg_id": f"{episode}:vehicle:{ident}:leg0", "parent_base_id": ident,
            "partition": "train", "episode": episode, "ordinal": 0, "label": label,
            "event_start": 0., "event_end": 2500., "line": "", "kind": "vehicle_activity_leg",
            "source_route_hash": "abc", "source_parent_route_hash": "abc", "realized_route_hash": "abc"}


class ExtractionChecks(unittest.TestCase):
    def test_one_preassigned_realization_and_no_fallback(self):
        selections = {seed: set() for seed in extraction.TRAIN_SEEDS}
        for i in range(1000):
            parent = f"synthetic-{i}"
            admitted = [seed for seed in extraction.TRAIN_SEEDS
                        if extraction.assignment_reason(parent, "train", seed) is None]
            self.assertEqual(len(admitted), 1)
            selections[admitted[0]].add(parent)
            other = next(seed for seed in extraction.TRAIN_SEEDS if seed != admitted[0])
            # Absence or failure of the designated realization is not an input
            # and can never cause admission from the other simulation.
            self.assertEqual(extraction.assignment_reason(parent, "train", other),
                             "other_preassigned_train_realization")
        self.assertFalse(selections[350101] & selections[350102])
        self.assertEqual(len(selections[350101] | selections[350102]), 1000)
        self.assertTrue(all(selections.values()))

    def test_episode_split_and_reserved_test(self):
        for seed in extraction.SEED_SPLIT:
            self.assertEqual(extraction.assignment_reason("old-test", "test", seed), "original_test_parent_excluded")
        for seed in extraction.TRAIN_SEEDS:
            self.assertEqual(extraction.assignment_reason("val-id", "val", seed), "other_parent_partition")
        self.assertEqual(extraction.assignment_reason("train-id", "train", extraction.VAL_SEED), "other_parent_partition")
        self.assertIsNone(extraction.assignment_reason("val-id", "val", extraction.VAL_SEED))
        with self.assertRaises(ValueError):
            extraction.assignment_reason("id", "train", 300001)

    def test_full_day_config_cannot_be_old_pilot(self):
        config = {"schema": "g35-native-full-day-v1", "seed": 350101, "split": "train",
                  "begin": 14400, "end": 50400, "native_step_s": .25}
        extraction.validate_config(config)
        for change in ({"end": 19516}, {"split": "val"}, {"native_step_s": 1}, {"seed": 300001}):
            with self.assertRaises(ValueError):
                extraction.validate_config({**config, **change})

    def test_native_long_leg_and_stops_are_not_windows(self):
        time = np.arange(10001, dtype=float) * .25
        x = np.maximum(time - 10., 0.)
        item = prepared.encode(np.c_[14400 + time, x, np.zeros_like(time)])
        item.update(xy=np.c_[x, np.zeros_like(time)], native_speed=np.r_[0., np.diff(x)/.25])
        arrays = extraction.arrays_from_rows([metadata()], [item])
        self.assertEqual(arrays["offset"].tolist(), [0, 10001])
        self.assertEqual(arrays["duration"].tolist(), [2500.])
        self.assertEqual(len(arrays["ids"]), 1)
        np.testing.assert_array_equal(arrays["time"], time)
        np.testing.assert_array_equal(arrays["progress"][:41], np.zeros(41))
        restored = prepared.exact_decode(item["path_xy"], item["path_arc"], item["progress"])
        np.testing.assert_allclose(restored[:, 0], x, atol=1e-7, rtol=0)

    def test_constant_leg_preserved_and_duplicate_identity_rejected(self):
        item = prepared.encode(np.c_[np.arange(5) * .25, np.ones(5), np.ones(5)])
        item.update(xy=np.ones((5, 2)), native_speed=np.zeros(5))
        arrays = extraction.arrays_from_rows([metadata()], [item])
        self.assertEqual(arrays["length"].tolist(), [0.])
        self.assertEqual(arrays["offset"].tolist(), [0, 5])
        with self.assertRaises(ValueError):
            extraction.arrays_from_rows([metadata(), metadata()], [item, item])

    def test_empty_schema_is_explicit(self):
        arrays = extraction.arrays_from_rows([], [])
        self.assertEqual(arrays["paths"].shape, (0, 129, 2))
        self.assertEqual(arrays["path_xy"].shape, (0, 2))
        self.assertEqual(arrays["offset"].tolist(), [0])

    def test_streaming_xml_to_dataset_keeps_parent_split_and_constant_leg(self):
        # Only the SUMO completion verification is mocked: this fixture tests
        # the real XML/event/coverage/encoding path, without simulating or
        # falsely certifying a short XML fixture as a full-day SUMO run.
        with tempfile.TemporaryDirectory(prefix="g35-synthetic-") as temporary:
            root = Path(temporary)
            source, simulation, output = root / "source", root / "simulation", root / "extracted"
            (source / "scenario").mkdir(parents=True)
            simulation.mkdir()
            train_ids = [f"synthetic-{i}" for i in range(100) if extraction.assigned_train_seed(f"synthetic-{i}") == 350101][:2]
            other = next(f"other-{i}" for i in range(100) if extraction.assigned_train_seed(f"other-{i}") == 350102)
            identifiers = [*train_ids, other, "dev-parent", "frozen-test-parent"]
            split_by_id = {identifier: "train" for identifier in identifiers}
            split_by_id.update({"dev-parent": "val", "frozen-test-parent": "test"})
            (root / "splits.json").write_text(json.dumps({"split_by_id": split_by_id}))
            (source / "scenario/most.sumocfg").write_text(
                '<configuration><input><route-files value="routes.xml"/><additional-files value="types.xml"/></input></configuration>')
            (source / "scenario/types.xml").write_text('<additional><vType id="passenger1" vClass="passenger"/></additional>')
            route_xml = '<routes>' + ''.join(f'<vehicle id="{identifier}"><route edges="a b"/></vehicle>' for identifier in identifiers) + '</routes>'
            (source / "scenario/routes.xml").write_text(route_xml)
            source_receipt = {"complete": True, "files": {name: {"sha256": legs.digest(source / name)}
                for name in ("scenario/routes.xml", "scenario/types.xml")}}
            (source / "complete.json").write_text(json.dumps(source_receipt))
            (simulation / "tripinfo.xml").write_text('<tripinfos>' + ''.join(
                f'<tripinfo id="{identifier}" vType="passenger1" depart="14400" arrival="14401"/>'
                for identifier in identifiers) + '</tripinfos>')
            with gzip.open(simulation / "vehroute.xml.gz", "wt") as stream:
                stream.write('<routes>' + ''.join(f'<vehicle id="{identifier}"><route edges="a b" exitTimes="14400.5 14401"/></vehicle>'
                             for identifier in identifiers) + '</routes>')
            (simulation / "stops.xml").write_text('<stops/>')
            with gzip.open(simulation / "fcd.xml.gz", "wt") as stream:
                stream.write('<fcd-export>')
                for j in range(5):
                    stream.write(f'<timestep time="{14400 + .25 * j}">')
                    for identifier in identifiers:
                        x = 0 if identifier == train_ids[1] else j
                        stream.write(f'<vehicle id="{identifier}" type="passenger1" x="{x}" y="0" speed="4" pos="{x}" lane="a_0"/>')
                    stream.write('</timestep>')
                stream.write('</fcd-export>')
            config = {"schema": "g35-native-full-day-v1", "seed": 350101, "split": "train",
                      "begin": 14400, "end": 50400, "native_step_s": .25,
                      "source_manifest_sha256": legs.digest(source / "complete.json")}
            (simulation / "config.json").write_text(json.dumps(config))
            receipt = {"config_sha256": legs.digest(simulation / "config.json")}
            (simulation / "simulation-complete.json").write_text(json.dumps(receipt))
            with patch('g35_simulate.verify_receipt', return_value=(config, receipt)), patch('g35_simulate.verify_source', return_value=source_receipt):
                result = extraction.extract(Namespace(source=source, simulation=simulation, cnr_splits=root / "splits.json", output=output))
            self.assertEqual(result["accepted_legs"], 2)
            self.assertEqual(result["groups"]["train"]["parents"], 2)
            self.assertEqual(result["groups"]["val"]["legs"], 0)
            self.assertEqual(result["unsupported_by_current_motion_trainer"]["zero_length_legs"], 1)
            self.assertFalse(result["train_ready"])
            self.assertEqual(result["partition_exclusions"]["original_test_templates_excluded_before_leg_extraction"], 1)
            self.assertEqual(result["assignment_exclusions"]["other_preassigned_train_realization"], 1)
            with np.load(output / "dataset.npz", allow_pickle=False) as arrays:
                self.assertEqual(set(arrays["parent_ids"].tolist()), set(train_ids))
                self.assertEqual(set(arrays["labels"].tolist()), {2})
                self.assertEqual(arrays["length"].tolist(), [4., 0.])
            with gzip.open(output / "leg-metadata.json.gz", "rt") as stream:
                metadata = json.load(stream)
            self.assertEqual(len(metadata["unsupported_zero_length_leg_ids"]), 1)
            self.assertEqual(legs.digest(output / "dataset.npz"), result["artifacts"]["dataset.npz"]["sha256"])


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtractionChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
