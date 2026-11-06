"""Small synthetic checks for G35 provenance, guard and immutable-attempt rules."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import g35_simulate as g


class OwnedChild:
    pid = 123456789
    returncode = None
    terminated = False
    killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fixture-owned-sumo", timeout)
        return self.returncode


class SimulationChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def config(self, output=None):
        output = output or self.root / "attempt"
        source, binary = self.root / "source", self.root / "sumo"
        return {"schema": g.SCHEMA, "seed": 350101, "split": "train",
            "source": str(source), "binary": str(binary), "output": str(output),
            "begin": g.BEGIN, "end": g.END, "native_step_s": g.DT,
            "source_commit": g.COMMIT, "source_manifest_sha256": g.SOURCE_MANIFEST_SHA,
            "version": "Eclipse SUMO sumo Version 1.28.0", "binary_sha256": "0" * 64,
            "script_sha256": "1" * 64, "resource_limits": g.resource_limits(),
            "command": g.build_command(source, binary, output, 350101)}

    def completed(self):
        folder = self.root / "attempt"
        folder.mkdir()
        g.save(folder / "config.json", self.config(folder))
        for name in g.RECEIPT_FILES:
            (folder / name).write_text("fixture\n")
        (folder / "summary.xml").write_text('<summary><step time="50399.75" running="0"/></summary>')
        (folder / "console.log").write_text("Warning: Teleporting vehicle\nCollision warning:\n")
        receipt = {"complete": True, "returncode": 0, "seed": 350101, "split": "train",
            "config_sha256": g.digest(folder / "config.json"),
            "files": {name: {"sha256": g.digest(folder / name), "bytes": (folder / name).stat().st_size}
                      for name in g.RECEIPT_FILES},
            "last_summary": g.final_summary(folder / "summary.xml"),
            "log_keyword_line_counts": g.warning_counts(folder / "console.log")}
        g.save(folder / "simulation-complete.json", receipt)
        return folder

    def test_frozen_command_retains_dynamics(self):
        config = self.config()
        g.validate_config(config)
        options = g.options(config["command"])
        self.assertEqual(options["--person-device.fcd.probability"], "0")
        self.assertEqual(options["--device.fcd.probability"], "1")
        self.assertEqual(options["--fcd-output.attributes"], g.NEW_ATTRIBUTES)
        self.assertTrue(all(options[k] == v for k, v in g.ENHANCEMENTS.items()))
        self.assertNotIn("--route-files", options)
        self.assertNotIn("--collision.action", options)

    def test_unknown_seed_and_crossed_split_rejected(self):
        for seed, split in ((350999, "train"), (350101, "val"), (350201, "train")):
            value = self.config()
            value.update(seed=seed, split=split)
            with self.assertRaises(ValueError):
                g.validate_config(value)

    def test_changed_clock_and_command_rejected(self):
        for key, value in (("native_step_s", 1), ("end", 19516), ("version", "SUMO 1.27.0")):
            config = self.config()
            config[key] = value
            with self.assertRaises(ValueError):
                g.validate_config(config)
        config = self.config()
        config["command"] += ["--scale", "2"]
        with self.assertRaises(ValueError):
            g.validate_config(config)

    def test_receipt_happy_path(self):
        config, receipt = g.verify_receipt(self.completed())
        self.assertEqual(config["seed"], 350101)
        self.assertEqual(receipt["log_keyword_line_counts"]["warning:"], 2)

    def test_tampered_output_rejected(self):
        folder = self.completed()
        (folder / "fcd.xml.gz").write_text("changed\n")
        with self.assertRaises(ValueError):
            g.verify_receipt(folder)

    def test_tampered_config_rejected(self):
        folder = self.completed()
        (folder / "config.json").write_text("{}")
        with self.assertRaises(ValueError):
            g.verify_receipt(folder)

    def test_failed_marker_rejects_completion(self):
        folder = self.completed()
        g.save(folder / "simulation-failed.json", {"complete": False})
        with self.assertRaises(ValueError):
            g.verify_receipt(folder)

    def test_incomplete_end_rejected(self):
        for time in ("19515.75", "50400", "nan"):
            path = self.root / "summary.xml"
            path.write_text(f'<summary><step time="{time}"/></summary>')
            with self.assertRaises(ValueError):
                g.final_summary(path)

    def test_partial_xml_progress_is_not_completion(self):
        path = self.root / "summary.xml"
        path.write_text('<summary><step time="15000"/><step time="15000.25"')
        self.assertEqual(g.summary_tail(path), {"time": "15000"})
        with self.assertRaises(Exception):
            g.final_summary(path)

    def test_budget_counts_nested_failed_attempts(self):
        folder = self.root / "failed"
        folder.mkdir()
        (folder / "partial").write_bytes(b"12345678")
        (self.root / "another").write_bytes(b"12")
        self.assertEqual(g.storage_state(self.root)["budget_root_bytes"], 10)

    def test_symlink_budget_escape_rejected(self):
        (self.root / "escape").symlink_to("/tmp", target_is_directory=True)
        with self.assertRaises(ValueError):
            g.storage_state(self.root)

    def test_guard_exact_boundaries_and_failures(self):
        g.enforce_storage({"free_bytes": g.MIN_FREE_BYTES, "budget_root_bytes": g.MAX_NEW_BYTES})
        for state in ({"free_bytes": g.MIN_FREE_BYTES - 1, "budget_root_bytes": 0},
                      {"free_bytes": g.MIN_FREE_BYTES, "budget_root_bytes": g.MAX_NEW_BYTES + 1}):
            with self.assertRaises(RuntimeError):
                g.enforce_storage(state)

    def test_existing_attempt_never_overwritten(self):
        output = self.root / "g35-data" / "train" / "350101"
        output.mkdir(parents=True)
        marker = output / "evidence"
        marker.write_text("old-failure")
        args = argparse.Namespace(source=self.root / "source", sumo=self.root / "sumo",
            output=output, budget_root=self.root / "g35-data", seed=350101, split="train")
        with self.assertRaises(FileExistsError):
            g.run(args)
        self.assertEqual(marker.read_text(), "old-failure")
        self.assertEqual(list(output.iterdir()), [marker])

    def test_pinned_source_inventory_and_tamper(self):
        source = self.root / "source"
        files = {}
        for name in g.FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            files[name] = {"sha256": g.digest(path), "bytes": path.stat().st_size}
        g.save(source / "complete.json", {"complete": True, "commit": g.COMMIT, "files": files})
        with patch.object(g, "SOURCE_MANIFEST_SHA", g.digest(source / "complete.json")), \
             patch.object(g, "NETWORK_SHA", g.digest(source / "scenario/in/most.net.xml")), \
             patch.object(g, "EXPECTED_BYTES", {}):
            self.assertTrue(g.verify_source(source)["complete"])
            (source / "scenario/in/route/most.buses.flows.xml").write_text("changed")
            with self.assertRaises(ValueError):
                g.verify_source(source)

    def test_guard_terminates_only_owned_child_and_keeps_failure(self):
        source = self.root / "source"
        source.mkdir()
        (source / "complete.json").write_text("{}")
        binary = self.root / "sumo"
        binary.write_text("fixture-binary")
        budget = self.root / "g35-data"
        output = budget / "train" / "350101"
        args = argparse.Namespace(source=source, sumo=binary, output=output,
            budget_root=budget, seed=350101, split="train")
        child = OwnedChild()
        okay = {"free_bytes": g.MIN_FREE_BYTES, "budget_root_bytes": 0}
        full = {"free_bytes": g.MIN_FREE_BYTES - 1, "budget_root_bytes": 0}
        with ExitStack() as stack:
            stack.enter_context(patch.object(g, "SOURCE_MANIFEST_SHA", g.digest(source / "complete.json")))
            stack.enter_context(patch.object(g, "verify_source", return_value={}))
            stack.enter_context(patch.object(g, "storage_state", side_effect=[okay, full]))
            stack.enter_context(patch.object(g.subprocess, "check_output", side_effect=[
                "Eclipse SUMO sumo Version 1.28.0\n", " ".join(g.ENHANCEMENTS)]))
            stack.enter_context(patch.object(g.subprocess, "Popen", return_value=child))
            with self.assertRaisesRegex(RuntimeError, "disk guard"):
                g.run(args)
        self.assertTrue(child.terminated)
        self.assertFalse(child.killed)
        receipt = json.loads((output / "simulation-failed.json").read_text())
        self.assertEqual(receipt["returncode"], -15)
        self.assertIn("disk guard", receipt["error"])
        self.assertFalse((output / "simulation-complete.json").exists())
        self.assertEqual(json.loads((output / "simulation-progress.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
