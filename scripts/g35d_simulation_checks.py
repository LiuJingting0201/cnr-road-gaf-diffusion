"""Focused capacity-only contract checks; never launches SUMO."""
import argparse
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import g35b_simulate as old
import g35d_simulate as new


class Checks(unittest.TestCase):
    def test_new_schema_and_locked_capacity(self):
        self.assertEqual(new.SCHEMA, 'g35-native-full-day-v3-capacity')
        self.assertEqual(new.MAX_NEW_BYTES, 40*1024**3)
        self.assertEqual(new.MIN_FREE_BYTES, 8*1024**3)
        self.assertEqual(new.EXECUTION_AMENDMENT,
                         'rerun interrupted fixed-seed train scene; retain completed val')

    def test_scientific_command_is_identical_for_both_train_seeds(self):
        for seed in (350101, 350102):
            args = (Path('/fixture/source'), Path('/fixture/sumo'), Path('/fixture/output'), seed)
            self.assertEqual(new.build_command(*args), old.build_command(*args))
        self.assertEqual((new.BEGIN, new.END, new.DT), (old.BEGIN, old.END, old.DT))
        self.assertEqual(new.NEW_ATTRIBUTES, old.NEW_ATTRIBUTES)
        self.assertEqual(new.FAMILIES, old.FAMILIES)

    def test_storage_counts_combined_roots_through_runtime_policy(self):
        calls = []
        result = {'budget_root_bytes': 17*1024**3, 'free_bytes': 12*1024**3,
                  'roots': {'old': 12*1024**3, 'new': 5*1024**3}}
        module = types.ModuleType('g35c_runtime')
        def combined(root):
            calls.append(root)
            return result
        module.storage_state = combined
        with patch.dict(sys.modules, {'g35c_runtime': module}):
            self.assertIs(new.storage_state(Path('g35-data')), result)
        self.assertEqual(calls, [Path('g35-data')])
        new.enforce_storage(result)
        for changed in (dict(result, budget_root_bytes=40*1024**3+1),
                        dict(result, free_bytes=8*1024**3-1)):
            with self.assertRaises(RuntimeError):
                new.enforce_storage(changed)

    def test_completed_validation_scene_cannot_be_rerun(self):
        with self.assertRaisesRegex(ValueError, 'completed val is preserved'):
            new.run(argparse.Namespace(seed=350201, split='val'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
