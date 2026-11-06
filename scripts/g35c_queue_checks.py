"""Stdlib-only G35c orchestration checks; never launch SUMO or GPU work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value) + '\n')
    temporary.replace(path)


def verify_completion(folder):
    receipt = json.loads((folder / 'complete.json').read_text())
    if not receipt.get('complete') or (folder / 'failed.json').exists():
        raise ValueError('Unclean mock completion')
    for name, spec in receipt['artifacts'].items():
        if digest(folder / name) != spec['sha256']:
            raise ValueError('Mock artifact changed')
    return receipt


def stop_group(process):
    if process.poll() is None:
        os.killpg(process.pid, 15)
        process.wait(timeout=3)


def load_queue():
    # Only dependency helpers are substituted. Execute the actual queue source,
    # whose control flow, Popen, completion ordering and failure paths are tested.
    helpers = types.ModuleType('g33_prepare')
    helpers.digest, helpers.save = digest, save
    runtime = types.ModuleType('g35c_runtime')
    runtime.ROOT = Path('g35-data')
    runtime.environment = lambda: dict(os.environ)
    runtime.guard = lambda: {}
    runtime.interrupted = lambda *_: None
    runtime.policy = lambda: ({}, [])
    training = types.ModuleType('g35_train_run')
    training.stop_group, training.verify_completion = stop_group, verify_completion
    spec = importlib.util.spec_from_file_location('g35c_queue_under_test', Path(__file__).with_name('g35c_data_run.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'g33_prepare': helpers, 'g35c_runtime': runtime, 'g35_train_run': training}):
        spec.loader.exec_module(module)
    return module


CHILD = r'''
import argparse,hashlib,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
a,_=p.parse_known_args();root=Path(os.environ['G35C_CHECK_ROOT'])
kind='merge' if Path(__file__).name=='g35_merge.py' else 'extract'
name='merge' if kind=='merge' else a.output.name
def event(value):
 fd=os.open(root/'events.jsonl',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
 os.write(fd,(json.dumps({'name':name,'kind':kind,'event':value,'time':time.monotonic(),'pid':os.getpid()})+'\n').encode());os.close(fd)
event('started');a.output.mkdir(parents=True)
if kind=='merge':
 assert all((root/'extracted'/s/'complete.json').exists() for s in ('350101','350102','350201'))
 assert (root/'adopted/complete.json').exists()
time.sleep(.04)
if os.environ.get('G35C_CHECK_FAIL')==name:
 event('failed');raise SystemExit(7)
payload=a.output/'payload';payload.write_bytes(name.encode())
receipt={'complete':True,'training_ready':True,'groups':{'train':{},'val':{},'all':{}},'artifacts':{'payload':{'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}}}
(a.output/'complete.json').write_text(json.dumps(receipt));event('completed')
'''


class QueueChecks(unittest.TestCase):
    def exercise(self, mode='normal', duplicate=False):
        queue = load_queue()
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='g35c-queue-check-') as temporary:
            base = Path(temporary)
            scripts = base/'scripts'; scripts.mkdir()
            for name in ('g35c_extract.py', 'g35_merge.py'):
                (scripts/name).write_text(CHILD)
            root = base/'g35-data'; root.mkdir()
            seeds = ('350101', '350102', '350201')
            for seed in seeds:
                folder = root/'adopted'/seed; folder.mkdir(parents=True)
                save(folder/'complete.json', {'complete': True})
            if mode != 'late':
                save(root/'adopted/complete.json', {'complete': True})
            manifest = {'episodes': [{'seed': int(s), 'simulation_path': 'mock-'+s} for s in seeds]}
            late_waits, published = [0], [None]
            def guard():
                if mode == 'late' and (root/'data-progress.json').exists():
                    progress = json.loads((root/'data-progress.json').read_text())
                    if set(seeds) <= set(progress['completed_stages']) and progress['active'] is None:
                        late_waits[0] += 1
                        if late_waits[0] == 4:
                            published[0] = time.monotonic()
                            save(root/'adopted/complete.json', {'complete': True})
                    if late_waits[0] > 30:
                        raise AssertionError('Overall adoption marker did not unblock merge')
                return {'free_bytes': 10*1024**3, 'budget_root_bytes': 0}
            env = dict(os.environ, G35C_CHECK_ROOT=str(root))
            if mode == 'failure':
                env['G35C_CHECK_FAIL'] = '350102'
            timer = types.SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda _: time.sleep(.01))
            try:
                os.chdir(base)
                with patch.object(queue, 'ROOT', root), patch.object(queue, 'NAMES', ('g35c_extract.py', 'g35_merge.py')), \
                     patch.object(queue, 'guard', guard), patch.object(queue, 'policy', return_value=(manifest, [])), \
                     patch.object(queue, 'environment', return_value=env), patch.object(queue, 'time', timer), \
                     patch.object(queue.signal, 'signal', return_value=None):
                    if mode == 'failure':
                        with self.assertRaisesRegex(RuntimeError, '350102 exited 7'):
                            queue.run()
                    else:
                        queue.run()
                        if duplicate:
                            before = (root/'events.jsonl').read_bytes()
                            with self.assertRaises(FileExistsError):
                                queue.run()
                            self.assertEqual(before, (root/'events.jsonl').read_bytes())
                            self.assertFalse((root/'data-failed.json').exists())
            finally:
                os.chdir(previous)
            events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            result = {n: json.loads((root/n).read_text()) for n in ('data-complete.json', 'data-failed.json') if (root/n).exists()}
            return events, result, late_waits[0], published[0]

    def test_one_extraction_process_at_a_time(self):
        events, result, _, _ = self.exercise()
        active = set()
        for event in events:
            if event['kind'] != 'extract':
                continue
            if event['event'] == 'started':
                self.assertFalse(active)
                active.add(event['name'])
            elif event['event'] == 'completed':
                active.remove(event['name'])
        self.assertFalse(active)
        self.assertTrue(result['data-complete.json']['complete'])

    def test_merge_after_three_validated_extractions(self):
        events, result, _, _ = self.exercise()
        completed = {e['name']: i for i, e in enumerate(events) if e['kind'] == 'extract' and e['event'] == 'completed'}
        merge_index = next(i for i, e in enumerate(events) if e['kind'] == 'merge' and e['event'] == 'started')
        self.assertEqual(len(completed), 3)
        self.assertTrue(all(i < merge_index for i in completed.values()))
        self.assertEqual(result['data-complete.json']['completed_stages'], ['350101', '350102', '350201', 'merge'])

    def test_wait_for_late_overall_adoption_marker(self):
        events, result, waits, published = self.exercise('late')
        self.assertGreaterEqual(waits, 4)
        self.assertIsNotNone(published)
        self.assertTrue(any(e['kind'] == 'merge' and e['event'] == 'completed' for e in events))
        self.assertTrue(result['data-complete.json']['complete'])

    def test_failed_extraction_never_merges(self):
        events, result, _, _ = self.exercise('failure')
        self.assertNotIn('data-complete.json', result)
        self.assertFalse(result['data-failed.json']['complete'])
        self.assertFalse(any(e['kind'] == 'merge' for e in events))
        self.assertFalse(any(e['name'] == '350201' for e in events))

    def test_duplicate_launch_preserves_completed_attempt(self):
        _, result, _, _ = self.exercise(duplicate=True)
        self.assertTrue(result['data-complete.json']['complete'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QueueChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({'complete': True, 'component': 'G35c queue coordination smoke',
                      'checks': result.testsRun, 'real_SUMO_or_GPU_launched': False}))
