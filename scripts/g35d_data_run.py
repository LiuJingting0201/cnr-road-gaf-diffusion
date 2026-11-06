"""Bounded recovery: two new train simulations, preserved completed validation."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from g33_prepare import digest, save
from g35c_runtime import ROOT, environment, guard, interrupted, policy
from g35_train_run import stop_group, verify_completion

NAMES = ('g35d_data_run.py', 'g35d_simulate.py', 'g35d_extract.py',
         'g35c_runtime.py', 'g35_data_run.py', 'g35b_data_run.py',
         'g35_train_run.py', 'g35_merge.py', 'g35b_simulate.py',
         'g35b_extract.py', 'g35c_extract.py', 'g31_legs.py', 'g33_prepare.py',
         'g30_prepare.py', 'g30_extract.py', 'g30_demand_audit.py', 'g29_prepare.py')
EPISODES = (350101, 350102, 350201)
VAL_SOURCE = Path('../g35b-job/g35-data/native/val/350201')


def verify_simulation(folder, seed, cancel):
    marker = folder / 'simulation-complete.json'
    before = digest(marker)
    receipt = json.loads(marker.read_text())
    split = 'val' if seed == 350201 else 'train'
    if (receipt.get('complete') is not True or receipt.get('returncode') != 0
            or receipt.get('seed') != seed or receipt.get('split') != split
            or (folder / 'simulation-failed.json').exists()
            or receipt.get('config_sha256') != digest(folder / 'config.json')):
        raise ValueError('Unclean simulation completion: ' + str(seed))
    files = receipt.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Simulation completion has no output inventory')
    for name, spec in files.items():
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Artifact escapes its simulation folder')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                if cancel.is_set():
                    raise InterruptedError('Queue cancelled simulation hash verification')
                h.update(chunk)
        if path.stat().st_size != spec['bytes'] or h.hexdigest() != spec['sha256']:
            raise ValueError('Simulation artifact changed: ' + name)
    if digest(marker) != before:
        raise ValueError('Simulation receipt changed during verification')
    return before


def run():
    ROOT.mkdir(exist_ok=True)
    started, active, futures, streams = time.monotonic(), {}, {}, {}
    ready, extracted, receipts = set(), set(), {}
    cancel = threading.Event()
    with (ROOT/'data.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((ROOT/n).exists() for n in ('data-started.json', 'data-complete.json', 'data-failed.json')):
            raise FileExistsError('Existing attempt protected; do not restart')
        policy()
        simulations = {seed: (VAL_SOURCE.resolve() if seed == 350201 else ROOT/'native/train'/str(seed)) for seed in EPISODES}
        if any(simulations[seed].exists() for seed in (350101, 350102)) or (ROOT/'extracted').exists() or (ROOT/'direct-legs').exists():
            raise FileExistsError('Preserve existing new-stage outputs')
        if not (simulations[350201]/'simulation-complete.json').exists() or (simulations[350201]/'simulation-failed.json').exists():
            raise ValueError('Clean completed original validation simulation required')
        hashes = {n: digest(Path('scripts')/n) for n in NAMES}
        save(ROOT/'data-started.json', {'pid': os.getpid(), 'started_unix': time.time(), 'code_sha256': hashes,
             'status': 'starting_recovery', 'new_train_seeds': [350101, 350102],
             'preserved_validation': str(simulations[350201]), 'extraction_concurrency': 1})
        pool = ThreadPoolExecutor(max_workers=2)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupted)
        def launch(name, args, folder, kind, seed=None):
            guard()
            stream = (ROOT/(name+'.log')).open('x'); streams[name] = stream
            proc = subprocess.Popen([sys.executable, '-u', *args], stdout=stream,
                                    stderr=subprocess.STDOUT, env=environment(), start_new_session=True)
            active[name] = (proc, folder, kind, seed)
            print(json.dumps({'started': name, 'pid': proc.pid}), flush=True)
        try:
            guard()
            futures[350201] = pool.submit(verify_simulation, simulations[350201], 350201, cancel)
            for seed in (350101, 350102):
                launch('simulate-'+str(seed), ['scripts/g35d_simulate.py', '--source', '../g30-job/g30-data/source',
                    '--sumo', 'g10-env/sumo/bin/sumo', '--output', str(simulations[seed]), '--seed', str(seed),
                    '--split', 'train', '--budget-root', str(ROOT), '--cnr-splits', '../data/processed/most64/dataset.json'],
                    simulations[seed], 'simulate', seed)
            while True:
                state = guard()
                if hashes != {n: digest(Path('scripts')/n) for n in NAMES}:
                    raise ValueError('Frozen recovery code changed')
                if time.monotonic()-started > 12*3600:
                    raise TimeoutError('Finite recovery queue exceeded twelve hours')
                for name, (proc, folder, kind, seed) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    if code != 0:
                        raise RuntimeError(f'{name} exited {code}; preserve all prior and partial outputs')
                    del active[name]; streams.pop(name).close()
                    if kind == 'simulate':
                        futures[seed] = pool.submit(verify_simulation, folder, seed, cancel)
                    else:
                        receipt = verify_completion(folder)
                        receipts[name] = digest(folder/'complete.json')
                        if kind == 'merge':
                            save(ROOT/'data-complete.json', {'complete': True, 'seconds': time.monotonic()-started,
                                'merged_receipt_sha256': receipts[name], 'training_ready': receipt['training_ready'],
                                'groups': receipt['groups'], 'completed_stages': sorted(receipts),
                                'code_sha256': hashes, 'stage_receipts': receipts, 'storage': guard()})
                            return
                        extracted.add(seed)
                    print(json.dumps({'completed': name, 'seconds': time.monotonic()-started}), flush=True)
                for seed, future in list(futures.items()):
                    if future.done():
                        receipts['simulate-'+str(seed)] = future.result()
                        ready.add(seed); del futures[seed]
                if not any(item[2] in ('extract', 'merge') for item in active.values()):
                    available = ready-extracted
                    if available:
                        seed = min(available)
                        folder = ROOT/'extracted'/str(seed)
                        launch('extract-'+str(seed), ['scripts/g35d_extract.py', '--source', '../g30-job/g30-data/source',
                            '--simulation', str(simulations[seed]), '--cnr-splits', '../data/processed/most64/dataset.json',
                            '--output', str(folder)], folder, 'extract', seed)
                    elif extracted == set(EPISODES):
                        folder = ROOT/'direct-legs'
                        launch('merge', ['scripts/g35_merge.py', '--inputs',
                            *[str(ROOT/'extracted'/str(seed)) for seed in EPISODES], '--output', str(folder)], folder, 'merge')
                save(ROOT/'data-progress.json', {'status': 'running', 'seconds': time.monotonic()-started,
                    'active': {n: p.pid for n, (p, _, _, _) in active.items()}, 'validating_simulations': sorted(futures),
                    'ready_simulations': sorted(ready), 'extracted_seeds': sorted(extracted), 'storage': state})
                time.sleep(10)
        except BaseException as error:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            cancel.set()
            for proc, _, _, _ in active.values():
                stop_group(proc)
            save(ROOT/'data-failed.json', {'complete': False, 'error': repr(error),
                'seconds': time.monotonic()-started, 'completed_stages': sorted(receipts),
                'extracted_seeds': sorted(extracted), 'owned_process_groups': [p.pid for p, _, _, _ in active.values()]})
            raise
        finally:
            cancel.set(); pool.shutdown(wait=True)
            for stream in streams.values():
                stream.close()


if __name__ == '__main__':
    run()
