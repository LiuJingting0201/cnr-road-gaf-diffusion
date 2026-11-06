"""Finite post-handoff extraction/merge queue, using existing SUMO outputs."""
import fcntl
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

from g33_prepare import digest, save
from g35c_runtime import ROOT, environment, guard, interrupted, policy
from g35_train_run import stop_group, verify_completion

NAMES = ('g35c_data_run.py', 'g35c_runtime.py', 'g35c_adopt.py', 'g35c_extract.py',
         'g35_data_run.py', 'g35b_data_run.py', 'g35_train_run.py',
         'g35_merge.py', 'g35b_simulate.py', 'g35b_extract.py', 'g31_legs.py',
         'g33_prepare.py', 'g30_prepare.py', 'g30_extract.py', 'g30_demand_audit.py',
         'g29_prepare.py')


def run():
    ROOT.mkdir(exist_ok=True)
    started, process, active_name, stream = time.monotonic(), None, None, None
    done, receipts = [], {}
    with (ROOT / 'data.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((ROOT / name).exists() for name in ('data-started.json', 'data-complete.json', 'data-failed.json')):
            raise FileExistsError('Preserve the existing attempt')
        hashes = {name: digest(Path('scripts') / name) for name in NAMES}
        manifest, _ = policy()
        episodes = {str(row['seed']): row for row in manifest['episodes']}
        if set(episodes) != {'350101', '350102', '350201'}:
            raise ValueError('Unexpected finite episode list')
        save(ROOT / 'data-started.json', {'pid': os.getpid(), 'started_unix': time.time(),
             'status': 'waiting_for_adopted_simulations', 'code_sha256': hashes,
             'extraction_concurrency': 1, 'scope': 'Continue exact existing SUMO realizations; no rerun'})
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupted)
        try:
            while True:
                state = guard()
                if hashes != {name: digest(Path('scripts') / name) for name in NAMES}:
                    raise ValueError('Frozen continuation code changed')
                if (ROOT / 'adopted/failed.json').exists():
                    raise RuntimeError('SUMO handoff monitor failed; preserve all artifacts')
                if time.monotonic() - started > 12*3600:
                    raise TimeoutError('Finite data continuation exceeded twelve hours')
                if process is not None and process.poll() is not None:
                    if process.returncode != 0:
                        raise RuntimeError(f'{active_name} exited {process.returncode}')
                    folder = ROOT / ('direct-legs' if active_name == 'merge' else 'extracted/' + active_name)
                    receipt = verify_completion(folder)
                    receipts[active_name] = digest(folder / 'complete.json')
                    done.append(active_name)
                    print(json.dumps({'completed': active_name, 'seconds': time.monotonic()-started}), flush=True)
                    process, active_name = None, None
                    stream.close(); stream = None
                    if done[-1] == 'merge':
                        save(ROOT / 'data-complete.json', {'complete': True, 'seconds': time.monotonic()-started,
                             'completed_stages': done, 'merged_receipt_sha256': receipts['merge'],
                             'training_ready': receipt['training_ready'], 'groups': receipt['groups'],
                             'stage_receipts': receipts, 'code_sha256': hashes, 'storage': guard()})
                        return
                if process is None:
                    available = [seed for seed in episodes if seed not in done and
                                 (ROOT / 'adopted' / seed / 'complete.json').exists()]
                    command = None
                    if available:
                        active_name = sorted(available)[0]
                        row = episodes[active_name]
                        command = [sys.executable, '-u', 'scripts/g35c_extract.py',
                            '--adoption', str(ROOT/'adopted'/active_name),
                            '--source', '../g30-job/g30-data/source',
                            '--simulation', row['simulation_path'],
                            '--cnr-splits', '../data/processed/most64/dataset.json',
                            '--output', str(ROOT/'extracted'/active_name)]
                    elif set(episodes) <= set(done) and (ROOT / 'adopted/complete.json').exists():
                        active_name = 'merge'
                        command = [sys.executable, '-u', 'scripts/g35_merge.py', '--inputs',
                                   *[str(ROOT/'extracted'/seed) for seed in sorted(episodes)],
                                   '--output', str(ROOT/'direct-legs')]
                    if command:
                        stream = (ROOT / ('extract-'+active_name+'.log')).open('x')
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                                   env=environment(), start_new_session=True)
                        print(json.dumps({'started': active_name, 'pid': process.pid}), flush=True)
                save(ROOT / 'data-progress.json', {'status': 'running', 'seconds': time.monotonic()-started,
                     'active': active_name, 'pid': process.pid if process else None,
                     'completed_stages': done, 'storage': state})
                time.sleep(10)
        except BaseException as error:
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            if process is not None:
                stop_group(process)
            save(ROOT / 'data-failed.json', {'complete': False, 'error': repr(error),
                 'completed_stages': done, 'seconds': time.monotonic()-started})
            raise
        finally:
            if stream is not None:
                stream.close()


if __name__ == '__main__':
    run()
