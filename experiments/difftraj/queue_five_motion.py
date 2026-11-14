"""Run the fixed single-class speed comparisons and labelled multi-class comparisons."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/workspace/cnr-experiments/difftraj-five-40k-v1')
MODES = ('taxi', 'bus', 'bicycle', 'motorcycle', 'passenger')
SEEDS = (42, 43)
GIB = 1024**3
KEYS = ('density_js', 'od_js', 'length_js', 'hotspot_f1')


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def jobs():
    # Start with four independent single-class baselines. Start the longer shared
    # models early enough to overlap single-class fits on the other devices.
    order = [(m, 'A', s) for m in ('taxi', 'bus') for s in SEEDS]
    order += [('all', a, s) for a in ('C', 'D') for s in SEEDS]
    order += [(m, a, s) for m in MODES for a in ('A', 'B') for s in SEEDS]
    return list(dict.fromkeys(order))


def key(job):
    return '-'.join(map(str, job))


def parent(job):
    mode, arm, seed = job
    if arm == 'B':
        return mode, 'A', seed
    if arm == 'D':
        return mode, 'C', seed
    if arm == 'C':
        return 'taxi', 'A', seed
    return None


def logrun(args, directory, name, gpu=None):
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{ROOT}:{ROOT / 'vendor'}", OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with (directory / name).open('a', buffering=1) as log:
        print(stamp(), *map(str, args), file=log, flush=True)
        subprocess.run([sys.executable, *map(str, args)], check=True, env=env,
                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def prepare():
    logrun([ROOT / 'prepare_five_motion.py', '--data', DATA, '--out', ROOT / 'prepared'],
           ROOT, 'prepare.log')
    return 'ready'


def read_metrics(job):
    mode, arm, seed = job
    d = json.loads((ROOT / 'runs' / key(job) / 'eval' / 'metrics.json').read_text())
    expected_n = 3000 if mode == 'all' else 600
    if (d['requests'] != expected_n or d['arm'] != arm or int(d['seed']) != seed or
            not all(math.isfinite(d[k]) for k in KEYS)):
        raise ValueError(f'Invalid scores for {job}')
    return d


def fit(job, gpu):
    mode, arm, seed = job
    folder = ROOT / 'runs' / key(job)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETE').exists():
        read_metrics(job)
        return
    # Disk allowance includes other workers, resume checkpoints and evaluations.
    if shutil.disk_usage(ROOT).free < 8 * GIB:
        raise OSError('Less than 8 GiB free before starting a fit')
    batch, micro = (1024, 256) if mode == 'all' else (205, 205)
    atomic(folder / 'assignment.json', dict(mode=mode, arm=arm, seed=seed, gpu=gpu,
           epochs=50, batch=batch, microbatch=micro, updates=9800, started=stamp()))
    try:
        if not (folder / 'TRAIN_COMPLETE').exists():
            logrun([ROOT / 'five_class_train.py', '--data', DATA, '--speed-dir', ROOT / 'prepared',
                    '--out', folder, '--mode', mode, '--arm', arm, '--seed', str(seed),
                    '--epochs', '50', '--batch-size', str(batch), '--microbatch', str(micro)],
                   folder, 'train.log', gpu)
        if not (folder / 'TRAIN_COMPLETE').is_file() or not (folder / 'final.pt').is_file():
            raise RuntimeError(f'No completed training checkpoint in {folder}')
        eval_folder = folder / 'eval'
        if not (eval_folder / 'metrics.json').exists():
            logrun([ROOT / 'five_class_evaluate.py', '--data', DATA, '--checkpoint',
                    folder / 'final.pt', '--output', eval_folder, '--split', 'validation'],
                   folder, 'evaluate.log', gpu)
        read_metrics(job)
        (folder / 'COMPLETE').write_text(stamp() + '\n')
    except Exception as exc:
        atomic(folder / 'failure.json', dict(at=stamp(), error=str(exc)))
        raise


def summarize(all_jobs):
    material = {job: read_metrics(job) for job in all_jobs}
    contrasts = []
    for mode in (*MODES, 'all'):
        base, speed = ('C', 'D') if mode == 'all' else ('A', 'B')
        for s in SEEDS:
            left, right = (material[mode, arm, s] for arm in (base, speed))
            for k in ('request_sha256', 'noise_sha256'):
                if left[k] != right[k]:
                    raise ValueError(f'Unpaired {k} in {mode} seed {s}')
            contrasts.append(dict(mode=mode, seed=s, baseline=base, extension=speed,
                                  delta_speed_minus_baseline={k:right[k]-left[k] for k in KEYS},
                                  baseline_metrics=f'runs/{key((mode,base,s))}/eval/metrics.json',
                                  extension_metrics=f'runs/{key((mode,speed,s))}/eval/metrics.json'))
    atomic(ROOT / 'comparison.json', dict(completed=stamp(), evaluation_split='validation',
           fits=len(material), within_population_comparisons=contrasts,
           results={key(j):{k:v[k] for k in KEYS} for j,v in material.items()}))


def run():
    if not (DATA / 'VERIFIED').is_file():
        raise ValueError('Verified dataset required')
    dataset_hash = hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest()
    if dataset_hash != (ROOT / 'release.sha256').read_text().strip():
        raise ValueError('Unexpected verified release')
    all_jobs = jobs()
    if len(all_jobs) != 24 or any((m == 'all') != (a in ('C','D')) for m,a,s in all_jobs):
        raise ValueError('Invalid training matrix')
    pending = list(all_jobs)
    done, errors, blocked = set(), {}, {}
    active = {}
    free_gpus = list(range(4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        prep = pool.submit(prepare)
        prep_status = 'running'
        previous = None
        while pending or active or prep_status == 'running':
            if prep_status == 'running' and prep.done():
                try:
                    prep.result()
                    prep_status = 'complete'
                except Exception as exc:
                    prep_status = 'failed'
                    errors['prepare'] = str(exc)
            for future, (job,gpu) in list(active.items()):
                if not future.done():
                    continue
                try:
                    future.result()
                    done.add(job)
                except Exception as exc:
                    errors[key(job)] = str(exc)
                free_gpus.append(gpu)
                del active[future]
            for job in pending.copy():
                ancestor = parent(job)
                speed = job[1] in ('B','D')
                if (ancestor and (key(ancestor) in errors or key(ancestor) in blocked)) or (speed and prep_status=='failed'):
                    blocked[key(job)] = 'dependency failed'
                    pending.remove(job)
                    continue
                if not free_gpus or (ancestor and ancestor not in done) or (speed and prep_status != 'complete'):
                    continue
                gpu = free_gpus.pop(0)
                active[pool.submit(fit, job, gpu)] = (job, gpu)
                pending.remove(job)
            state = dict(prepare=prep_status, running={key(j):g for j,g in active.values()},
                         pending=list(map(key,pending)), completed=sorted(map(key,done)), errors=errors, blocked=blocked)
            encoded = json.dumps(state, sort_keys=True)
            if encoded != previous:
                atomic(ROOT / 'queue-status.json', dict(**state, updated=stamp()))
                previous = encoded
            if pending or active or prep_status == 'running':
                time.sleep(3)
    if errors or blocked:
        raise RuntimeError(f'Failed or blocked experiments: {errors} {blocked}')
    summarize(all_jobs)
    (ROOT/'COMPLETE').write_text(stamp()+'\n')


if __name__ == '__main__':
    with (ROOT/'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(0)
        if not (ROOT/'COMPLETE').is_file():
            try:
                run()
            except BaseException as exc:
                atomic(ROOT/'FAILED.json', dict(at=stamp(), error=str(exc)))
                raise
