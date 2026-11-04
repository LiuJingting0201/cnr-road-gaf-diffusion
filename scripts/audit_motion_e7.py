"""Paired vehicle uncertainty and motion-copy diagnostics; saved validation only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from cnr.cluster_wasserstein import cluster_w1, nearest_motion

root = Path('runs/motion-audit-e7')
root.mkdir(parents=True, exist_ok=False)
out = Path('docs/results/motion-audit-e7')
out.mkdir(parents=True, exist_ok=False)
plan = {'pid': os.getpid(), 'bootstrap_repetitions': 2000, 'bootstrap_seed': 20261108,
        'comparison': 'empirical minus diffusion W1; positive favors diffusion',
        'population': '33 fixed validation vehicles, class-stratified paired clusters, all K=8 candidates',
        'conditioning': 'Fixed 3 diffusion checkpoints, 3 empirical pools and planned paths; no model refits',
        'copy_tolerance_mean_increment_m': .001,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
(root / 'plan.json').write_text(json.dumps(plan, indent=2))
(out / 'plan.json').write_text(json.dumps(plan, indent=2))


def motion(xy):
    speed = np.linalg.norm(np.diff(xy, axis=1), axis=-1) / .25
    return {'speed_w1_mps': speed, 'acceleration_w1_mps2': np.diff(speed, axis=1) / .25,
            'travel_w1_m': (speed.sum(1) * .25)[:, None]}


def memory_summary(query, labels, train, train_labels):
    raw, indices = nearest_motion(query, train, labels, train_labels)
    normalize = lambda x: x / np.maximum(x.mean(1, keepdims=True), 1e-6)
    shape, _ = nearest_motion(normalize(query), normalize(train), labels, train_labels)
    return {'samples': len(query), 'mean_nearest_increment_mae_m': float(raw.mean()),
            'median_nearest_increment_mae_m': float(np.median(raw)),
            'quantile05_nearest_increment_mae_m': float(np.quantile(raw, .05)),
            'mean_nearest_speed_profile_mae_mps': float(raw.mean() / .25),
            'near_copy_fraction_at_1mm_mean_increment': float(np.mean(raw <= .001)),
            'median_normalized_shape_mae': float(np.median(shape)),
            'unique_nearest_training_parents': len(np.unique(indices)),
            'per_class': {str(int(c)): {'median_increment_mae_m': float(np.median(raw[labels == c])),
                'near_copy_fraction': float(np.mean(raw[labels == c] <= .001))} for c in np.unique(labels)}}


try:
    cfg = json.loads(Path('runs/route-diverse-e4r/seed42/config.json').read_text())
    data = Path(cfg['data']); source = Path(cfg['dataset']['source_directory'])
    train = np.load(data / 'train.npz'); val = np.load(data / 'val.npz')
    raw = np.load(source / 'val.npz')
    training_increments = np.maximum(0, np.diff(train['progress'], axis=1))
    baseline = np.load('runs/route-diverse-e4r/seed42/candidates.npz')
    groups = [np.flatnonzero(baseline['condition_id'] == c) for c in np.unique(baseline['condition_id'])]
    condition_indices = np.array([g[0] for g in groups])
    source_index = baseline['source_index'][condition_indices]
    ids = raw['ids'][source_index]
    assert len(np.unique(ids)) == len(groups) == 33
    labels = baseline['labels'][condition_indices]
    classes = np.unique(labels)
    real_metrics = motion(raw['xy'][source_index])
    rng = np.random.default_rng(plan['bootstrap_seed'])
    counts = {int(c): np.vstack((np.ones(np.sum(labels == c)),
        rng.multinomial(np.sum(labels == c), np.full(np.sum(labels == c), 1 / np.sum(labels == c)),
                        size=plan['bootstrap_repetitions']))) for c in classes}
    results = {k: {} for k in real_metrics}; memory = {}; hashes = {}
    for family in ['route-diverse-e4r', 'empirical-motion-e3b']:
        family_w1 = {k: [] for k in real_metrics}; memory[family] = []
        for seed in [42, 43, 44]:
            path = Path(f'runs/{family}/seed{seed}/candidates.npz')
            sample = np.load(path); hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for k in ['route', 'route_length', 'labels', 'source_index', 'condition_id']:
                assert np.array_equal(sample[k], baseline[k]), k
            generated = motion(sample['xy'])
            old = json.loads(Path(f'docs/results/{family}/seed{seed}/results.json').read_text())
            expected = next(row for row in old['rows'] if row['variant'] == 'all_candidates')
            for key, values in generated.items():
                grouped = np.stack([values[g].ravel() for g in groups])
                by_class = [cluster_w1(grouped[labels == c], real_metrics[key][labels == c], counts[int(c)]) for c in classes]
                macro = np.mean(by_class, axis=0)
                assert np.isclose(macro[0], expected[key], rtol=1e-5, atol=1e-4), (family, seed, key, macro[0], expected[key])
                family_w1[key].append(macro)
                print(json.dumps({'family': family, 'seed': seed, 'metric': key, 'point': macro[0]}), flush=True)
            increments = np.maximum(0, np.diff(sample['progress'], axis=1))
            mem = memory_summary(increments, sample['labels'], training_increments, train['labels'])
            mem['seed'] = seed
            if family == 'empirical-motion-e3b':
                assert mem['near_copy_fraction_at_1mm_mean_increment'] == 1., 'Copy positive-control failed'
            memory[family].append(mem)
        for key in generated:
            results[key][family] = np.mean(family_w1[key], axis=0)
    rows = {}
    for key, families in results.items():
        diff = families['empirical-motion-e3b'] - families['route-diverse-e4r']
        rows[key] = {'diffusion': float(families['route-diverse-e4r'][0]),
            'empirical': float(families['empirical-motion-e3b'][0]),
            'empirical_minus_diffusion': float(diff[0]),
            'paired_percentile95': np.quantile(diff[1:], [.025, .975]).tolist()}
    index = {int(s): i for i, s in enumerate(val['source_index'])}
    selected = np.array([index[int(s)] for s in source_index])
    memory['real_validation_reference'] = memory_summary(np.maximum(0, np.diff(val['progress'][selected], axis=1)), labels,
                                                        training_increments, train['labels'])
    result = {'paired_w1': rows, 'memory': memory, 'plan': plan, 'candidate_hashes': hashes,
              'class_vehicle_counts': {str(int(c)): int(np.sum(labels == c)) for c in classes}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    lines = ['# Motion uncertainty and training-copy diagnostics', '',
        'All saved K=8 candidates; no selection or resampling of generated trajectories. 2,000 class-stratified vehicle bootstrap draws are shared across generators, all three fixed seeds, and metrics. Each vehicle retains its full candidate pool and real reference. Exact empirical-CDF W1, no histogram approximation. Test untouched.', '',
        '| Metric | Diffusion | Empirical | Empirical minus diffusion [paired 95% interval] |', '|---|---:|---:|---:|']
    for key, r in rows.items():
        lo, hi = r['paired_percentile95']
        lines.append(f'| {key} | {r["diffusion"]:.4f} | {r["empirical"]:.4f} | {r["empirical_minus_diffusion"]:.4f} [{lo:.4f}, {hi:.4f}] |')
    lines += ['', 'Positive differences favor diffusion. Intervals condition on saved models, paths and pools; they do not cover training variability or correct exploratory multiple comparisons. Only 33 vehicles, including three taxis, constrain inference.', '',
        '## Motion-domain nearest training profiles', '',
        'Same-class, full 511-step arc increments aligned in time. Near-copy tolerance is mean absolute difference <= 0.001m per increment, fixed before running. Uncapped intended progress is audited, so route truncation cannot hide a copied profile. Shape distance divides each profile by its own mean. These are diagnostics, not a proof of absence of memorization.', '',
        '| Population | Seed | Median nearest increment MAE m | Near-copy % | Median normalized shape MAE |', '|---|---:|---:|---:|---:|']
    for family, records in memory.items():
        for rec in records if isinstance(records, list) else [records]:
            lines.append(f'| {family} | {rec.get("seed", "reference")} | {rec["median_nearest_increment_mae_m"]:.6f} | {100*rec["near_copy_fraction_at_1mm_mean_increment"]:.2f} | {rec["median_normalized_shape_mae"]:.4f} |')
    lines += ['', 'Empirical motion is the known-copy positive control. A low copy rate for diffusion rules out only this narrow aligned-neighbor criterion, not shifted, warped, partial or other forms of memorization. Real references are evaluation-only; their future motion is never given to synthesis. No new GAF usefulness claim follows from this audit.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    done = {'complete': True, 'time': time.time()}
    (root / 'complete.json').write_text(json.dumps(done)); (out / 'complete.json').write_text(json.dumps(done))
except Exception as ex:
    failure = {'error': str(ex), 'time': time.time()}
    (root / 'failed.json').write_text(json.dumps(failure)); (out / 'failed.json').write_text(json.dumps(failure))
    raise
