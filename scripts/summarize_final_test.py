"""Assemble all locked comparisons, paired W1 uncertainty and motion-copy checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from cnr.final_protocol import verify_lock, sha256
from cnr.cluster_wasserstein import cluster_w1, nearest_motion

p = argparse.ArgumentParser(); p.add_argument('--lock', required=True); a = p.parse_args()
lock = verify_lock(a.lock)
out = Path('docs/results/final-test-e7'); out.mkdir(parents=True, exist_ok=False)
data = Path('data/processed/final-test-e7')
coverage = json.loads((data / 'complete.json').read_text())
planning = json.loads((data / 'planned/coverage.json').read_text())
raw = np.load(data / 'reference/test.npz')
first = np.load('runs/final-diffusion-e7/seed42/candidates.npz')
groups = [np.flatnonzero(first['condition_id'] == c) for c in np.unique(first['condition_id'])]
ix = np.array([g[0] for g in groups]); labels = first['labels'][ix]; sources = first['source_index'][ix]
assert len(np.unique(raw['ids'][sources])) == len(groups)
def motion(x):
    speed = np.linalg.norm(np.diff(x, axis=1), axis=-1) / .25
    return {'speed_w1_mps': speed, 'acceleration_w1_mps2': np.diff(speed, axis=1) / .25,
            'travel_w1_m': (speed.sum(1) * .25)[:, None]}
reference = motion(raw['xy'][sources]); classes = np.unique(labels)
rng = np.random.default_rng(20261112)
counts = {int(c): np.vstack((np.ones(np.sum(labels == c)),
    rng.multinomial(np.sum(labels == c), np.full(np.sum(labels == c), 1 / np.sum(labels == c)), size=2000))) for c in classes}
train = np.load('data/processed/long-path-e3-512/train.npz')
train_inc = np.maximum(0, np.diff(train['progress'], axis=1))
normalize = lambda x: x / np.maximum(x.mean(1, keepdims=True), 1e-6)
def memory(q, labs):
    d, _ = nearest_motion(q, train_inc, labs, train['labels'])
    shape, _ = nearest_motion(normalize(q), normalize(train_inc), labs, train['labels'])
    return {'median_increment_mae_m': float(np.median(d)), 'near_copy_fraction': float(np.mean(d <= .001)),
            'median_normalized_shape_mae': float(np.median(shape))}
reports = {}; bootstrap = {}; memories = {}
keys = ['graph_consistent', 'capped', 'speed_w1_mps', 'acceleration_w1_mps2', 'travel_w1_m',
        'gaf_audit_accuracy', 'tcn_audit_accuracy', 'stop_fraction']
for family in ['diffusion', 'empirical']:
    rows = []; boots = {k: [] for k in reference}; memories[family] = []
    for seed in [42, 43, 44]:
        report = json.loads(Path(f'docs/results/final-{family}-e7/seed{seed}/results.json').read_text()); rows.append(report)
        sample = np.load(f'runs/final-{family}-e7/seed{seed}/candidates.npz')
        for key in ['route', 'route_length', 'labels', 'source_index', 'condition_id']:
            assert np.array_equal(sample[key], first[key]), key
        expected = next(r for r in report['rows'] if r['variant'] == 'all_candidates')
        for key, values in motion(sample['xy']).items():
            grouped = np.stack([values[g].ravel() for g in groups])
            macro = np.mean([cluster_w1(grouped[labels == c], reference[key][labels == c], counts[int(c)]) for c in classes], axis=0)
            assert np.isclose(macro[0], expected[key], atol=1e-4, rtol=1e-5)
            boots[key].append(macro)
        mem = memory(np.maximum(0, np.diff(sample['progress'], axis=1)), sample['labels']) | {'seed': seed}
        if family == 'empirical': assert mem['near_copy_fraction'] == 1.
        memories[family].append(mem)
    bootstrap[family] = {k: np.mean(v, axis=0) for k, v in boots.items()}
    reports[family] = {}
    for method in ['all_candidates', 'random', 'gaf', 'tcn', 'shuffled_gaf']:
        selected = [next(r for r in report['rows'] if r['variant'] == method) for report in rows]
        reports[family][method] = {k: {'mean': float(np.mean([r[k] for r in selected])),
            'std_population': float(np.std([r[k] for r in selected]))} for k in keys}
        reports[family][method]['fallback_conditions'] = [r['fallback_conditions'] for r in selected]
    reports[family]['diagnostics'] = [{k: r[k] for k in ['seed', 'candidate_pairwise_mean_distance_m',
        'mean_unique_gaf_images_per_pool', 'valid_candidate_fraction', 'encoding_failures']} for r in rows]
paired = {}
for key in reference:
    delta = bootstrap['empirical'][key] - bootstrap['diffusion'][key]
    paired[key] = {'empirical_minus_diffusion': float(delta[0]), 'percentile95': np.quantile(delta[1:], [.025, .975]).tolist()}
observed = np.load(data / 'reference/progress.npz'); lookup = {int(s): i for i, s in enumerate(observed['source_index'])}
memories['real_test_reference'] = memory(np.maximum(0, np.diff(observed['progress'][[lookup[int(s)] for s in sources]], axis=1)), labels)
downstream = json.loads(Path('docs/results/final-downstream-e7/summary.json').read_text())
auxiliary = json.loads(Path('docs/results/final-auxiliary-e7/summary.json').read_text())
summary = {'coverage': coverage, 'planning': planning, 'generation': reports, 'paired_w1': paired,
           'motion_memory': memories, 'downstream': downstream, 'auxiliary': auxiliary,
           'missing_generation_classes': [int(c) for c in range(1, 6) if c not in classes],
           'lock_sha256': sha256(a.lock), 'scope': 'Locked final held-out-ID test, fixed models/pools; exploratory validation selection disclosed'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'LOCK_MANIFEST.json').write_text(json.dumps(lock, indent=2))
(out / 'PROTOCOL.md').write_text(Path('docs/FINAL_TEST_PROTOCOL.md').read_text())
lines = ['# Locked final held-out-ID evaluation', '',
    f'Test IDs: {coverage["test_ids_total"]}; encodable 512-point windows: {coverage["encodable_windows"]}; strict moving requests: {planning["requested_conditions"]}; complete K=8 planned pools: {planning["served_conditions"]}; unserved conditions: {planning["failed_conditions"]}. No missing pools are replaced with real future routes.', '',
    f'Generation class counts before planning: {coverage["generation_class_counts"]}. Missing served classes: {summary["missing_generation_classes"]}. Macro generation values average supported classes only. All eligible test conditions are used, unlike the 33-condition validation pilot.', '',
    '| Generator | Selection | Graph % | Speed W1 | Acceleration W1 | Travel W1 | GAF audit % | TCN audit % | Cap % |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for family, methods in reports.items():
    for method, row in methods.items():
        if method == 'diagnostics': continue
        values = []
        for k in ['graph_consistent', 'speed_w1_mps', 'acceleration_w1_mps2', 'travel_w1_m', 'gaf_audit_accuracy', 'tcn_audit_accuracy', 'capped']:
            values.append(f'{row[k]["mean"]*(1 if "w1" in k else 100):.3f}')
        lines.append('| '+family+' | '+method+' | '+' | '.join(values)+' |')
lines += ['', '## Paired vehicle uncertainty', '', 'Positive empirical-minus-diffusion W1 favors diffusion. Fixed models, routes and pools; 2,000 paired class-stratified vehicle resamples, not training variability or multiplicity-adjusted inference.', '',
          '| Metric | Difference [95% interval] |', '|---|---:|']
for key, r in paired.items():
    lo, hi = r['percentile95']; lines.append(f'| {key} | {r["empirical_minus_diffusion"]:.4f} [{lo:.4f}, {hi:.4f}] |')
lines += ['', '## Downstream synthetic-data utility', '', 'Same 36 fixed E5 learners, six classes, no test-driven retraining. Macro-F1 is averaged across the three learner seeds. Intervals are conditional on one augmentation generator and pool.', '',
          '| Learner | Group | Macro-F1 % | GAF minus this group pp [95% interval] |', '|---|---|---:|---:|']
for arch, methods in downstream['models'].items():
    for group, row in methods.items():
        d = methods['synthetic_gaf']['paired_delta_f1'][group]; lo, hi = d['percentile95']
        lines.append(f'| {arch} | {group} | {row["macro_f1_mean"]*100:.2f} | {d["point"]*100:.2f} [{lo*100:.2f}, {hi*100:.2f}] |')
lines += ['', '## Auxiliary classifier audit on all encodable test windows', '',
          '| Self-trained family | Accuracy % | Six-class macro-F1 % |', '|---|---:|---:|']
for family, row in auxiliary['models'].items():
    lines.append(f'| {family} | {row["accuracy_mean"]*100:.2f} | {row["macro_f1_mean"]*100:.2f} |')
lines += ['', '## Limits and interpretation', '',
    'GAF classifiers are self-trained on inherited Hilbert/GASF/GADF/MTF, not supplied group CNNs. Same-family auditor agreement is biased toward its selector and cannot establish utility. Road consistency mostly follows from the common path decoder, not diffusion learning or GAF. Pedestrians are not synthesized; no traffic-light, collision, city-transfer or precise SUMO-version claim is made.', '',
    'Full aligned motion nearest-training diagnostics are in summary.json. Near-copy tolerance is 0.001m mean increment difference, with empirical motion as a known-copy control; this does not detect every form of memorization. Larger candidate distance need not mean more realistic diversity. All failures and coverage restrictions remain in the report. Final tests do not authorize further result-driven model selection.', '',
    'Fixed map examples are generated separately: seed42, first served condition per class, every method from the same pool.']
(out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines), flush=True)
