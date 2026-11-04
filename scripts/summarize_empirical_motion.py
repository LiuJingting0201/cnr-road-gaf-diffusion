"""E3b: compare empirical motion and learned diffusion on identical candidate paths."""
import json
from pathlib import Path
import numpy as np

out = Path('docs/results/empirical-motion-e3b')
metrics = ['graph_consistent', 'capped', 'speed_w1_mps', 'acceleration_w1_mps2',
           'travel_w1_m', 'gaf_audit_accuracy', 'tcn_audit_accuracy', 'stop_fraction']
summary = {}
for family in ['route-diverse-e4r', 'empirical-motion-e3b']:
    reports = [json.loads(Path(f'docs/results/{family}/seed{s}/results.json').read_text()) for s in [42, 43, 44]]
    result = {}
    for method in ['all_candidates', 'random', 'gaf', 'tcn', 'shuffled_gaf']:
        rows = [next(r for r in rep['rows'] if r['variant'] == method) for rep in reports]
        result[method] = {k: {'mean': float(np.mean([r[k] for r in rows])),
            'std_population': float(np.std([r[k] for r in rows]))} for k in metrics}
        result[method]['fallback_conditions'] = [r['fallback_conditions'] for r in rows]
    result['diagnostics'] = [{k: rep[k] for k in ['seed', 'valid_candidate_fraction',
        'candidate_pairwise_mean_distance_m', 'median_nearest_training_track_32point_distance_m',
        'encoding_failures', 'mean_unique_gaf_images_per_pool']} for rep in reports]
    summary[family] = result
for seed in [42, 43, 44]:
    d = np.load(f'runs/route-diverse-e4r/seed{seed}/candidates.npz')
    e = np.load(f'runs/empirical-motion-e3b/seed{seed}/candidates.npz')
    for k in ['route', 'route_length', 'labels', 'source_index', 'condition_id']:
        assert np.array_equal(d[k], e[k]), (seed, k)
    assert np.allclose(d['xy'][:, 0], e['xy'][:, 0], atol=1e-4)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
lines = ['# Empirical motion versus learned path diffusion', '',
    'Same 33 validation conditions, K=8, 512 points, candidate paths and shared decoder. Empirical baseline draws entire class-matched training arc-increment sequences, with replacement, then maps them onto the same paths. No validation future is supplied. Frozen test untouched.', '',
    'Diffusion has three training seeds; empirical motion has three resampling seeds and no learned generator. Standard deviations are descriptive, not a significance test. Both use the same frozen original-Hilbert GAF and raw TCN selectors/auditors. Teacher agreement does not establish utility.', '',
    '| Generator | Selection | Graph % | Speed W1 m/s | Acceleration W1 m/s2 | Travel W1 m | GAF audit % | TCN audit % | Cap % |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for family, result in summary.items():
    for method, row in result.items():
        if method == 'diagnostics': continue
        values = []
        for k in ['graph_consistent', 'speed_w1_mps', 'acceleration_w1_mps2', 'travel_w1_m',
                  'gaf_audit_accuracy', 'tcn_audit_accuracy', 'capped']:
            scale = 1 if 'w1' in k else 100
            values.append(f'{row[k]["mean"]*scale:.2f} +/- {row[k]["std_population"]*scale:.2f}')
        lines.append('| ' + family + ' | ' + method + ' | ' + ' | '.join(values) + ' |')
lines += ['', '## Interpretation limits', '',
    'The empirical baseline deliberately reuses complete training motion profiles; geometry is new but motion is memorized. It is a strong no-training baseline, not an independent simulator. Diffusion needs evidence beyond topology, which both methods inherit from the same path decoder. Absolute-coordinate nearest-training distances cannot alone detect copied speed profiles.', '',
    'Only 425 strict moving training tracks and 33 evaluation conditions are represented. All retained candidates and fallback counts are reported, including caps and invalid trajectories; no failed samples are silently replaced. This is a post-E5 validation comparison, not a final test or downstream augmentation experiment.']
(out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
root = Path('runs/empirical-motion-e3b')
for path in root.rglob('*.json'):
    if path.name in ['config.json', 'complete.json', 'plan.json', 'failed.json']:
        dest = out / 'records' / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(path.read_text())
print('\n'.join(lines), flush=True)
