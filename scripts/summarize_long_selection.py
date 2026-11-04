"""Publish complete E3/E4 metrics with fixed-population and auditor limitations."""
import json
from pathlib import Path
import numpy as np

out=Path('docs/results/long-selection-e4')
reports=[json.loads((out/f'seed{s}/results.json').read_text()) for s in [42,43,44]]
fields=['graph_consistent','capped','speed_w1_mps','acceleration_w1_mps2','travel_w1_m','gaf_audit_accuracy','tcn_audit_accuracy','real_gaf_audit_accuracy','real_tcn_audit_accuracy','stop_fraction']
summary={}
for variant in ['all_candidates','random','gaf','tcn','shuffled_gaf']:
    rows=[next(r for r in rep['rows'] if r['variant']==variant) for rep in reports]
    summary[variant]={k:{'mean':float(np.mean([r[k] for r in rows])),'std_population':float(np.std([r[k] for r in rows]))} for k in fields}
    summary[variant]['fallback_conditions']=[r['fallback_conditions'] for r in rows]
(out/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Long trajectory diffusion and original GAF candidate selection','',
       'Three diffusion seeds, 20,000 updates each, 512 points (127.75 s), five moving non-pedestrian classes. 425 training / 57 available validation tracks; generation uses 33 selected validation conditions and K=8 draws each. Planned validation paths use only start, class and the training-only transition prior. Test untouched.','',
       'All selection methods use the same candidate pool and shared validity mask. Teacher seed42; auditors seeds43/44. GAF uses inherited Hilbert/squeeze/GASF/GADF/MTF, with duration-matched self-trained ResNet18. The raw trajectory TCN uses the same real training cohort and 50 epochs. Different network families and ImageNet initialization are disclosed, not claimed to be identical architectures.','',
       '| Method | Graph consistency % | GAF audit % | Raw TCN audit % | Speed W1 m/s | Acceleration W1 m/s2 | Route cap % |',
       '|---|---:|---:|---:|---:|---:|---:|']
for name,row in summary.items():
    vals=[]
    for key in ['graph_consistent','gaf_audit_accuracy','tcn_audit_accuracy','speed_w1_mps','acceleration_w1_mps2','capped']:
        scale=1 if 'w1' in key else 100
        vals.append(f'{row[key]["mean"]*scale:.2f} +/- {row[key]["std_population"]*scale:.2f}')
    lines.append('| '+name+' | '+' | '.join(vals)+' |')
lines+=['','## Representation and coverage diagnostics','']
for r in reports:
    lines.append(f'- Seed {r["seed"]}: valid candidates {r["valid_candidate_fraction"]:.3f}; identical GAF pools {r["identical_gaf_pool_fraction"]:.3f}; mean distinct images per eight draws {r["mean_unique_gaf_images_per_pool"]:.2f}; candidate-pair distance {r["candidate_pairwise_mean_distance_m"]:.2f} m; encoding failures {r["encoding_failures"]}.')
lines+=['','## Interpretation limits','',
        'Road consistency belongs mainly to the path decoder, not to GAF. Same-family auditor agreement can favor its selector; inspect both families and motion distributions. Real-reference auditor accuracy is recorded in summary.json and each seed report to expose evaluator limitations. A high classifier score is not synthetic-data utility.', '',
        'Taxi has only three evaluation conditions, motorcycle six. Seeds share these conditions and are not independent vehicles; mean/std is descriptive, not a statistical significance test. Full-trajectory and short-window historical experiments are different populations. Each condition produces one selected sample, so retained within-condition diversity is not established; only original candidate-pool diversity is reported.', '',
        'Next decision must combine independent class agreement, motion realism, selection fallback and image distinguishability before downstream augmentation. Preserve negative findings and assess broader route coverage; do not expand classifier architecture search.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Small reproducibility records only; no model weights or raw trajectories.
for family in ['long-path-e3','long-selection-e4']:
    r=Path('runs')/family
    for p in r.rglob('*'):
        if p.is_file() and p.name in ['config.json','complete.json','metrics.jsonl','plan.json','gate512.json','selected_data.json','training_complete.json']:
            d=out/'records'/family/p.relative_to(r)
            d.parent.mkdir(parents=True,exist_ok=True);d.write_text(p.read_text())
print('\n'.join(lines),flush=True)
