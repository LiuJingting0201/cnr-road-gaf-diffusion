"""Publish complete E3/E4 metrics with fixed-population and auditor limitations."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--tag',default='long-selection-e4');a=p.parse_args()
out=Path('docs/results')/a.tag
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
search_note = ('All methods use the previously established search limit; this experiment resamples paths while freezing models.' if a.tag == 'route-diverse-e4r' else 'Expanded-limit runs are post-hoc sensitivity checks on the same saved candidates, not new training or resampling.')
lines+=['','## Representation and coverage diagnostics','',f'Lane candidate search limit per point: {reports[0]["max_candidates"]}. {search_note}']
for r in reports:
    lines.append(f'- Seed {r["seed"]}: valid candidates {r["valid_candidate_fraction"]:.3f}; identical GAF pools {r["identical_gaf_pool_fraction"]:.3f}; mean distinct images per eight draws {r["mean_unique_gaf_images_per_pool"]:.2f}; candidate-pair distance {r["candidate_pairwise_mean_distance_m"]:.2f} m; encoding failures {r["encoding_failures"]}.')
lines+=['','## Interpretation limits','',
        'Road consistency belongs mainly to the path decoder, not to GAF. Same-family auditor agreement can favor its selector; inspect both families and motion distributions. Real-reference auditor accuracy is recorded in summary.json and each seed report to expose evaluator limitations. A high classifier score is not synthetic-data utility.', '',
        'Taxi has only three evaluation conditions, motorcycle six. Seeds share these conditions and are not independent vehicles; mean/std is descriptive, not a statistical significance test. Full-trajectory and short-window historical experiments are different populations. Each condition produces one selected sample, so retained within-condition diversity is not established; only original candidate-pool diversity is reported.', '',
        'Next decision must combine independent class agreement, motion realism, selection fallback and image distinguishability before downstream augmentation. Preserve negative findings and assess broader route coverage; do not expand classifier architecture search.']
if a.tag=='route-diverse-e4r':
    lines+=['','## Route-motion candidate experiment','',
        'This is a post-E5 route-diversity experiment, not another search-cap sensitivity run. Each candidate has an independently planned path, using the same training-only prior and 4096m budget. Generator checkpoints, participant conditions, K=8, classifier checkpoints and initial DDIM noise are frozen. Compare with long-selection-e4-cap64; no new classifier or diffusion training.', '']
    for seed in [42,43,44]:
        r=json.loads(Path(f'runs/route-diverse-e4r/seed{seed}/complete.json').read_text())
        lines.append(f'- Seed {seed}: mean distinct paths {r["mean_unique_routes_per_pool"]:.2f}/8; planning fallbacks {r["route_planning_fallback"]}; route caps {r["route_end_capped_fraction"]:.3f}.')
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Small reproducibility records only; no model weights or raw trajectories.
for family in ['long-path-e3','long-selection-e4']+(['route-diverse-e4r'] if a.tag=='route-diverse-e4r' else []):
    r=Path('runs')/family
    for p in r.rglob('*'):
        if p.is_file() and p.name in ['config.json','complete.json','metrics.jsonl','plan.json','gate512.json','selected_data.json','training_complete.json']:
            d=out/'records'/family/p.relative_to(r)
            d.parent.mkdir(parents=True,exist_ok=True);d.write_text(p.read_text())
print('\n'.join(lines),flush=True)
