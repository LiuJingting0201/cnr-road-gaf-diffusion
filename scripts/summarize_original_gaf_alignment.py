import json,statistics
from pathlib import Path
root=Path('docs/results/original-gaf-alignment');results=[json.loads((root/f'seed{s}/results.json').read_text()) for s in [42,43,44]]
summary={}
for scope in ['full_vehicle_validation','short_window_transfer']:
 summary[scope]={k:{'mean':statistics.mean(x[scope][k] for x in results),'seed_sd':statistics.stdev(x[scope][k] for x in results)} for k in ['accuracy','macro_f1']}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Original Hilbert / GAF participant-classification alignment','',
 'User clarification: the group CNN is not supplied. The label is participant class. This run trains a new CNN on the original representation; it does not claim to reproduce an unknown group architecture or reported accuracy.','',
 'Original source: commit 4773ad65bce88d38b8b108ef5e4773fd99883c94; five source files retained unchanged with hashes. Full-frame sequence and RGB pixel parity tests passed. Train-only coordinate/channel bounds are explicit deployment adaptations, frozen for validation.','',
 'Original pipeline: full vehicle XY -> normalized pointwise Hilbert (eps=.0037) -> original squeeze/mirror to 224 -> GASF/GADF/8-bin MTF -> RGB uint8. Not displacement GAF.','',
 '2,146 training vehicles / 269 validation vehicles. Seeds 42/43/44, 4,000 CNN updates each, fixed final checkpoint. No test encoding or evaluation. 26 repository tests passed.','',
 '| Real evaluation population | Accuracy % | Macro-F1 % |','|---|---:|---:|']
for scope in summary:
 d=summary[scope];values=[f"{100*d[k]['mean']:.2f} +/- {100*d[k]['seed_sd']:.2f}" for k in ['accuracy','macro_f1']];lines.append('| '+scope+' | '+' | '.join(values)+' |')
lines+=['','The second row is transfer to 2,791 real 64-point windows with the same frozen encoder and full-vehicle-trained CNN. Populations differ; this is a time-scale/domain-shift diagnostic, not a fair head-to-head classifier benchmark. It is NOT generated-trajectory classification performance.','',
 'The original precision is a 32x32 grid, approximately 312m x 195m per cell in this dataset. 16.73% of full validation trajectories, 46.86% of real short windows, and 50.94% of the existing 320 generated windows have a constant Hilbert sequence. For 37.5% of the 80 generated conditions, all four candidates produce identical RGB images. See resolution.json for per-class results.','',
 'Decision: original-encoding full-trajectory classification has usable but incomplete class information; directly transferring this CNN to short-fragment guidance is unsupported. Do not restart the custom displacement-GAF guidance or claim that poor transfer disproves the group method. First match duration and representation resolution within the original Hilbert/GAF family, with every adaptation explicit. Discrete Hilbert and uint8 operations do not provide a coordinate gradient; score-based candidate/resampling methods are a later option only after classifier suitability is established.','',
 'The final objective remains map-conditioned participant-specific trajectory synthesis. This audit is a prerequisite, not a switch to image generation or a classification-only thesis.']
(root/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))
