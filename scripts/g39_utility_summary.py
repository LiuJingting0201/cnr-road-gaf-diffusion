"""All-regime summary of completed independent G39 augmentation evidence."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def summarize(folder):
    receipt=json.loads((folder/'complete.json').read_text())
    assert receipt['complete'] and receipt['fit_count']==26
    regimes=['real']+[f'{k}-{a}' for k in ['only','plus'] for a in ['empirical','B','S','V','SV','RAW']]
    assert set(receipt['results'])=={f'{r}-{s}' for r in regimes for s in [3901,3902]}
    output={};rows=[]
    for r in regimes:
        fits=[receipt['results'][f'{r}-{s}'] for s in [3901,3902]]
        output[r]={'macro_f1_seeds':[x['pooled']['macro_f1'] for x in fits],
            'macro_f1_mean':mean(x['pooled']['macro_f1'] for x in fits),
            'accuracy_mean':mean(x['pooled']['accuracy'] for x in fits),
            'nonbus_parent_equal_f1_mean':mean(x['nonbus_source_parent_equal_macro_f1'] for x in fits),
            'per_class_f1_mean':[mean(x['pooled']['per_class_f1'][i] for x in fits) for i in range(5)]}
        for seed,x in zip([3901,3902],fits):
            for c,name in enumerate(['bus','motorcycle','private_car','bycicle','taxi_or_uber']):
                rows.append({'regime':r,'seed':seed,'participant':name,'f1':x['pooled']['per_class_f1'][c]})
    for r in regimes:
        output[r]['delta_vs_real_pp']=100*(output[r]['macro_f1_mean']-output['real']['macro_f1_mean'])
        output[r]['delta_vs_plus_B_pp']=100*(output[r]['macro_f1_mean']-output['plus-B']['macro_f1_mean'])
    (folder/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    with (folder/'per-class.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# G39 completed independent augmentation stage','',
        'All26 fixed fits are complete. Frozen historical tests were not used. This stage uses a separately trained raw-series CNN; no GAF input or teacher weights enter its evaluation.','',
        '| Regime | F1 seed3901 (%) | F1 seed3902 (%) | Mean (%) | Nonbus parent-equal mean (%) |',
        '|---|---:|---:|---:|---:|']
    for r in regimes:
        d=output[r];a,b=d['macro_f1_seeds']
        lines.append(f"| {r} | {a*100:.2f} | {b*100:.2f} | {d['macro_f1_mean']*100:.2f} | {d['nonbus_parent_equal_f1_mean']*100:.2f} |")
    lines+=['','`real`:1000 balanced real training parents; `empirical`:1000 resampled rows from that same base; each synthetic source:1000 matched unscored candidate identities. `only` trains on that source alone; `plus` adds it to the real base. All regimes receive2000 updates, final checkpoint, two downstream seeds.','',
        'The base diffusion augmentation improves mean macroF1 by'+f"{output['plus-B']['delta_vs_real_pp']:.2f}"+' percentage points versus real-only; empirical duplication does not provide that gain. This supports synthetic-data utility on these held-out conditions. Synthetic-only remains below real-only.','',
        'Speed-GAF augmentation adds'+f"{output['plus-V']['delta_vs_plus_B_pp']:.2f}"+' points over base augmentation, but raw-feature augmentation is numerically very close ('+f"{output['plus-V']['macro_f1_mean']*100:.2f}% versus {output['plus-RAW']['macro_f1_mean']*100:.2f}%"+'). This does not establish GAF-specific superiority over an alternative learned representation. The dual-GAF gain over base is smaller and not consistent in sign across the two seeds.','',
        'Evaluation uses two held-out OD cohorts repeated across windows, with common bus service templates and the same map. Do not turn two seeds into a population significance claim. Nonbus parent-equal results and every class/episode confusion are retained. Additional synthetic public OD conditions are part of the augmentation; this is not an isolated test of fine-scale dynamics.','',
        'Physical-quality results remain unchanged: the G39 main comparison does not show an overall native speed/acceleration improvement. Utility is a meaningful separate success criterion for data synthesis, not permission to describe all generated motion as better. Factorial mechanism review is still pending at this stage.','',
        'Sources: complete.json contains every fit and per-episode/per-class outcome; verification.json records independent artifact and prediction-order/F1 checks.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    summarize(p.parse_args().folder)
