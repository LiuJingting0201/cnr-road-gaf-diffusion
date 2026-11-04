"""E5 fixed-final-step utility and paired class-stratified vehicle bootstrap."""
import json
from pathlib import Path
import numpy as np

root=Path('runs/augmentation-e5');out=Path('docs/results/augmentation-e5');out.mkdir(parents=True,exist_ok=False)
meta=json.loads(Path('data/processed/augmentation-e5/dataset.json').read_text());groups=meta['groups']
seeds=[100,101,102];result={}

def macro(y,p):
    cm=np.bincount(y*6+p,minlength=36).reshape(6,6)
    return float(np.mean(2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)))

for arch in ['tcn','gru']:
    runs={};preds={};reference_ids=None
    for group in groups:
        rows=[];pr=[]
        for seed in seeds:
            d=root/f'{arch}-{group}-seed{seed}';r=json.loads((d/'complete.json').read_text());assert r['complete']
            z=np.load(d/'validation.npz');y=z['labels']
            if reference_ids is None:reference_ids=z['ids'].copy()
            assert np.array_equal(reference_ids,z['ids'])
            rows.append(r);pr.append(z['probabilities'].argmax(-1))
            target=out/d.name;target.mkdir()
            for f in ['complete.json','metrics.jsonl','config.json']:(target/f).write_text((d/f).read_text())
        preds[group]=np.array(pr)
        runs[group]={k:{'mean':float(np.mean([r['final_validation'][k] for r in rows])),'std_population':float(np.std([r['final_validation'][k] for r in rows]))} for k in ['accuracy','macro_f1']}
        runs[group]['mean_per_class_recall']=np.mean([r['final_validation']['per_class_recall'] for r in rows],axis=0).tolist()
    rng=np.random.default_rng(20261106)
    samples=[np.concatenate([rng.choice(np.flatnonzero(y==c),np.sum(y==c),replace=True) for c in range(6)]) for _ in range(2000)]
    boot={group:np.array([np.mean([macro(y[ix],p[ix]) for p in preds[group]]) for ix in samples]) for group in groups}
    for group in groups:
        runs[group]['paired_bootstrap_macro_f1_delta']={}
        for comparator in ['real_only','repeat','synthetic_random']:
            diff=boot[group]-boot[comparator]
            runs[group]['paired_bootstrap_macro_f1_delta'][comparator]={'observed':runs[group]['macro_f1']['mean']-runs[comparator]['macro_f1']['mean'],'percentile95':np.quantile(diff,[.025,.975]).tolist()}
    result[arch]=runs
(out/'summary.json').write_text(json.dumps({'models':result,'dataset':meta,'bootstrap':'2000 paired class-stratified vehicle resamples; fixed three learner initializations and one generator; exploratory, not multiplicity-adjusted'},indent=2))
lines=['# Downstream utility of training-only trajectory synthesis','',
       'Known training-path motion augmentation: one diffusion checkpoint (seed42), 425 training conditions, eight candidates each; one selected sample per condition. No validation trajectory is used as an augmentation condition. Real downstream data: 1,437 train / 164 validation, six classes. Synthetic extras cover five classes, no pedestrians. Final test untouched.','',
       'Six groups use fixed 1,150 updates of 64 uniform-with-replacement draws. Normalization and class weights are frozen from real training data. Three fresh learner seeds 100/101/102, TCN and bidirectional GRU. No selector weights reused. All augmented groups have the same 425 extra labels and parent conditions; repeat/timewarp control the changed class frequency.','',
       '| Learner | Training group | Macro-F1 % | Accuracy % | Delta F1 vs real-only pp [paired 95% interval] |','|---|---|---:|---:|---:|']
for arch,rows in result.items():
    for group,row in rows.items():
        f=row['macro_f1'];acc=row['accuracy'];d=row['paired_bootstrap_macro_f1_delta']['real_only']
        lines.append(f'| {arch} | {group} | {f["mean"]*100:.2f} +/- {f["std_population"]*100:.2f} | {acc["mean"]*100:.2f} +/- {acc["std_population"]*100:.2f} | {d["observed"]*100:.2f} [{d["percentile95"][0]*100:.2f}, {d["percentile95"][1]*100:.2f}] |')
lines+=['','## Scope and interpretation','',
        f'Validation counts by class (pedestrian, bus, motorcycle, car, bicycle, taxi): {meta["validation_class_counts"]}. Small minority classes limit certainty. Intervals condition on one generator checkpoint and a fixed synthesis pool; they do not include generator variation or correct multiple comparisons.', '',
        'This is validation development, not a final test result or a low-label experiment. Known training paths make this motion augmentation, not novel-route synthesis. The timewarp control uses piecewise interpolation of raw coordinates and is a simple training perturbation, not a separately validated SUMO simulator.', '',
        'Compare synthetic methods with both real-only and repeat/timewarp controls, and GAF-selected with random synthetic on both learner families. Teacher agreement does not establish downstream gain; report negative results. All selection fallbacks and input hashes are in summary.json.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
for name in ['plan.json','pool/config.json','pool/complete.json']:
    dest=out/'records'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text((root/name).read_text())
print('\n'.join(lines),flush=True)
