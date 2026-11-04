"""Paired, class-stratified vehicle-cluster bootstrap of classifier gains.

Average draws, generator seeds and windows within vehicle before resampling.
These CIs describe this selected validation cohort, not a final test result.
"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from cnr.representations import RepresentationClassifier
from cnr.data import CLASSES

torch.set_num_threads(4)
manifest={int(r['array_index']):r['vehicle_id'] for r in csv.DictReader(open('data/processed/most64/manifest.csv')) if r['split']=='val'}
models={}
for name,variant in [('gaf','delta_gaf'),('tcn','xy_motion_tcn')]:
 models[name]=[]
 for seed in [43,44]:
  p=Path('runs/representation-r1')/f'{variant}-seed{seed}';cfg=json.loads((p/'config.json').read_text());m=RepresentationClassifier(cfg['variant'],cfg['stats']).cuda().eval()
  m.load_state_dict(torch.load(p/'best.pt',weights_only=True,map_location='cuda'));models[name].append(m)
def score(xy,lab):
 with torch.no_grad():
  x=torch.as_tensor(xy,device='cuda');return {name:torch.stack([torch.cat([m(x[j:j+64]).softmax(-1) for j in range(0,len(x),64)]) for m in group]).mean(0).argmax(-1).cpu().numpy()==lab for name,group in models.items()}
rows=[]
for family in ['path-g5-repeat','preserved-g6']:
 pooled={};cohort={};length_errors={}
 for seed in [42,43,44]:
  root=Path('runs')/family/f'seed{seed}';base=np.load(root/'none.npz');bs=score(base['xy'],base['labels'])
  for variant in ['gaf_0.3','gaf_1','gaf_3','tcn_0.3','tcn_1','tcn_3','shuffled_gaf']:
   s=np.load(root/f'{variant}.npz');assert np.array_equal(s['source_index'],base['source_index'])
   ss=score(s['xy'],s['labels']);length_errors.setdefault(variant,[]).append(float(np.max(np.abs(s['progress'][:,-1]-base['progress'][:,-1]))))
   for i,(idx,c) in enumerate(zip(s['source_index'],s['labels'])):
    vid=manifest[int(idx)];cohort[vid]=int(c)
    for name in models:pooled.setdefault((variant,name,vid),[]).append(float(ss[name][i])-float(bs[name][i]))
 class_ids={int(c):sorted(k for k,v in cohort.items() if v==c) for c in sorted(set(cohort.values()))}
 for variant in ['gaf_0.3','gaf_1','gaf_3','tcn_0.3','tcn_1','tcn_3','shuffled_gaf']:
  row={'family':family,'variant':variant,'vehicle_counts':{CLASSES[c]:len(v) for c,v in class_ids.items()},'max_arc_length_change_m':max(length_errors[variant])}
  for name in models:
   vectors=[np.array([np.mean(pooled[(variant,name,vid)]) for vid in vids]) for vids in class_ids.values()]
   rng=np.random.default_rng(2026);draws=np.mean([v[rng.integers(len(v),size=(2000,len(v)))].mean(1) for v in vectors],axis=0)
   row[name]={'vehicle_macro_paired_gain':float(np.mean([v.mean() for v in vectors])),'bootstrap_percentile_95':np.quantile(draws,[.025,.975]).tolist()}
  rows.append(row);print(json.dumps(row),flush=True)
out=Path('docs/results/g6-analysis');out.mkdir(parents=True,exist_ok=True)
(out/'vehicle_bootstrap.json').write_text(json.dumps({'bootstrap_replicates':2000,'rows':rows,'unit':'vehicle, class-stratified; average repetitions/seeds/windows first; macro vehicle-weighted estimates differ from window-weighted headline','scope':'posthoc validation exploration; no multiple-comparison correction; teacher seed fixed; uncertainty does not include teacher variation'},indent=2))
