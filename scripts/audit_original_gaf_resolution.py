"""Exact original encoding resolution on full trips, real windows and fixed-route samples."""
import hashlib,json
from pathlib import Path
import numpy as np
from cnr.original_gaf import trajectory_sequence,encode_frozen
p=Path('data/processed/original-hilbert224');meta=json.loads((p/'dataset.json').read_text());lo=np.array(meta['xy_min']);hi=np.array(meta['xy_max'])
classes=meta['source']['classes'];rows={}
def describe(h,labels):
 distinct=np.array([len(np.unique(x)) for x in h]);return {'samples':len(h),'constant_fraction':float(np.mean(distinct==1)),'median_distinct_hilbert_cells':float(np.median(distinct)),'per_class_constant_fraction':{name:float(np.mean(distinct[np.asarray(labels)==c]==1)) for c,name in enumerate(classes) if np.any(np.asarray(labels)==c)}}
full=np.load(p/'val.npz');rows['real_full_vehicle']=describe(full['sequences'],full['labels'])
w=np.load('data/processed/most64/val.npz');h=np.stack([trajectory_sequence(x,lo,hi)[0] for x in w['xy']]);rows['real_64_point_windows']=describe(h,w['labels'])
g=np.load('runs/path-g5-repeat/seed42/none.npz');h=np.stack([trajectory_sequence(x,lo,hi)[0] for x in g['xy']]);rows['generated_fixed_route_64']=describe(h,g['labels'])
images,_=encode_frozen(h,np.array(meta['channel_min']),np.array(meta['channel_max']));hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in images]
unique=[len({hashes[j] for j in np.flatnonzero(g['source_index']==i)}) for i in np.unique(g['source_index'])]
rows['generated_same_condition']={'conditions':len(unique),'draws_each':4,'all_four_images_identical_fraction':float(np.mean(np.array(unique)==1)),'mean_distinct_images':float(np.mean(unique))}
report={'rows':rows,'hilbert_eps':.0037,'depth':5,'grid_shape':[32,32],'cell_width_m':((hi-lo)/32).tolist(),'scope':'Representation diagnostics only; no classifier-guided generation was run. Full trajectories, real windows and generated windows are different populations, not a model performance comparison.'}
out=Path('docs/results/original-gaf-alignment');out.mkdir(parents=True,exist_ok=True);(out/'resolution.json').write_text(json.dumps(report,indent=2));(out/'dataset.json').write_text(json.dumps(meta,indent=2));print(json.dumps(report,indent=2))
