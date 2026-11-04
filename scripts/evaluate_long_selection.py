"""E3/E4: original-Hilbert GAF selection vs raw TCN on identical K=8 pools.

All candidate outcomes retained. Independent seed auditors are not unbiased ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torchvision.models import resnet18
from cnr.candidate_selection import select_candidates
from cnr.data import CLASSES
from cnr.original_gaf import trajectory_sequence, encode_frozen
from cnr.representations import RepresentationClassifier
from cnr.topology import LaneConsistency

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--max-candidates',type=int,default=12);p.add_argument('--tag',default='long-selection-e4');a=p.parse_args()
torch.set_num_threads(4)
run=Path(f'runs/long-path-e3/seed{a.seed}')
cfg=json.loads((run/'config.json').read_text())
data=Path(cfg['data']);routeval=np.load(data/'val.npz')
source=Path(cfg['dataset']['source_directory'])
raw=np.load(source/'val.npz');training=np.load(data/'train.npz')['xy']
s=np.load(run/'candidates.npz');xy=s['xy'];labels=s['labels'];real=raw['xy'][s['source_index']]
assert xy.shape==real.shape and np.isfinite(xy).all()
assert np.array_equal(labels,raw['labels'][s['source_index']])
groups=[np.flatnonzero(s['condition_id']==i) for i in np.unique(s['condition_id'])]
assert all(len(g)==8 and len(np.unique(s['source_index'][g]))==1 for g in groups)
meta=json.loads((source/'dataset.json').read_text());length=xy.shape[1]
out=Path(f'docs/results/{a.tag}/seed{a.seed}');out.mkdir(parents=True,exist_ok=False)
predroot=Path(f'runs/{a.tag}/seed{a.seed}');predroot.mkdir(parents=True,exist_ok=False)
provenance={}

def gaf_images(tracks):
    seq=[];ok=[];clipped=0
    for track in tracks:
        try:
            h,n=trajectory_sequence(track,np.array(meta['xy_min']),np.array(meta['xy_max']),eps=meta['hilbert_eps'])
            ok.append(True);clipped+=n
        except ValueError:
            h=np.zeros(224);ok.append(False)
        seq.append(h)
    im,n=encode_frozen(seq,np.array(meta['channel_min']),np.array(meta['channel_max']))
    return im,np.array(ok),clipped,n

images,enc_valid,coord_clip,channel_clip=gaf_images(xy)
real_images,real_enc,_,_=gaf_images(real)
assert real_enc.all()
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

@torch.no_grad()
def infer(model,values,gaf):
    result=[]
    for i in range(0,len(values),32):
        x=torch.from_numpy(values[i:i+32]).cuda().float()
        if gaf:x=(x.permute(0,3,1,2)/255-mean)/std
        result.append(model(x).softmax(-1).cpu().numpy())
    return np.concatenate(result)

prob={'gaf':{},'tcn':{}};real_prob={'gaf':{},'tcn':{}}
for seed in [42,43,44]:
    path=Path(f'runs/gaf-duration-e2/length{length}-seed{seed}')
    checkpoint=path/'final.pt'
    model=resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,6)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model=model.cuda().eval()
    prob['gaf'][seed]=infer(model,images,True);real_prob['gaf'][seed]=infer(model,real_images,True)
    provenance[f'gaf{seed}']={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    del model
    path=Path(f'runs/long-selection-e4/tcn-seed{seed}')
    c=json.loads((path/'config.json').read_text());checkpoint=path/'latest.pt'
    model=RepresentationClassifier('xy_motion_tcn',c['stats'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model=model.cuda().eval()
    prob['tcn'][seed]=infer(model,xy,False);real_prob['tcn'][seed]=infer(model,real,False)
    provenance[f'tcn{seed}']={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    del model

auditors={k:(v[43]+v[44])/2 for k,v in prob.items()}
real_auditors={k:(v[43]+v[44])/2 for k,v in real_prob.items()}
matcher=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=a.max_candidates)
graphs=[]
for i,(x,label) in enumerate(zip(xy,labels)):
    graphs.append(matcher.match(x,CLASSES[int(label)]))
    if i%64==0:print(json.dumps({'seed':a.seed,'graph_progress':i,'total':len(xy)}),flush=True)
graph=np.array([g['graph_consistent'] for g in graphs])
truncated=np.array([g['candidate_truncation_warning'] for g in graphs])
capped=s['progress'][:,-1]>=s['route_length']
valid=graph & ~truncated & ~capped & enc_valid
# No motion cutoff is tuned on this evaluation. Motion realism is reported below.
speed=np.linalg.norm(np.diff(xy,axis=1),axis=-1)/.25
rspeed=np.linalg.norm(np.diff(real,axis=1),axis=-1)/.25
acc=np.diff(speed,axis=1)/.25;racc=np.diff(rspeed,axis=1)/.25
travel=speed.sum(1)*.25;rtravel=rspeed.sum(1)*.25
rng=np.random.default_rng(20261104)
order=rng.random(len(xy))
target=np.array([labels[g[0]] for g in groups]);shuffled=rng.permutation(target)
shuffled_per_sample=np.empty(len(xy),dtype=np.int64)
for g,c in zip(groups,shuffled):shuffled_per_sample[g]=c
gaf_score=prob['gaf'][42][np.arange(len(xy)),labels]
tcn_score=prob['tcn'][42][np.arange(len(xy)),labels]
shuffle_score=prob['gaf'][42][np.arange(len(xy)),shuffled_per_sample]
rows=[]

def report(name,indices,fallback):
    row={'variant':name,'samples':len(indices),'per_class':{},'fallback_conditions':int(np.sum(fallback))}
    for c in np.unique(labels):
        ix=indices[labels[indices]==c]
        row['per_class'][CLASSES[int(c)]]={
            'n':len(ix),'graph_consistent':float(graph[ix].mean()),'capped':float(capped[ix].mean()),
            'speed_w1_mps':float(wasserstein_distance(speed[ix].ravel(),rspeed[ix].ravel())),
            'acceleration_w1_mps2':float(wasserstein_distance(acc[ix].ravel(),racc[ix].ravel())),
            'travel_w1_m':float(wasserstein_distance(travel[ix],rtravel[ix])),
            'stop_fraction':float(np.mean(speed[ix]<.1)),'real_stop_fraction':float(np.mean(rspeed[ix]<.1)),
            'gaf_teacher_accuracy':float(np.mean(prob['gaf'][42][ix].argmax(-1)==c)),
            'gaf_audit_accuracy':float(np.mean(auditors['gaf'][ix].argmax(-1)==c)),
            'tcn_audit_accuracy':float(np.mean(auditors['tcn'][ix].argmax(-1)==c)),
            'real_gaf_audit_accuracy':float(np.mean(real_auditors['gaf'][ix].argmax(-1)==c)),
            'real_tcn_audit_accuracy':float(np.mean(real_auditors['tcn'][ix].argmax(-1)==c))}
    for key in next(iter(row['per_class'].values())):
        if key!='n':row[key]=float(np.mean([r[key] for r in row['per_class'].values()]))
    row['selected_indices']=indices.tolist()
    return row

rows.append(report('all_candidates',np.arange(len(xy)),[]))
for name,score in [('random',None),('gaf',gaf_score),('tcn',tcn_score),('shuffled_gaf',shuffle_score)]:
    chosen,fallback=select_candidates(groups,valid,score,order)
    rows.append(report(name,chosen,fallback))
    np.savez_compressed(predroot/f'{name}.npz',xy=xy[chosen],labels=labels[chosen],source_index=s['source_index'][chosen],indices=chosen)
np.savez_compressed(predroot/'scores.npz',valid=valid,gaf_teacher=prob['gaf'][42],tcn_teacher=prob['tcn'][42],gaf_audit=auditors['gaf'],tcn_audit=auditors['tcn'])
unique_images=[len({hashlib.sha256(images[i].tobytes()).hexdigest() for i in g}) for g in groups]
pair_dist=[float(np.linalg.norm(xy[g[i]]-xy[g[j]],axis=-1).mean()) for g in groups for i in range(len(g)) for j in range(i)]
# Absolute-coordinate nearest training trajectory at 32 evenly spaced times, diagnostic only.
indices=np.linspace(0,length-1,32).astype(int)
nearest=[]
for x in xy:
    nearest.append(float(np.linalg.norm(training[:,indices]-x[indices][None],axis=-1).mean(-1).min()))
summary={'max_candidates':a.max_candidates,'tag':a.tag,'seed':a.seed,'config':cfg,'candidate_sha256':hashlib.sha256((run/'candidates.npz').read_bytes()).hexdigest(),
    'teachers':provenance,'rows':rows,'conditions':len(groups),'unique_vehicles':len(np.unique(s['source_index'])),
    'valid_candidate_fraction':float(valid.mean()),'encoding_failures':int((~enc_valid).sum()),'coordinate_clipped_points':coord_clip,'channel_clipped_values':channel_clip,
    'identical_gaf_pool_fraction':float(np.mean(np.array(unique_images)==1)),'mean_unique_gaf_images_per_pool':float(np.mean(unique_images)),
    'candidate_pairwise_mean_distance_m':float(np.mean(pair_dist)),'median_nearest_training_track_32point_distance_m':float(np.median(nearest)),
    'validity':'shared: graph-consistent in 5m envelope, no candidate truncation, no route-end cap, GAF encodable; no tuned motion rejection',
    'scope':'Five-class strict moving subset, validation exploration. Teacher seed42, audit seeds43/44; architecture bias remains. All variants same K=8 pool. One selected sample per condition cannot establish retained within-condition diversity. Real references used for evaluation only. Test untouched.'}
(out/'results.json').write_text(json.dumps(summary,indent=2))
(out/'graphs.json').write_text(json.dumps(graphs))
print(json.dumps({'seed':a.seed,'rows':[{k:v for k,v in r.items() if k not in ['per_class','selected_indices']} for r in rows],'identical_gaf_pool_fraction':summary['identical_gaf_pool_fraction']}),flush=True)
