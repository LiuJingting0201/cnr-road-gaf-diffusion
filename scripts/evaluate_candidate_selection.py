"""Equal-budget selection audit on one fixed diffusion candidate pool.

No model is trained here; gains are selection gains, never claimed generator gains.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch
from cnr.models import TrajectoryDenoiser
from cnr.road_diffusion import RoadDenoiser, RoadFields
from cnr.representations import RepresentationClassifier
from cnr.roads import RoadNetwork
from cnr.data import CLASSES


def load_classifier(path):
    cfg=json.loads((path/'config.json').read_text())
    m=RepresentationClassifier(cfg['variant'],cfg['stats']).cuda()
    m.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True));m.eval()
    return m


@torch.no_grad()
def probabilities(model,xy):
    return np.concatenate([model(torch.from_numpy(xy[i:i+128]).cuda()).softmax(-1).cpu().numpy()
                           for i in range(0,len(xy),128)])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generator',default='runs/baseline-001/raw_diffusion')
    p.add_argument('--study',default='runs/representation-r1')
    p.add_argument('--delta-study',default=None,help='Optional same-input delta TCN study directory')
    p.add_argument('--fields',default='data/maps/road_fields_4m.npz')
    p.add_argument('--ddim-steps',type=int,default=None)
    p.add_argument('--data',default='data/processed/most64')
    p.add_argument('--map',default='data/maps/most.net.xml')
    p.add_argument('--output',default='runs/candidate-selection-g1')
    p.add_argument('--per-class-vehicles',type=int,default=8)
    p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--repetitions',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    data=Path(a.data);val=np.load(data/'val.npz');meta=json.loads((data/'dataset.json').read_text())
    frame=pd.read_csv(data/'manifest.csv',dtype={'vehicle_id':str});frame=frame[frame.split=='val']
    rng=np.random.default_rng(a.seed)
    per_class=min(a.per_class_vehicles,min(frame[frame['class']==c].vehicle_id.nunique() for c in CLASSES))
    if per_class<2:raise ValueError('Too few independent validation vehicles')
    selected=[];vehicles=[]
    for name in CLASSES:
        candidates=sorted(frame[frame['class']==name].vehicle_id.unique())
        chosen=rng.choice(candidates,per_class,replace=False)
        for vehicle in chosen:
            selected.append(int(rng.choice(frame[frame.vehicle_id==vehicle].array_index.to_numpy())))
            vehicles.append(vehicle)
    selected=np.array(selected);starts=val['xy'][selected,0];labels=val['labels'][selected]
    c=len(labels);r=a.repetitions;k=a.candidates
    lab=np.repeat(labels,r*k);start=np.repeat(starts,r*k,axis=0)
    gpath=Path(a.generator);gconfig=json.loads((gpath/'config.json').read_text())
    is_relative='relative_mean' in gconfig
    model=(RoadDenoiser(use_map=gconfig['variant']!='relative') if is_relative else TrajectoryDenoiser()).cuda()
    state=torch.load(gpath/'latest.pt',map_location='cuda',weights_only=True)
    model.load_state_dict(state['ema' if is_relative else 'model']);model.eval()
    fields=RoadFields(a.fields).cuda() if is_relative and model.use_map else None
    mean=torch.tensor(meta['mean'],device='cuda',dtype=torch.float32)
    std=torch.tensor(meta['std'],device='cuda',dtype=torch.float32)
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    ddim_steps=a.ddim_steps or (100 if is_relative else 50)
    if not 2<=ddim_steps<=1000:raise ValueError('DDIM steps must be in [2,1000]')
    times=torch.linspace(999,0,ddim_steps).long().tolist();tracks=[]
    if is_relative:
        rm=torch.tensor(gconfig['relative_mean'],device='cuda',dtype=torch.float32)
        rs=torch.tensor(gconfig['relative_std'],device='cuda',dtype=torch.float32)
    # Loading classifiers/models must not change the sampling RNG stream.
    torch.manual_seed(a.seed)
    with torch.no_grad():
        for base in range(0,len(lab),128):
            ss=torch.from_numpy(start[base:base+128]).cuda();yy=torch.from_numpy(lab[base:base+128]).cuda()
            zz=(ss-mean)/std;x=torch.randn(len(ss),2,meta['length']-1,device='cuda')
            patch=fields.patches(ss,yy) if fields is not None else None
            for j,t in enumerate(times):
                args=(x,torch.full((len(ss),),t,device='cuda'),yy,zz)
                eps=model(*args,patch) if is_relative else model(*args)
                x0=(x-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
                prev=alpha[times[j+1]] if j+1<len(times) else torch.tensor(1.,device='cuda')
                x=prev.sqrt()*x0+(1-prev).sqrt()*eps
            future=x.transpose(1,2)*rs+rm+ss[:,None] if is_relative else x.transpose(1,2)*std+mean
            tracks.append(torch.cat((ss[:,None],future),1).cpu().numpy())
    xy=np.concatenate(tracks)
    if not np.isfinite(xy).all():raise RuntimeError('Nonfinite samples')
    np.savez_compressed(out/'candidate_pool.npz',xy=xy.reshape(c,r,k,meta['length'],2),labels=labels,validation_indices=selected)
    roads=RoadNetwork(a.map);dist=np.empty(xy.shape[:2],dtype=np.float64)
    for i,name in enumerate(CLASSES):
        mask=lab==i;dist[mask]=roads.distances(xy[mask],name).reshape(-1,xy.shape[1])
    # Exclude the supplied start from generated-point quality metrics.
    roadscore=dist[:,1:].mean(1).reshape(c,r,k)
    scores={};source_hashes={}
    selectors=[('hilbert',Path(a.study)/'hilbert_fine_rgb-seed42'),
               ('delta_gaf',Path(a.study)/'delta_gaf-seed42'),
               ('sequence',Path(a.study)/'xy_motion_tcn-seed42')]
    if a.delta_study:selectors.append(('delta_tcn',Path(a.delta_study)/'delta_tcn-seed42'))
    for key,path in selectors:
        m=load_classifier(path)
        pr=probabilities(m,xy);scores[key]=pr[np.arange(len(lab)),lab].reshape(c,r,k)
        source_hashes[key]=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest();del m
    audit=[]
    for seed in [43,44]:
        path=Path(a.study)/f'xy_motion_tcn-seed{seed}';m=load_classifier(path)
        audit.append(probabilities(m,xy));del m
    auditprob=np.mean(audit,axis=0);auditpred=auditprob.argmax(-1)
    choices={'random':np.zeros((c,r),dtype=int),'road':roadscore.argmin(-1)}
    top_road=np.argsort(roadscore,axis=-1)[...,:max(1,k//4)]
    for key,score in scores.items():
        choices[key]=score.argmax(-1)
        shortlist=np.take_along_axis(score,top_road,axis=-1).argmax(-1)
        choices['road_'+key]=np.take_along_axis(top_road,shortlist[...,None],axis=-1)[...,0]
    base=np.arange(c*r).reshape(c,r)*k
    stats={};condition_stats={}
    for policy,choice in choices.items():
        ix=(base+choice).reshape(-1);yy=lab[ix];xx=xy[ix];dd=dist[ix,1:]
        per_class_stats={}
        for label,name in enumerate(CLASSES):
            mask=yy==label;generated_speed=np.linalg.norm(np.diff(xx[mask],axis=1),axis=-1)/.25
            real_speed=np.linalg.norm(np.diff(val['xy'][val['labels']==label],axis=1),axis=-1)/.25
            per_class_stats[name]={'audit_target_accuracy':float(np.mean(auditpred[ix][mask]==label)),
                 'road_within_5m':float(np.mean(dd[mask]<=5)), 'road_distance_mean_m':float(dd[mask].mean()),
                 'speed_wasserstein_mps':float(wasserstein_distance(generated_speed.ravel(),real_speed.ravel())),
                 'speed_p95_mps':float(np.quantile(generated_speed,.95))}
        shaped=xx.reshape(c,r,meta['length'],2)
        differences=[np.linalg.norm(shaped[:,i,1:]-shaped[:,j,1:],axis=-1).mean(1)
                     for i in range(r) for j in range(i+1,r)]
        stats[policy]={'samples':len(ix),'road_within_5m':float(np.mean(dd<=5)),
            'road_distance_mean_m':float(dd.mean()),'audit_macro_target_accuracy':float(np.mean(auditpred[ix]==yy)),
            'mean_class_speed_wasserstein_mps':float(np.mean([v['speed_wasserstein_mps'] for v in per_class_stats.values()])),
            'within_condition_mean_pairwise_distance_m':float(np.mean(differences)) if differences else None,
            'per_class':per_class_stats}
        condition_stats[policy]={'road_within_5m':(dd<=5).reshape(c,r,-1).mean((1,2)),
            'audit_target_accuracy':(auditpred[ix]==yy).reshape(c,r).mean(1),
            'road_distance_mean_m':dd.reshape(c,r,-1).mean((1,2))}
        np.savez_compressed(out/f'selected_{policy}.npz',xy=xx,labels=yy,candidate_indices=ix)
    # Paired stratified bootstrap over independent vehicle conditions, not individual points.
    brng=np.random.default_rng(9326)
    resample=np.concatenate([brng.choice(np.flatnonzero(labels==label),(2000,per_class),replace=True)
                             for label in range(6)],axis=1)
    comparisons={}
    pairs=[('delta_gaf','random'),('road_delta_gaf','road')]
    if a.delta_study:pairs += [('delta_gaf','delta_tcn'),('road_delta_gaf','road_delta_tcn')]
    for left,right in pairs:
        comparisons[left+' minus '+right]={}
        for metric in condition_stats[left]:
            delta=condition_stats[left][metric]-condition_stats[right][metric]
            lo,hi=np.quantile(delta[resample].mean(1),[.025,.975])
            comparisons[left+' minus '+right][metric]={'difference':float(delta.mean()),'ci95':[float(lo),float(hi)]}
    report={'protocol':vars(a),'status':'complete','scope':'validation-only fixed-pool selection audit; no generator modification',
        'sampler':{'coordinates':'relative' if is_relative else 'absolute','weights':'ema' if is_relative else 'model','ddim_steps':ddim_steps},
        'conditions':c,'independent_vehicles_per_class':per_class,'candidates_total':len(xy),
        'selected_fraction':1/k,'road_then_semantic':'top quarter by mean permitted-lane distance, then target probability',
        'auditor':'ensemble of raw XY/motion TCN seeds 43 and 44; selectors use seed42',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'generator_sha256':hashlib.sha256((gpath/'latest.pt').read_bytes()).hexdigest(),
        'map_sha256':hashlib.sha256(Path(a.map).read_bytes()).hexdigest(),
        'selector_hashes':source_hashes,'policies':stats,'paired_comparisons':comparisons,
        'bootstrap':'2000 paired stratified resamples of vehicle conditions; exploratory, no multiplicity correction',
        'elapsed_s':time.monotonic()-started,
        'limitations':['no topology or directional validation','no test evaluation','classifier scores on generated samples may be unreliable',
                      'sequence selector and auditors share architecture but not parameters',
                      'pairwise distance is dispersion, not proof of plausible diversity','one generator seed; diagnostic only']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'conditions.json').write_text(json.dumps({'vehicle_ids':vehicles,'validation_indices':selected.tolist()}))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
