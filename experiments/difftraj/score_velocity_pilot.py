"""Score integrated and directly generated tracks on identical single-class validation requests."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from five_class_evaluate import (MODES, array_digest, features, motion_scores, occupancy,
                                 paper_scores, read_observations, string_digest)
from utils.utils import resample_trajectory


def load_draws(path):
    with np.load(path, allow_pickle=False) as z:
        ids = z['ids'].astype(str)
        xy = np.asarray(z['xy_m'], dtype='f8')
        noise = str(z['noise_sha256'].item())
    if xy.shape != (1000, 200, 2) or len(set(ids.tolist())) != 1000 or not np.isfinite(xy).all():
        raise ValueError(f'Expected 1,000 unique requests and finite metre tracks: {path}')
    if len(noise) != 64 or any(c not in '0123456789abcdef' for c in noise):
        raise ValueError(f'Invalid initial noise fingerprint: {path}')
    return ids, xy, noise


def endpoint_scores(obs, gen, head, norm):
    """Cell hits require being inside the requested physical grid as well as matching the cell."""
    low, high = norm['lo'], norm['hi']
    oe = np.stack([(p[0], p[-1]) for p in obs])
    ge = gen[:, [0, -1]]
    def endpoint_cells(points):
        grid = np.floor((points-low)/(high-low)*16).clip(0,15).astype('i8')
        return 16*grid[:,:,0] + grid[:,:,1]
    requested = head[:, 6:8].astype('i8')
    np.testing.assert_array_equal(endpoint_cells(oe), requested)
    inside = ((ge >= low) & (ge <= high)).all(axis=2)
    hit = (endpoint_cells(ge) == requested) & inside
    err = np.linalg.norm(ge-oe, axis=2)
    # This metric is invariant to translating a whole trajectory, isolating accumulated displacement error.
    drift = np.linalg.norm((ge[:,1]-ge[:,0])-(oe[:,1]-oe[:,0]),axis=1)
    def distance(values):
        return dict(mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)))
    return (dict(start_error_m=distance(err[:,0]),end_error_m=distance(err[:,1]),
                 net_displacement_error_m=distance(drift),start_cell_hit=float(hit[:,0].mean()),
                 end_cell_hit=float(hit[:,1].mean()),joint_cell_hit=float(hit.all(1).mean())),
            err,drift,hit)


def score(data, mode, ids, xy, noise, per_trip=None):
    with np.load(data/'validation/index.npz') as z:
        expected=np.flatnonzero(z['classes']==mode)
        row_of={key:i for i,key in zip(expected,z['ids'][expected].astype(str))}
    if len(row_of)!=1000 or set(ids.tolist())!=set(row_of):
        raise ValueError(f'Expected all and only 1,000 {mode} validation IDs.')
    rows=np.array([row_of[key] for key in ids])
    with np.load(data/'normalization.npz') as z:
        norm=dict(z)
    head=np.load(data/'validation/head.npy',mmap_mode='r')[rows]
    coords=np.load(data/'validation/traj.npy',mmap_mode='r')[rows].astype('f8')*norm['std']+norm['mean']
    raw=read_observations(data,ids)
    duration=np.array([len(path)-1 for path in raw],dtype='f8')
    header_duration=head[:,2].astype('f8')*norm['header_std'][1]+norm['header_mean'][1]
    np.testing.assert_allclose(header_duration,duration,atol=.002,rtol=1.e-6)
    restored=[resample_trajectory(path,int(d)+1) for path,d in zip(xy,duration)]
    obs=occupancy(raw,norm['lo'],norm['hi'])
    syn=occupancy(restored,norm['lo'],norm['hi'])
    train_length=np.load(data/'train/head.npy',mmap_mode='r')[:,1].astype('f8')*norm['header_std'][0]+norm['header_mean'][0]
    bins=np.unique(np.quantile(train_length,np.arange(1,20)/20))
    all_rows=np.arange(len(xy))
    paper=paper_scores(obs,syn,all_rows,all_rows,bins)
    f0,v0=features(coords,duration)
    f1,v1=features(xy,duration)
    motion=motion_scores(f0,f1,v0,v1,all_rows)
    dt=duration[:,None]/199
    a0,a1=[np.diff(v,axis=1)/dt for v in (v0,v1)]
    motion['accel_w1_mps2']=float(wasserstein_distance(a0.ravel(),a1.ravel()))
    endpoints,err,drift,hit=endpoint_scores(raw,xy,head,norm)
    if per_trip:
        with per_trip.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['id','origin_error_m','destination_error_m','displacement_error_m','origin_hit','destination_hit'])
            for key,e,d,valid in zip(ids,err,drift,hit):
                w.writerow([key,*e,d,*valid.astype(int)])
    result=dict(requests=len(xy),mode=mode,split='validation',request_sha256=string_digest(ids),
                head_sha256=array_digest(head),noise_sha256=noise,paper=paper,motion=motion,endpoints=endpoints)
    return result,coords,duration


def plot(net, output, reference, baseline, generated, ids, durations, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from xml.etree import ElementTree

    roads=[]
    for e in ElementTree.parse(net).iter('edge'):
        lane=e.find('lane')
        if lane is not None and e.get('function')!='internal':
            roads.append(np.array([p.split(',')[:2] for p in lane.get('shape').split()],dtype=float)/1000)
    panels=[('SUMO',reference)]
    if baseline is not None:
        panels.append(('Coordinate diffusion',baseline))
    panels.append(('Velocity diffusion',generated))
    # Stable IDs prevent plot selection from changing if draw order changes.
    chosen=np.argsort(ids,kind='stable')[[0,499,999]]
    max_speed=max(np.linalg.norm(np.diff(arr[chosen],axis=1),axis=2).max() for _,arr in panels)
    if not np.isfinite(max_speed):
        raise ValueError('Invalid sample trajectory')
    fig,axes=plt.subplots(3,len(panels),figsize=(len(panels)*3.9,10),constrained_layout=True)
    norm=Normalize(0,25,clip=True)
    for r,i in enumerate(chosen):
        points=np.concatenate([arr[i] for _,arr in panels])/1000
        c=(points.max(0)+points.min(0))/2
        half=max(.25,float(np.ptp(points,axis=0).max())*.55)
        for j,(title,arr) in enumerate(panels):
            ax=axes[r,j]; line=arr[i]/1000
            ax.add_collection(LineCollection(roads,lw=.35,color='#bac4ca',zorder=0))
            segment=LineCollection(np.stack([line[:-1],line[1:]],axis=1),cmap='plasma',norm=norm,lw=1.8)
            segment.set_array(np.linalg.norm(np.diff(arr[i],axis=0),axis=1)/(durations[i]/199))
            ax.add_collection(segment); ax.scatter(*line[0],s=16,fc='white',ec='black',zorder=4)
            ax.scatter(*line[-1],s=22,marker='x',c='black',zorder=4)
            ax.set(xlim=(c[0]-half,c[0]+half),ylim=(c[1]-half,c[1]+half),aspect='equal')
            ax.tick_params(labelsize=7)
            if r==0: ax.set_title(title)
            if j==0: ax.set_ylabel(f'{ids[i]}\nNorth (km)',fontsize=8)
            if r==2: ax.set_xlabel('East (km)')
    fig.suptitle(f'{mode.title()} trips on the MoST network · circle = start, × = end')
    fig.colorbar(segment,ax=axes,location='bottom',shrink=.6,label='Interval speed (m/s, capped at 25)')
    fig.savefig(output,dpi=165)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--draws',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=MODES,default='taxi')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--per-trip',type=Path)
    p.add_argument('--net',type=Path)
    p.add_argument('--map',type=Path)
    args=p.parse_args()
    ids,xy,noise=load_draws(args.draws)
    baseline=None
    if args.baseline:
        b_ids,baseline,b_noise=load_draws(args.baseline)
        if not np.array_equal(ids,b_ids) or noise!=b_noise:
            raise ValueError('Paired baseline requires matching request order and initial noise fingerprint.')
    result,reference,durations=score(args.data,args.mode,ids,xy,noise,args.per_trip)
    if args.map:
        if args.net is None:
            p.error('--net must accompany --map')
        plot(args.net,args.map,reference,baseline,xy,ids,durations,args.mode)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    tmp.replace(args.output)
