"""Compare two speed clocks on the identical frozen route for every trip and label switch."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from five_class_train import make_model
from five_class_evaluate import MODES, array_digest, features, occupancy, read_observations, sample, string_digest
from pooled_velocity_evaluate import scores
from retime_difftraj_paths import retime
from route_clock_features import DIM, describe
from velocity_difftraj import decode
from velocity_label_balance import coverage
from velocity_label_contrasts import SIGNATURE, motion_signature


def create_maps(out, seed, selected, labels, ids, raw, duration, clocks, names=("Original clock","Route-conditioned clock"),
                legend="original (orange), route-conditioned (blue)"):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from plot_supervisor_sumo_maps import map_shapes
    net=Path('/workspace/cnr-experiments/gaf-inloop-v1/g27-job/g27-data/kit-check/assets/most.net.xml')
    roads=map_shapes(net)
    fig,axes=plt.subplots(len(selected),4,figsize=(13,2.55*len(selected)))
    for i,r in enumerate(selected):
        dt=duration[r]/199
        t=np.linspace(0,duration[r],200)
        obs=np.stack([np.interp(t,np.arange(len(raw[r])),c) for c in raw[r].T],axis=1)
        paths=[obs,*[p[r] for p in clocks]]
        vs=[np.linalg.norm(np.diff(p,axis=0),axis=1)/dt for p in paths]
        color=Normalize(0,np.ceil(max(v.max() for v in vs)))
        bounds=np.concatenate(paths)/1000
        center=(bounds.min(0)+bounds.max(0))/2
        radius=max(.15,.55*np.ptp(bounds,axis=0).max())
        for ax,xy,v,title in zip(axes[i,:3],paths,vs,['SUMO',*names]):
            xy=xy/1000
            ax.add_collection(LineCollection(roads,colors='#d9dce0',linewidths=.45))
            line=LineCollection(np.stack([xy[:-1],xy[1:]],axis=1),cmap='viridis',norm=color,linewidths=2)
            line.set_array(v)
            ax.add_collection(line)
            ax.plot(*xy[0],'o',mec='black',mfc='white',ms=4)
            ax.plot(*xy[-1],'x',color='black',ms=5)
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),aspect='equal')
            ax.set_axis_off()
            ax.set_title(title+' · '+MODES[labels[r]],fontsize=9)
        ax=axes[i,3]
        for v,c in zip(vs,['#42494d','#a65020','#127b9a']):
            ax.plot(t[1:],v,color=c,linewidth=.8)
        ax.set(title=str(ids[r]),xlabel='Time (s)',ylabel='Speed (m/s)')
        ax.title.set_fontsize(9)
        bar=fig.colorbar(line,ax=axes[i,2],fraction=.04,pad=.015)
        bar.ax.tick_params(labelsize=7)
    fig.suptitle(f'Fixed generated route and noise · seed {seed}\nCircle = start, cross = end. Map colour = m/s. Speed curves: SUMO (grey), {legend}.',fontsize=11)
    fig.subplots_adjust(top=.93,wspace=.25,hspace=.50)
    fig.savefig(out/f'map-{seed}.png',dpi=155,bbox_inches='tight')
    plt.close(fig)


def evaluate(a):
    data,run,seed=a.data,a.run,a.seed
    old=a.base
    route_seed=a.route_seed or seed
    norm=dict(np.load(run/'prepared/normalization.npz'))
    ck=torch.load(run/f'seed-{seed}/final.pt',map_location='cpu',weights_only=True)
    assert ck['seed']==seed and ck['epoch']==10 and ck['arm']=='C'
    with np.load(old/f'coordinate-{route_seed}/eval/draw-cache.npz') as cr, np.load(old/f'velocity-{seed}/eval/draw-cache.npz') as vr:
        for key in ('ids','validation_rows','draw_rows','draw_labels','keys','true_labels','noise_sha256','head_sha256'):
            np.testing.assert_array_equal(cr[key],vr[key])
        ids,rows,actual=cr['ids'],cr['validation_rows'],cr['true_labels']
        n=len(ids)
        route=cr['xy_m'][:n].astype('d')
        keys=vr['keys']
        original=vr['xy_m'].astype('d')
        np.testing.assert_array_equal(cr['draw_rows'][:n],np.arange(n))
        np.testing.assert_array_equal(cr['draw_labels'][:n],actual)
        noise=np.random.default_rng(42).standard_normal((n,2,200),dtype='f4')
        h=np.load(data/'validation/head.npy')[rows]
        assert array_digest(noise)==cr['noise_sha256'].item() and array_digest(h)==cr['head_sha256'].item()
    matched,info=coverage(h,h[:,4].astype('d')*norm['header_std'][3]+norm['header_mean'][3],actual,ids)
    bike,bus=MODES.index('bicycle'),MODES.index('bus')
    left,right=matched[bike,bus]
    assert len(left)==53 and n==3000
    # For the intervention use the original request's route even after changing class.
    slots=np.r_[np.arange(n),left,right]
    targets=np.r_[actual,np.full(len(left),bus),np.full(len(right),bike)]
    lookup=dict(zip(keys,np.arange(len(keys))))
    old_vel=original[[lookup[r*len(MODES)+c] for r,c in zip(slots,targets)]]
    scale=dict(np.load(run/f'features-{seed}/scale.npz'))
    bends=(describe(route)-scale['mean'])/scale['sd']
    conditions=np.column_stack([h,bends]).astype('f4')
    model=make_model('C',route_dim=DIM).eval()
    model.load_state_dict(ck['weights'])
    tokens=sample(model,conditions[slots],targets,noise[slots],{'mean':np.zeros(2),'std':np.ones(2)},64)
    new_vel=decode(tokens,h[slots],norm,norm['velocity_token_mean'],norm['velocity_token_std']).astype('f4').astype('d')
    raw=read_observations(data,ids)
    duration=np.array([len(p)-1 for p in raw])
    ref=np.load(data/'validation/traj.npy',mmap_mode='r')[rows]*norm['std']+norm['mean']
    occ=occupancy(raw,norm['lo'],norm['hi'])
    tr=np.load(data/'train/head.npy',mmap_mode='r')[:,1].astype('d')*norm['header_std'][0]+norm['header_mean'][0]
    bins=np.unique(np.quantile(tr,np.arange(1,20)/20))
    reference=features(ref,duration)
    obs=motion_signature(ref,duration)
    report=dict(seed=seed,route_seed=route_seed,n=n,route_sha256=array_digest(route),noise_sha256=array_digest(noise),requests_sha256=string_digest(ids),
                pairs=info['bus_minus_bicycle'],observed_gap=dict(zip(SIGNATURE,(obs[right]-obs[left]).mean(0))),arms={})
    clocks=[]
    real_speeds=reference[1]
    for arm,vel in [('baseline',old_vel),('conditioned',new_vel)]:
        draws,factors=zip(*(retime(route[r],v) for r,v in zip(slots,vel)))
        xy=np.stack(draws)
        np.testing.assert_array_equal(xy[:,[0,-1]],route[slots][:,[0,-1]])
        classes,macro=scores(raw,reference,occ,bins,xy[:n],h,actual,duration,norm)
        summary=motion_signature(xy,duration[slots])
        m=len(left)
        diff=((summary[n:n+m]-summary[left])+(summary[right]-summary[n+m:]))/2
        report['arms'][arm]=dict(macro=macro,per_class=classes,label_effect=dict(zip(SIGNATURE,diff.mean(0))),
            median_absolute_length_rescaling=float(np.median(np.abs(np.asarray(factors[:n])-1))))
        generated_speeds=features(xy[:n],duration)[1]
        trip_w1=np.abs(np.sort(generated_speeds,axis=1)-np.sort(real_speeds,axis=1)).mean(1)
        mean_bias=[abs(generated_speeds[actual==k].mean()-real_speeds[actual==k].mean()) for k in range(len(MODES))]
        report['arms'][arm].update(mean_paired_trip_speed_w1_mps=float(trip_w1.mean()),
            macro_speed_w1_mean_bias_lower_bound_mps=float(np.mean(mean_bias)))
        clocks.append(xy[:n])
    reportdir=run/'report'
    reportdir.mkdir(exist_ok=True)
    if route_seed != seed:
        reportdir=reportdir/f'common-routes-{route_seed}'
        reportdir.mkdir(exist_ok=True)
    create_maps(reportdir,seed,[np.flatnonzero(actual==i)[0] for i in range(len(MODES))],actual,ids,raw,duration,clocks)
    (reportdir/f'metrics-{seed}.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','base','run'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--route-seed',type=int,choices=(42,43))
    evaluate(p.parse_args())
