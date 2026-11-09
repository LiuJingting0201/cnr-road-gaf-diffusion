"""Fixed-order complete journeys and original GAF views from completed synthesis.

No score-based example selection. Images are recomputed from generated outputs,
not target GAF inputs used to generate those outputs. Raw tracks stay server-side.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from g33_evaluate import read_pool, signals
from g33_train import digest, save
from g33_model import CLASSES
from g40_model import ARMS
from g35c_teachers import speed_profile
from g36_feedback import spatial_sequence
from g28_teachers import image_inputs
from cnr.original_gaf import encode_frozen

PLOTTED=('B','J-S','J-V','J-SV','J-RAW')
NAMES=('Bus','Motorcycle','Private car','Bicycle','Taxi / ride-hailing')
COLORS={'SUMO':'#232830','B':'#77838a','J-S':'#9459a8','J-V':'#178879','J-SV':'#d76535','J-RAW':'#5575b4'}


def main(a):
    assert json.loads((a.root/'all-stages-complete.json').read_text())['complete']
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    episode=a.root/'400201';seed=42;sources={};pools={};rows={};lookup={};figures=[]
    for arm in ARMS:
        folder=episode/'generated'/f'{arm}-{seed}';r=json.loads((folder/'complete.json').read_text())
        assert r['complete'] and not r['true_future_GAF_opened']
        for n in ('samples.npz','candidates.json'):assert digest(folder/n)==r['artifacts'][n]
        with np.load(folder/'samples.npz',allow_pickle=False) as f:pools[arm]={k:f[k] for k in f.files}
        rows[arm]=json.loads((folder/'candidates.json').read_text())['rows']
        lookup[arm]={r['request_id']:r['index'] for r in rows[arm] if r['candidate']==0}
        sources[str(folder/'complete.json')]=digest(folder/'complete.json')
    _,reference=read_pool(episode/'reference','references.npz');refs={str(x):i for i,x in enumerate(reference['ids'])}
    sources[str(episode/'reference/complete.json')]=digest(episode/'reference/complete.json')
    common=set(refs).intersection(*(set(x) for x in lookup.values()))
    meta=json.loads((a.cache/'complete.json').read_text())['spatial']['frozen_parameters']
    sources[str(a.cache/'complete.json')]=digest(a.cache/'complete.json');sources[str(a.network)]=digest(a.network)
    lanes=[]
    for _,node in ET.iterparse(a.network,events=('end',)):
        if node.tag=='lane' and node.get('shape'):
            lanes.append(np.array([[float(x) for x in p.split(',')[:2]] for p in node.get('shape').split()]))
        node.clear()
    def track(pool,index):
        lo,hi=pool['offsets'][index:index+2];v=pool['txy'][lo:hi].copy();v[:,0]-=v[0,0];return v
    def output(fig,name):
        for ext in ('pdf','png'):fig.savefig(a.output/(name+'.'+ext),bbox_inches='tight',dpi=160)
        plt.close(fig);figures.append(name)
    def images(txy):
        seq,clipped=spatial_sequence(txy[:,1:],meta)
        spatial,_=encode_frozen(seq[None],np.asarray(meta['channel_min']),np.asarray(meta['channel_max']))
        speed,_=speed_profile(txy[:,0],txy[:,1:])
        with torch.no_grad():
            velocity=image_inputs('speed',speed[None])
            velocity=velocity*velocity.new_tensor([.229,.224,.225])[None,:,None,None]+velocity.new_tensor([.485,.456,.406])[None,:,None,None]
        return spatial[0],velocity[0].permute(1,2,0).cpu().numpy().clip(0,1),seq,speed,clipped
    cases=[]
    for label,cls in enumerate(CLASSES):
        eligible=sorted(i for i in common if int(reference['labels'][refs[i]])==label);assert eligible
        ident=eligible[0]
        tracks={'SUMO':track(reference,refs[ident]),**{arm:track(pools[arm],lookup[arm][ident]) for arm in PLOTTED}}
        allxy=np.concatenate([v[:,1:] for v in tracks.values()]);origin=allxy.min(0);span=np.maximum(allxy.max(0)-origin,1)
        pad=max(60,.08*span.max());lo=origin-pad;hi=allxy.max(0)+pad
        nearby=[p-origin for p in lanes if np.all(p.max(0)>=lo) and np.all(p.min(0)<=hi)]
        fig,axs=plt.subplots(2,3,figsize=(10.5,7))
        for ax,key in zip(axs[0],('SUMO','B','J-SV')):
            t=tracks[key];ax.add_collection(LineCollection(nearby,colors='#dce1e5',linewidths=.35))
            points=ax.scatter(t[:,1]-origin[0],t[:,2]-origin[1],c=t[:,0]/t[-1,0],s=4,cmap='viridis',vmin=0,vmax=1,rasterized=True)
            ax.scatter(*(t[0,1:]-origin),marker='o',s=28,facecolor='white',edgecolor='black',zorder=5)
            ax.scatter(*(t[-1,1:]-origin),marker='x',s=30,c='black',zorder=5)
            ax.set(xlim=(-pad,span[0]+pad),ylim=(-pad,span[1]+pad),xlabel='Local east (m)',ylabel='Local north (m)',title=key)
            ax.set_aspect('equal',adjustable='box')
        for key,t in tracks.items():
            _,speed,accel,_,_,arc=signals(t[:,0],t[:,1:]);kw={'c':COLORS[key],'lw':1.4 if key=='SUMO' else .9,'ls':'--' if key=='SUMO' else '-'}
            axs[1,0].plot(t[:,0],arc/arc[-1],label=key,**kw)
            axs[1,1].plot(np.arange(len(speed))+.5,speed,**kw)
            # A distribution view avoids false framewise pairing of different trips.
            if len(accel):
                values=np.sort(accel);axs[1,2].plot(values,np.arange(1,len(values)+1)/len(values),**kw)
        axs[1,0].set(title='Whole-trip progress',xlabel='Elapsed time (s)',ylabel='Fraction of distance')
        axs[1,1].set(title='Whole-trip speed',xlabel='Elapsed time (s)',ylabel='One-second speed (m/s)')
        axs[1,2].set(title='Acceleration distribution',xlabel='One-second acceleration (m/s²)',ylabel='Empirical CDF')
        for ax in axs[1]:ax.grid(alpha=.2)
        handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=6,loc='lower center',frameon=False)
        fig.suptitle(NAMES[label]+': first direct draws on a new request',fontsize=14)
        fig.subplots_adjust(top=.91,bottom=.12,wspace=.36,hspace=.5)
        output(fig,'g40-journey-'+cls)
        rendered={key:images(t) for key,t in tracks.items()}
        fig,axs=plt.subplots(6,3,figsize=(7.8,11.2),gridspec_kw={'width_ratios':[1,1,1.55]})
        ymax=max(35,max(x[3].max() for x in rendered.values())*1.05);end=max(t[-1,0] for t in tracks.values())
        record={'class':cls,'request_sha256':hashlib.sha256(ident.encode()).hexdigest(),'eligible_requests':len(eligible),'tracks':{}}
        for row,(key,t) in enumerate(tracks.items()):
            sp,ve,seq,speed,clipped=rendered[key]
            axs[row,0].imshow(sp,interpolation='nearest');axs[row,1].imshow(ve,interpolation='nearest')
            for ax in axs[row,:2]:ax.set_xticks([]);ax.set_yticks([])
            axs[row,0].set_ylabel(key)
            _,v,_,_,_,_=signals(t[:,0],t[:,1:]);axs[row,2].plot(np.arange(len(v))+.5,v,color=COLORS[key],lw=.8)
            axs[row,2].set(xlim=(0,end),ylim=(0,ymax),ylabel='m/s');axs[row,2].grid(alpha=.15)
            if row==0:
                for ax,title in zip(axs[row],('Spatial Hilbert\nimage','Speed\nimage','Corresponding\nmotion')):ax.set_title(title)
            if row==5:axs[row,2].set_xlabel('Elapsed time (s)')
            record['tracks'][key]={'duration_s':float(t[-1,0]),'points':len(t),'track_sha256':hashlib.sha256(t.tobytes()).hexdigest(),
                'spatial_image_sha256':hashlib.sha256(sp.tobytes()).hexdigest(),'speed_image_sha256':hashlib.sha256(ve.tobytes()).hexdigest(),
                'coordinate_clipped':clipped}
        fig.suptitle(NAMES[label]+': GAF views reconstructed from the outputs',fontsize=13)
        fig.tight_layout(rect=(0,.01,1,.97),h_pad=1.1);output(fig,'g40-gaf-'+cls);cases.append(record)
    save(a.output/'complete.json',{'complete':True,'rule':'400201,seed42,first lexicographic request per class jointly supported by reference and all eight arms; candidate0 only',
        'sources':sources,'source_sha256':digest(__file__),'cases':cases,'figures':figures,
        'scope':'Descriptive fixed-order examples, no scores, no favorable re-selection, no new fitting or generation.',
        'images':'Original frozen spatial transform and unchanged speed transform; output visualization, not future conditions fed into synthesis.',
        'artifacts':{p.name:digest(p) for p in a.output.iterdir() if p.suffix in ('.pdf','.png')}})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','cache','network','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
