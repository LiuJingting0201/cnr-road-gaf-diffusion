"""Predetermined examples: first condition of each class, repetition zero.

Real validation trajectories are contextual examples, not prediction targets.
Plot all six classes; do not select cases by quality.
"""
import json
from pathlib import Path
import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from cnr.data import CLASSES
from cnr.roads import RoadNetwork

root=Path('runs/candidate-selection-g3');report=json.loads((root/'report.json').read_text())
meta=json.loads((root/'conditions.json').read_text());real=np.load('data/processed/most64/val.npz')
policies=['random','road','road_delta_gaf'];samples={p:np.load(root/f'selected_{p}.npz')['xy'] for p in policies}
roads=RoadNetwork('data/maps/most.net.xml');perclass=report['independent_vehicles_per_class'];repeats=report['protocol']['repetitions']
out=Path('docs/results/candidate-selection-g3');out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(6,3,figsize=(12,18));records=[]
for label,name in enumerate(CLASSES):
    condition=label*perclass;idx=condition*repeats
    ref=real['xy'][meta['validation_indices'][condition]]
    tracks=[samples[p][idx] for p in policies]
    allxy=np.concatenate([ref]+tracks);lo=allxy.min(0)-25;hi=allxy.max(0)+25
    center=(lo+hi)/2;radius=max((hi-lo).max()/2,40);lo=center-radius;hi=center+radius
    roads.distances(ref[:1],name);tree=roads.trees[name]
    nearby=tree.geometries[tree.query(shapely.box(*lo,*hi))]
    segments=[np.asarray(line.coords) for line in nearby]
    records.append({'class':name,'condition':condition,'repetition':0,'vehicle_id':meta['vehicle_ids'][condition],
                    'validation_index':meta['validation_indices'][condition]})
    for col,(policy,track) in enumerate(zip(policies,tracks)):
        ax=axes[label,col];ax.add_collection(LineCollection(segments,colors='#c8cdd1',linewidths=.65))
        ax.plot(ref[:,0],ref[:,1],'--',color='#2b343b',lw=1.2,label='Real context')
        ax.plot(track[:,0],track[:,1],color='#b04b21' if 'gaf' in policy else '#247ba5',lw=1.8,label='Generated')
        ax.scatter(*ref[0],s=30,color='#1c8b4d',zorder=5,label='Given start')
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',title=f'{name} | {policy}')
        ax.tick_params(labelsize=7)
        if label==5:ax.set_xlabel('SUMO x (m)')
        if col==0:ax.set_ylabel('SUMO y (m)')
axes[0,0].legend(fontsize=7,loc='best')
fig.suptitle('G3 fixed examples: same start, same candidate budget\nGray: permitted lane centerlines; dashed: real context, not a future target',fontsize=12)
fig.tight_layout(rect=(0,0,1,.965))
fig.savefig(out/'map_examples.png',dpi=160);fig.savefig(out/'map_examples.svg');plt.close(fig)
(out/'map_examples.json').write_text(json.dumps({'selection':'first condition per class, repetition zero, fixed before inspecting plots','examples':records},indent=2))
