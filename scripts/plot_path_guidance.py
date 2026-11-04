"""Fixed first condition per class, all repeated draws; no example selection by quality."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from cnr.roads import RoadNetwork
from cnr.data import CLASSES
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
variants=['none','gaf_0.3','gaf_3'];samples={v:np.load(root/f'{v}.npz') for v in variants}
base=samples['none'];classes=np.unique(base['labels']);roads=RoadNetwork('data/maps/most.net.xml')
fig,axes=plt.subplots(len(classes),3,figsize=(13,4*len(classes)),squeeze=False)
for row,c in enumerate(classes):
 source=base['source_index'][np.flatnonzero(base['labels']==c)[0]];ix=np.flatnonzero(base['source_index']==source)
 allxy=np.concatenate([samples[v]['xy'][ix].reshape(-1,2) for v in variants]);lo=allxy.min(0)-15;hi=allxy.max(0)+15
 segments=[]
 for shape in roads.shapes:
  x0,y0,x1,y1=shape.bounds
  if x1>=lo[0] and x0<=hi[0] and y1>=lo[1] and y0<=hi[1]:segments.append(np.asarray(shape.coords))
 for col,v in enumerate(variants):
  ax=axes[row,col];ax.add_collection(LineCollection(segments,colors='#c3cad2',linewidths=.6))
  for track in samples[v]['xy'][ix]:ax.plot(track[:,0],track[:,1],lw=1.2,alpha=.8)
  ax.scatter(allxy[0,0],allxy[0,1],color='black',s=15);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=f'{CLASSES[c]} | {v} | source {source}',xlabel='local x (m)',ylabel='local y (m)');ax.set_aspect('equal')
fig.suptitle('First condition per class; all four draws shown; same route and initial noise across columns')
fig.tight_layout(rect=(0,0,1,.98));out=Path('docs/results')/root.parent.name/root.name;out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fixed_examples.png',dpi=150);fig.savefig(out/'fixed_examples.svg');plt.close(fig)
