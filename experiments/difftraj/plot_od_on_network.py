"""Overlay unmodified samples and OD request cells on native SUMO road geometry."""
import gzip,sys
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
src,out=map(Path,sys.argv[1:])
d=np.load(src/'map.npz'); old=np.load(src/'42-destination.npz')
for key in ('before','after'):
    assert np.allclose(d[key][:,-1],old[key][:3],atol=.02,rtol=0)
roads=[]
with gzip.open(src/'net.xml.gz') as f:
    for _,edge in ET.iterparse(f,events=['end']):
        if edge.tag!='edge':continue
        if not edge.get('function'):
            lanes=edge.findall('lane')
            if lanes:
                roads.append(np.array([tuple(map(float,t.split(',')[:2])) for t in lanes[0].get('shape').split()])/1000)
        edge.clear()
colors=['#165f9c','#c44d1d'];w=d['width']/1000
for i in range(3):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    pts=np.concatenate([d[k][i].reshape(-1,2) for k in ('before','after','reference','O','A','B')])/1000
    low=pts.min(0)-w*.8;high=pts.max(0)+w*.8
    for ax,mode,key,col,title in zip(axes,['before','after'],['A','B'],colors,['Original destination','Switched destination']):
        ax.add_collection(LineCollection(roads,colors='#c1c7ca',linewidths=.55,zorder=0))
        for tag,color in [('O','#427c51'),(key,col)]:
            c=d[tag][i]/1000
            ax.add_patch(Rectangle(c-w/2,*w,facecolor=color,alpha=.16))
            ax.add_patch(Rectangle(c-w/2,*w,fill=False,edgecolor=color,lw=1.6))
            ax.annotate(tag,(c[0]-w[0]/2,c[1]+w[1]/2),fontsize=9,color=color,weight='bold')
        if mode=='before':
            real=d['reference'][i]/1000;ax.plot(*real.T,'--',color='#30363a',lw=1.2)
        g=d[mode][i]/1000;ax.plot(*g.T,color=col,lw=1.6)
        ax.scatter(*g[0],facecolors='white',edgecolors=col,s=45,zorder=5);ax.scatter(*g[-1],marker='X',c=col,s=66,zorder=5)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='Local east (km)',title=title,aspect='equal')
    axes[0].set_ylabel('Local north (km)')
    fig.suptitle(f'Monaco MoST taxi network  |  Destination switch {i+1}',weight='bold')
    fig.text(.5,.025,'Grey: SUMO roads     Dashed: original observed trip     O: fixed origin cell     A / B: requested destination     Circle / cross: generated start / end',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(out/f'map-{i+1}.png',dpi=180);plt.close(fig)
