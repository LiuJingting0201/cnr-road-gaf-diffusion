"""Align ordered, scheduled SUMO platforms to matched observed and generated trajectories."""
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from bus_low_speed_spells import root,data,net,paths,ids,dt
from plot_supervisor_sumo_maps import map_shapes

src=root/'gaf-inloop-v1/g30-job/g30-data/source/scenario/in'
lanes={n.get('id'):n for n in ET.parse(net).iter('lane')}
platforms={}
for stop in ET.parse(src/'add/most.busstops.add.xml').iter('busStop'):
    lane=lanes[stop.get('lane')]
    xy=np.array([z.split(',')[:2] for z in lane.get('shape').split()],float)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    length=float(lane.get('length'))
    ends=np.array([float(stop.get(k)) for k in ('startPos','endPos')])
    ends=np.clip(np.where(ends<0,length+ends,ends),0,length)*arc[-1]/length
    offsets=np.linspace(*ends,max(2,int(np.ceil(ends[1]-ends[0]))+1))
    samples=np.stack([np.interp(offsets,arc,col) for col in xy.T],axis=1)
    platforms[stop.get('id')]=(samples.mean(0),cKDTree(samples))
by_episode=defaultdict(list)
for i,key in enumerate(ids): by_episode[key.split('_')[0]].append((key,i))
schedules={}
for episode,items in by_episode.items():
    tree=ET.parse(data/'raw_shards'/episode/'routes.xml')
    routes={x.get('id'):[s.get('busStop') for s in x.findall('stop')] for x in tree.findall('route')}
    assign={x.get('id'):x.get('route') for x in tree.findall('vehicle')}
    for k,i in items:
        rid=assign[k]
        schedules[i]=(rid,routes[rid])
        assert schedules[i][1] and all(z in platforms for z in schedules[i][1])

def runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def visit_candidates(dist,slow,radius):
    candidates=[]
    for a,b in runs(dist<radius):
        spans=runs(slow[a:b])
        dwell=max((v-u for u,v in spans),default=0)
        candidates.append((a,b,dwell,float(dist[a:b].min())))
    return candidates

def align(distances,slow,radius):
    # Ordered assignments with disjoint time intervals. Maximise approached stops,
    # then recognised slow visits, then lowest distance. A skipped stop is allowed.
    states={0:((0,0,0.),[])}
    for dist in distances:
        options=visit_candidates(dist,slow,radius)
        nxt={}
        for time,(score,path) in states.items():
            for item in [None,*options]:
                if item is None:
                    end,new=time,score
                else:
                    a,end,dwell,d=item
                    if a<time: continue
                    new=(score[0]+1,score[1]+(dwell>=3),score[2]-d)
                if end not in nxt or new>nxt[end][0]:
                    nxt[end]=(new,path+[item])
        states=nxt
    return max(states.values(),key=lambda x:x[0])[1]

def measure(path,step,stops):
    n=int(round(step*199))
    t=(np.arange(n)+.5)/step
    ix=t.astype(int).clip(0,198)
    xy=path[ix]+(t-ix)[:,None]*(path[ix+1]-path[ix])
    slow=np.linalg.norm(path[ix+1]-path[ix],axis=1)/step<1
    dist=[platforms[k][1].query(xy)[0] for k in stops]
    result={}
    for radius in (15,30,50):
        matched=align(dist,slow,radius)
        result[radius]=np.array([-1 if v is None else v[2] for v in matched],int)
    return result

def stats(a):
    served=a>=3
    return dict(n=len(a),missed=int(sum(a<0)),pass_only=int(sum((a>=0)&~served)),
                served=int(served.sum()),median_s=float(np.median(a[served])),
                mean_s=float(np.mean(a[served])),at_least_25=int(sum(a>=25)))
allstats={}; examples={}; lookup={}
for name,trajectories in paths.items():
    collect=defaultdict(list)
    for i,tr in enumerate(trajectories):
        route,stops=schedules[i]
        for r,a in measure(tr,dt[i],stops).items():
            collect[r].append(a)
            if i in (0,499,999) and r==30: examples[name,i]=a
    lookup[name]={r:np.concatenate(v) for r,v in collect.items()}
    allstats[name]={r:stats(a) for r,a in lookup[name].items()}
for key,values in lookup.items():
    for r,a in values.items():
        ref=lookup['SUMO'][r]; use=ref>=3
        loss=np.where(a>=3,ref-a,ref)
        both=use&(a>=3)
        allstats[key][r]['paired']=dict(ref_served=int(use.sum()),missed=int((use&(a<0)).sum()),pass_only=int((use&(a>=0)&(a<3)).sum()),both=int(both.sum()),paired_reference_median=float(np.median(ref[both])),paired_generated_median=float(np.median(a[both])),loss_seconds_by_reason=[int(loss[mask].sum()) for mask in (use&(a<0),use&(a>=0)&(a<3),both)])
Path('metrics.json').write_text(json.dumps(allstats,indent=2)+'\n')
roads=map_shapes(net)
colors=['#b23b39','#ed9d16','#178459']
for seed in (42,43):
    fig,axes=plt.subplots(3,3,figsize=(12.6,11))
    fig.subplots_adjust(left=.06,right=.985,bottom=.07,top=.88,wspace=.15,hspace=.36)
    for row,i in enumerate((0,499,999)):
        route,stops=schedules[i]
        centers=np.stack([platforms[k][0] for k in stops])/1000
        names=['SUMO',f'base-{seed}',f'speed-{seed}']
        pts=np.vstack([centers]+[paths[name][i]/1000 for name in names])
        mid=(pts.max(0)+pts.min(0))/2
        rad=max(.2,np.ptp(pts,axis=0).max()*.56)
        for col,name in enumerate(names):
            ax=axes[row,col]; xy=paths[name][i]/1000
            ax.add_collection(LineCollection(roads,color='#d1d7dc',linewidths=.55))
            ax.plot(*xy.T,color='#46647c',lw=1.4)
            slow=np.linalg.norm(np.diff(xy,axis=0),axis=1)*1000/dt[i]<1
            ax.scatter(*((xy[1:]+xy[:-1])/2)[slow].T,color='#91268a',s=8,zorder=4)
            d=examples[name,i]
            for j,(x,y) in enumerate(centers):
                ax.scatter(x,y,s=52,marker='o',facecolors='none',lw=1.6,edgecolors=colors[0 if d[j]<0 else 1 if d[j]<3 else 2],zorder=5)
            ax.set(aspect='equal',xlim=(mid[0]-rad,mid[0]+rad),ylim=(mid[1]-rad,mid[1]+rad))
            ax.tick_params(labelsize=7)
            ax.set_title(f'{["SUMO","Original","Speed guidance"][col]}   ·   {sum(d>=3)}/{len(d)} with low speed',fontsize=10)
            if col==0: ax.set_ylabel(f'Bus {i+1}  ·  {route}\nNorth (km)',fontsize=8)
            if row==2: ax.set_xlabel('East (km)')
    fig.suptitle(f'Scheduled platforms and low-speed segments   ·   seed {seed}',y=.975)
    fig.legend([Line2D([],[],marker='o',ls='',markerfacecolor='none',color=c) for c in colors]+[Line2D([],[],marker='.',ls='',color='#91268a')],['No assigned approach','Approach only','Low speed ≥3 s','Low-speed positions'],loc='upper center',ncol=4,bbox_to_anchor=(.5,.95))
    fig.savefig(f'stations-{seed}.png',dpi=180);plt.close(fig)
