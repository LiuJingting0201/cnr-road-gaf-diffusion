"""Render completed G36 trajectories and exact GAF inputs, without new selection.

Rule fixed before rendering: morning episode 360201, expanded seed42/teachers,
first lexicographic jointly supported request per class; all five classes kept.
Only figures and hash provenance leave the server, never trajectory archives.
"""
import hashlib,json,os,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from map_speed import map_speed, speed_limit
import torch
from g33_evaluate import read_pool,digest,CLASSES,signals
from g36_feedback import spatial_sequence
from g23_modes import speed_series
from g28_teachers import image_inputs
from cnr.original_gaf import encode_frozen
R=Path('g36-data/360201');O=Path(os.environ.get('G36_VISUAL_OUTPUT','g36-data/visual-v18-final'));O.mkdir(exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
methods=['Uniform','S','V','SV'];names=['Bus','Motorcycle','Private car','Bicycle','Taxi / ride-hailing'];colors={'SUMO':'#20252c','Uniform':'#73808a','S':'#9556ac','V':'#118576','SV':'#cc6137'}
inputs={};figures=[]
def load(folder,name):
 rec,d=read_pool(folder,name);inputs[str(folder/name)]=digest(folder/name);return d
pool=load(R/'pools/new-42','samples.npz');ref=load(R/'reference','references.npz')
selected={m:load(R/'selected/new-42-new'/m,'samples.npz') for m in methods}
mp=np.load(R/'maps/maps.npz');inputs[str(R/'maps/maps.npz')]=digest(R/'maps/maps.npz')
refs={str(i):j for j,i in enumerate(ref['ids'])};look={m:{str(i):j for j,i in enumerate(p['request_ids'])} for m,p in selected.items()};maps={str(i):j for j,i in enumerate(mp['ids'])}
common=set(refs)&set(maps)
for v in look.values():common &= set(v)
cache=Path('../g35d-job/g35-data/gaf-cache/complete.json');meta=json.loads(cache.read_text())['spatial']['frozen_parameters'];inputs[str(cache)]=digest(cache)
statuspath=R/'selected/new-42-new/encoding-status.json';enc={v['id']:v for v in json.loads(statuspath.read_text())['rows']};inputs[str(statuspath)]=digest(statuspath)
sc=np.load(R/'selected/new-42-new/scores.npz');slookup={str(i):j for j,i in enumerate(sc['ids'])}
inputs[str(R/'selected/new-42-new/scores.npz')]=digest(R/'selected/new-42-new/scores.npz')
net=Path('../g30-job/g30-data/source/scenario/in/most.net.xml');lanes=[]
for _,elem in ET.iterparse(net,events=('end',)):
 if elem.tag=='lane' and elem.get('shape'):lanes.append(np.array([[float(v) for v in x.split(',')[:2]] for x in elem.get('shape').split()]))
 elem.clear()
inputs[str(net)]=digest(net)
def track(d,j):
 t=d['txy'][d['offsets'][j]:d['offsets'][j+1]].copy();t[:,0]-=t[0,0];return t
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(O/(name+'.'+ext),bbox_inches='tight',dpi=180)
 plt.close(fig);figures.append(name)
def imgs(t,ident=None):
 seq,clip=spatial_sequence(t[:,1:],meta);rgb,_=encode_frozen(seq[None],np.array(meta['channel_min']),np.array(meta['channel_max']))
 h=hashlib.sha256(rgb[0].tobytes()).hexdigest()
 if ident is not None:assert h==enc[ident]['spatial_image_sha256']
 v,_=speed_series(t[:,0],t[:,1:])
 with torch.no_grad():
  x=image_inputs('speed',v[None]);x=x*x.new_tensor([.229,.224,.225])[None,:,None,None]+x.new_tensor([.485,.456,.406])[None,:,None,None]
 return rgb[0],x[0].permute(1,2,0).cpu().numpy().clip(0,1),seq,v,h,clip
cases=[]
for label,cls in enumerate(CLASSES):
 eligible=sorted(i for i in common if int(ref['labels'][refs[i]])==label);ident=eligible[0]
 tracks={'SUMO':track(ref,refs[ident]),**{m:track(p,look[m][ident]) for m,p in selected.items()}}
 cand=[i for i,r in enumerate(pool['request_ids']) if str(r)==ident];assert len(cand)==8
 mi=maps[ident];route=mp['vertices'][mp['offsets'][mi]:mp['offsets'][mi+1]];origin=route.min(0);span=np.maximum(route.max(0)-origin,1);pad=max(60,.1*span.max());lo=origin-pad;hi=route.max(0)+pad
 nearby=[l-origin for l in lanes if np.all(l.max(0)>=lo) and np.all(l.min(0)<=hi)]
 vmax=speed_limit(tracks)
 fig,axs=plt.subplots(2,2,figsize=(8,7.5));mapaxes=axs[0]
 for ax,key,title in zip(mapaxes,['SUMO','Uniform'],['Independent SUMO realization','Diffusion output: Uniform']):
  t=tracks[key];ax.add_collection(LineCollection(nearby,colors='#d9dfe3',linewidths=.4));ax.plot(*(route-origin).T,color='#8495a3',lw=2.8,alpha=.55)
  p=ax.scatter(t[:,1]-origin[0],t[:,2]-origin[1],c=map_speed(t),s=5,cmap='viridis',vmin=0,vmax=vmax,rasterized=True)
  ax.scatter(*(t[0,1:]-origin),marker='o',s=36,facecolor='white',edgecolor='black',zorder=5);ax.scatter(*(t[-1,1:]-origin),marker='x',s=40,c='black',zorder=5)
  ax.set(xlim=(-pad,span[0]+pad),ylim=(-pad,span[1]+pad),xlabel='Local east (m)',ylabel='Local north (m)',title=title);ax.set_aspect('equal',adjustable='box')
 fig.colorbar(p,ax=list(mapaxes),orientation='horizontal',fraction=.045,pad=.17,label='One-second speed (m/s)',shrink=.75)
 for i in cand:
  t=track(pool,i);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(t[:,1:],axis=0),axis=1))];axs[1,0].plot(t[:,0],arc/arc[-1],c='#b7c7d2',lw=.75,alpha=.8)
 for m,t in tracks.items():
  _,v,_,_,_,arc=signals(t[:,0],t[:,1:]);kw={'color':colors[m],'lw':1.5 if m=='SUMO' else 1,'ls':'--' if m=='SUMO' else '-', 'alpha':.9}
  axs[1,0].plot(t[:,0],arc/arc[-1],**kw,label=m);axs[1,1].plot(np.arange(len(v))+.5,v,**kw)
 axs[1,0].set(title='Eight candidates and selected progress',xlabel='Elapsed time (s)',ylabel='Fraction of travelled distance');axs[1,1].set(title='Selected full-trip movement',xlabel='Elapsed time (s)',ylabel='One-second speed (m/s)')
 for ax in axs[1]:ax.grid(alpha=.2)
 handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.015),frameon=False)
 fig.suptitle(names[label]+': complete generated journeys',fontsize=14);fig.subplots_adjust(top=.91,bottom=.13,hspace=.75,wspace=.35)
 save(fig,'g36-journey-'+cls)
 images={};record={'class':cls,'request_sha256':hashlib.sha256(ident.encode()).hexdigest(),'eligible':len(eligible),'tracks':{}}
 for m,t in tracks.items():
  cid=None if m=='SUMO' else str(selected[m]['ids'][look[m][ident]])
  images[m]=imgs(t,cid);record['tracks'][m]={'duration_s':float(t[-1,0]),'points':len(t),'spatial_image_sha256':images[m][4],'coordinate_clipped':images[m][5]}
  if cid:
   record['tracks'][m].update(candidate_sha256=hashlib.sha256(cid.encode()).hexdigest(),scores=sc['probabilities'][slookup[cid],:,label].tolist())
 fig,axs=plt.subplots(5,3,figsize=(7.6,10),gridspec_kw={'width_ratios':[1,1,1.5]})
 for row,(m,t) in enumerate(tracks.items()):
  sp,ve,seq,v,_,_=images[m];axs[row,0].imshow(sp,interpolation='nearest');axs[row,1].imshow(ve,interpolation='nearest')
  for ax in axs[row,:2]:ax.set_xticks([]);ax.set_yticks([])
  axs[row,0].set_ylabel(m,fontsize=11)
  _,speed,_,_,_,_=signals(t[:,0],t[:,1:]);axs[row,2].plot(np.arange(len(speed))+.5,speed,color=colors[m],lw=.8);axs[row,2].set_xlim(0,max(v[-1,0] for v in tracks.values()));axs[row,2].set_ylim(0,max(35,max(images[k][3].max() for k in images)*1.05));axs[row,2].grid(alpha=.15);axs[row,2].set_ylabel('m/s')
  if m!='SUMO':
   ps,pv=record['tracks'][m]['scores'];axs[row,2].text(.97,.92,f'pS={ps:.3g}, pV={pv:.3g}',transform=axs[row,2].transAxes,ha='right',va='top',fontsize=9)
  if row==0:
   for ax,title in zip(axs[row],['Spatial Hilbert image','Speed image','Corresponding speed']):ax.set_title(title)
  if row==4:axs[row,2].set_xlabel('Elapsed time (s)')
 fig.suptitle(names[label]+': the actual two-view GAF inputs',fontsize=13);fig.tight_layout(rect=(0,.02,1,.96),h_pad=1.0);save(fig,'g36-gaf-'+cls)
 if label==0:
  # Exact sequence-to-field example, with two different normalization rules.
  sp,ve,seq,v,_,_=images['Uniform'];fig,axs=plt.subplots(2,4,figsize=(9,5.4),gridspec_kw={'width_ratios':[1.5,1,1,1]})
  axs[0,0].plot(seq,color='#9556ac',lw=1);axs[0,0].set(title='Ordered Hilbert sequence',xlabel='Compressed index (224)',ylabel='Hilbert value')
  axs[1,0].plot(v,color='#118576',lw=1);axs[1,0].set(title='Whole-trip speed sequence',xlabel='Relative index (128)',ylabel='Speed (m/s)')
  for row,img in enumerate([sp/255,ve]):
   for j,title in enumerate(['GASF','GADF','MTF']):
    axs[row,j+1].imshow(img[:,:,j],cmap='viridis',vmin=0,vmax=1,interpolation='nearest');axs[row,j+1].set_title(title);axs[row,j+1].set_xticks([]);axs[row,j+1].set_yticks([])
  fig.suptitle('One generated bus journey, two mathematical views',fontsize=14);fig.tight_layout(rect=(0,0,1,.93),w_pad=1.1,h_pad=2);save(fig,'g36-gaf-construction')
  fig,axs=plt.subplots(6,4,figsize=(8,10.2),gridspec_kw={'height_ratios':[.75,1,1,.75,1,1]})
  for k,i in enumerate(cand):
   row=(k//4)*3;col=k%4;cid=str(pool['ids'][i]);t=track(pool,i);sp,ve,seq,v,_,_=imgs(t,cid);tags=[m for m in methods if str(selected[m]['ids'][look[m][ident]])==cid]
   _,speed,_,_,_,_=signals(t[:,0],t[:,1:]);axs[row,col].plot(np.arange(len(speed))+.5,speed,lw=.7,c='#176b94');axs[row,col].set_ylim(0,35);axs[row,col].set_title('Candidate '+str(k+1)+'\n'+('/'.join(tags) or 'not selected'),fontsize=10);axs[row,col].tick_params(labelsize=8)
   axs[row+1,col].imshow(sp);axs[row+2,col].imshow(ve)
   for ax in axs[row+1:row+3,col]:ax.set_xticks([]);ax.set_yticks([])
  for row in [0,3]:
   for ax,title in zip(axs[row:row+3,0],['Speed (m/s)','Spatial','Velocity']):ax.set_ylabel(title)
  fig.suptitle('Eight candidates before GAF selection',fontsize=14);fig.tight_layout(rect=(0,0,1,.96),h_pad=1.2);save(fig,'g36-eight-candidates')
 cases.append(record)
receipt={'complete':True,'rule':'episode360201, expanded native seed42 and expanded teachers, first lexicographic jointly supported request per class, all five retained; illustrative not inferential','inputs_sha256':inputs,'script_sha256':digest(Path(__file__)),'figures':figures,'cases':cases,'spatial_hash_checks':'every plotted generated image equals recorded scoring image','speed_images':'exact image_inputs inverse ImageNet normalization; no per-panel contrast scaling','scope':'read-only completed artifacts, no fitting/scoring/selection/test access'}
(O/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'complete':True,'figures':len(figures),'cases':len(cases)}))
