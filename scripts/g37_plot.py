"""Plot first supported OD per class, every catalog route; no quality selection."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from g33_evaluate import digest,signals
from g36_feedback import spatial_sequence
from g23_modes import speed_series
from g28_teachers import image_inputs
from cnr.original_gaf import encode_frozen
import torch
R=Path('g37-data');O=R/'figures';O.mkdir(exist_ok=False)
assert json.loads((R/'evaluation/complete.json').read_text())['complete']
cat=json.loads((R/'routes/routes.json').read_text())['rows'];rows=json.loads((R/'generated/candidates.json').read_text())['rows'];sel=json.loads((R/'generated/selections.json').read_text())['rows'];p=np.load(R/'generated/samples.npz');meta=json.loads(Path('../g35d-job/g35-data/gaf-cache/complete.json').read_text())['spatial']['frozen_parameters'];names=['Bus (ordered stops retained)','Motorcycle','Private car','Bicycle','Taxi / ride-hailing'];colors={'Uniform':'#68737b','S':'#9c55af','V':'#078a79','SV':'#d47b2b'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def track(i):return p['txy'][p['offsets'][i]:p['offsets'][i+1]]
cases=[]
for label in range(5):
 eligible=sorted([r for r in cat if r['label']==label and r['routes']],key=lambda r:r['id'])
 if not eligible:continue
 r=eligible[0];origin=np.array(r['routes'][0]['vertices'][0]);fig,axs=plt.subplots(1,3,figsize=(13,4.4));fig.suptitle(names[label]+': OD-conditioned route and motion candidates')
 for j,route in enumerate(r['routes']):
  xy=np.array(route['vertices'])-origin;axs[0].plot(*xy.T,label=f'R{j+1}: {route["length"]:.0f}m',lw=1.4,alpha=.8)
 chosen={s['method']:rows[s['index']] for s in sel if s['arm']=='multi' and s['request_id']==r['id']}
 for m,row in chosen.items():
  t=track(row['index']);xy=t[:,1:]-origin;axs[1].plot(*xy.T,lw=1.5,color=colors[m],alpha=.8,label=f'{m}: R{row["route_index"]+1}');_,v,_,_,_,_=signals(t[:,0],t[:,1:]);axs[2].plot(np.arange(len(v))+.5,v,color=colors[m],lw=.8,label=m)
 for ax in axs[:2]:ax.set(xlabel='Local east (m)',ylabel='Local north (m)');ax.set_aspect('equal',adjustable='datalim');ax.scatter(0,0,c='black',s=20,zorder=5)
 axs[0].set_title('All viable route candidates');axs[0].legend(fontsize=8);axs[1].set_title('Recorded multi-route selections');axs[1].legend(fontsize=8);axs[2].set(title='Selected movement',xlabel='Elapsed time (s)',ylabel='One-second speed (m/s)');axs[2].legend(fontsize=8);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(O/f'routes-{label}.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(4,2,figsize=(6,10))
 for i,(m,row) in enumerate(chosen.items()):
  t=track(row['index']);seq,_=spatial_sequence(t[:,1:],meta);rgb,_=encode_frozen(seq[None],np.asarray(meta['channel_min']),np.asarray(meta['channel_max']));assert hashlib.sha256(rgb[0].tobytes()).hexdigest()==row['spatial_image_sha256'];v,_=speed_series(t[:,0],t[:,1:])
  with torch.no_grad():im=image_inputs('speed',v[None]);im=im*im.new_tensor([.229,.224,.225])[None,:,None,None]+im.new_tensor([.485,.456,.406])[None,:,None,None]
  axs[i,0].imshow(rgb[0]);axs[i,1].imshow(im[0].permute(1,2,0).cpu().numpy().clip(0,1));axs[i,0].set_ylabel(f'{m}: R{row["route_index"]+1}')
  for ax in axs[i]:ax.set_xticks([]);ax.set_yticks([])
 axs[0,0].set_title('Spatial Hilbert GAF');axs[0,1].set_title('Speed GAF');fig.suptitle(names[label]);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(O/f'gaf-{label}.png',dpi=150,bbox_inches='tight');plt.close(fig)
 cases.append({'class':label,'request_sha256':hashlib.sha256(r['id'].encode()).hexdigest(),'routes':len(r['routes'])})
(O/'complete.json').write_text(json.dumps({'complete':True,'case_rule':'first supported OD per class, all catalog routes and selectors','cases':cases,'script_sha256':digest(__file__),'generated_sha256':digest(R/'generated/complete.json')},indent=2)+'\n')
