"""Finite OD multi-route diffusion pilot; reference opened only by evaluate action."""
import argparse,hashlib,inspect,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from g33_model import MotionDenoiser,sample,CLASSES
from g33_evaluate import digest,save,curve_xy,extract_record,weighted_signal_distance,read_pool
from g36_feedback import load_reader,spatial_sequence,choose
from g33_feedback import permutation
from g23_modes import speed_series
from g28_teachers import image_inputs
from cnr.original_gaf import encode_frozen
METHODS=['Uniform','S','V','SV'];ARMS=['single','multi']
def guard(start,out):
 if time.monotonic()-start>7200 or shutil.disk_usage('.').free<12*1024**3:raise RuntimeError('G37 time/disk guard')
 if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>8*1024**3:raise RuntimeError('G37 artifact guard')
def generate(catalog,modelroot,teachers,cache,out):
 out.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
 try:
  receipt=json.loads((catalog.parent/'complete.json').read_text());assert receipt['complete'] and receipt['routes_sha256']==digest(catalog)
  models={};ch=digest(cache/'complete.json');meta=json.loads((cache/'complete.json').read_text())['spatial']['frozen_parameters']
  for v in ['spatial','speed']:models[v],_=load_reader(teachers/f'{v}-42',ch,v)
  mr=json.loads((modelroot/'complete.json').read_text());assert mr['complete'] and mr['model_sha256']==digest(modelroot/'model.pt') and mr['stats_sha256']==digest(modelroot/'stats.json') and mr['config_sha256']==digest(modelroot/'config.json')
  conf=json.loads((modelroot/'config.json').read_text());assert conf['model_source_sha256']==digest(inspect.getsourcefile(MotionDenoiser))
  state=torch.load(modelroot/'model.pt',weights_only=False,map_location='cpu');assert state['seed']==42 and state['arm']=='native_0_25s' and state['classes']==CLASSES
  model=MotionDenoiser(base=state['width']).cuda().eval();model.load_state_dict(state['state_dict']);stats=json.loads((modelroot/'stats.json').read_text());assert state['stats']==stats
  rows=[];tracks=[];selections=[];progress=[];calls={'n':0};model.register_forward_hook(lambda *args:calls.__setitem__('n',calls['n']+1))
  for r in json.loads(catalog.read_text())['rows']:
   if not r['routes']:progress.append({'request_id':r['id'],'label':r['label'],'status':'no_route'});continue
   single_tracks={}
   for arm in ARMS:
    batch=[];scores=[]
    for k in range(8):
     guard(started,out);route_index=0 if arm=='single' else k%len(r['routes']);route=r['routes'][route_index];name=f"{r['id']}/candidate-{k:02d}"
     request={'id':name,'label':r['label'],'length':route['length'],'path':np.asarray(route['path'],np.float32)}
     with torch.no_grad():draw=sample(model,[request],stats,'native_0_25s',42,device='cuda',ddim_steps=100)[0]
     xy=curve_xy(np.asarray(route['vertices']),draw['progress']);txy=np.c_[draw['time'],xy]
     assert np.allclose(xy[[0,-1]],np.asarray(route['vertices'])[[0,-1]],atol=1e-8,rtol=0) and np.all(np.diff(draw['progress'])>=0)
     if arm=='single':single_tracks[k]=txy
     elif route_index==0:assert np.array_equal(txy,single_tracks[k]),'same-route control differs'
     row={'id':name,'request_id':r['id'],'label':r['label'],'arm':arm,'candidate':k,'route_index':route_index,'route_length':route['length'],'detour_ratio':route['detour_ratio'],'duration':draw['duration'],'samples':len(txy),'status':'generated'}
     try:
      seq,clipped=spatial_sequence(xy,meta);rgb,chclip=encode_frozen(seq[None],np.asarray(meta['channel_min']),np.asarray(meta['channel_max']));v,_=speed_series(txy[:,0],xy)
      with torch.no_grad():
       probs=[]
       for view,value in [('spatial',rgb),('speed',v[None])]:
        logits=models[view](image_inputs(view,value));assert torch.isfinite(logits).all();probs.append(float(logits.softmax(1)[0,r['label']]))
      row.update(probabilities=probs,spatial_image_sha256=hashlib.sha256(rgb[0].tobytes()).hexdigest(),xy_clipped_points=clipped,channel_clipped=int(chclip),speed_above_35=int((v>35).sum()))
     except ValueError as e:row.update(probabilities=None,encoding_failed=str(e));probs=[np.nan,np.nan]
     row['index']=len(rows);batch.append(row);scores.append(probs);rows.append(row);tracks.append(txy)
    q=np.array(scores);choice=choose(q[:,0],q[:,1],permutation(r['id'],8),bool(np.isfinite(q).all()))
    for method,k in choice.items():selections.append({'request_id':r['id'],'label':r['label'],'arm':arm,'method':method,'index':batch[k]['index'],'fallback':not bool(np.isfinite(q).all())})
    progress.append({'request_id':r['id'],'label':r['label'],'arm':arm,'status':'generated','seconds':time.monotonic()-started});save(out/'progress.json',{'done':progress,'candidates':len(rows),'seconds':time.monotonic()-started});print(json.dumps(progress[-1]),flush=True)
  np.savez_compressed(out/'samples.npz',offsets=np.r_[0,np.cumsum([len(t) for t in tracks])],txy=np.concatenate(tracks),ids=np.asarray([r['id'] for r in rows]),request_ids=np.asarray([r['request_id'] for r in rows]),labels=np.asarray([r['label'] for r in rows]),arms=np.asarray([r['arm'] for r in rows]))
  save(out/'candidates.json',{'rows':rows});save(out/'selections.json',{'rows':selections})
  summary={}
  for arm in ARMS:
   summary[arm]={}
   for label,c in enumerate(CLASSES):
    cc=[r for r in rows if r['label']==label and r['arm']==arm];ids=sorted({r['request_id'] for r in cc});detail={}
    uniform={s['request_id']:rows[s['index']] for s in selections if s['arm']==arm and s['method']=='Uniform' and s['label']==label}
    for m in METHODS:
     selected=[rows[s['index']] for s in selections if s['arm']==arm and s['method']==m and s['label']==label]
     detail[m]={'n':len(selected),'route_changed_from_uniform':sum(v['route_index']!=uniform[v['request_id']]['route_index'] for v in selected),'mean_detour':float(np.mean([v['detour_ratio'] for v in selected])) if selected else None,'mean_length_m':float(np.mean([v['route_length'] for v in selected])) if selected else None}
    summary[arm][c]={'generated_requests':len(ids),'candidates':len(cc),'unique_spatial_images_per_pool':[len({v.get('spatial_image_sha256') for v in cc if v['request_id']==i and v.get('spatial_image_sha256')}) for i in ids],'selection':detail}
  save(out/'complete.json',{'complete':True,'reference_opened':False,'learned_route_choice':False,'denominator':20,'candidates':len(rows),'network_forward_calls':calls['n'],'seconds':time.monotonic()-started,'model_sha256':digest(modelroot/'model.pt'),'catalog_sha256':digest(catalog),'cache_sha256':ch,'samples_sha256':digest(out/'samples.npz'),'candidates_sha256':digest(out/'candidates.json'),'selections_sha256':digest(out/'selections.json'),'script_sha256':digest(__file__),'mechanism':summary})
 except BaseException as e:save(out/'failed.json',{'error':repr(e),'seconds':time.monotonic()-started});raise

def evaluate(generated,reference,out):
 out.mkdir(parents=True,exist_ok=False)
 try:
  receipt=json.loads((generated/'complete.json').read_text());assert receipt['complete'] and not receipt['reference_opened']
  for name in ['samples','candidates','selections']:
   ext='npz' if name=='samples' else 'json';assert receipt[name+'_sha256']==digest(generated/(name+'.'+ext))
  _,ref=read_pool(reference,'references.npz');pool=np.load(generated/'samples.npz');rows=json.loads((generated/'candidates.json').read_text())['rows'];sel=json.loads((generated/'selections.json').read_text())['rows']
  rr={str(id):dict(extract_record(ref['txy'][ref['offsets'][i]:ref['offsets'][i+1]]),request_id=str(id),label=int(ref['labels'][i])) for i,id in enumerate(ref['ids'])}
  gr=[dict(extract_record(pool['txy'][pool['offsets'][i]:pool['offsets'][i+1]]),**row) for i,row in enumerate(rows)];results={}
  for arm in ARMS:
   results[arm]={}
   for m in METHODS:
    indices=[s['index'] for s in sel if s['arm']==arm and s['method']==m];groups={}
    for label,c in enumerate(CLASSES):
     a=[gr[i] for i in indices if gr[i]['label']==label and gr[i]['request_id'] in rr];b=[rr[v['request_id']] for v in a];d={'requested':4,'paired':len(a),'native_speed_w1_mps':weighted_signal_distance(a,b,'speed'),'native_acceleration_w1_mps2':weighted_signal_distance(a,b,'acceleration'),'physical':{}}
     if a:
      for k in a[0]['physical']:
       pairs=[(x['physical'][k],y['physical'][k]) for x,y in zip(a,b) if x['physical'][k] is not None and y['physical'][k] is not None]
       d['physical'][k]={'n':len(pairs),'w1':float(wasserstein_distance(*zip(*pairs))) if pairs else None,'mae':float(np.mean([abs(x-y) for x,y in pairs])) if pairs else None}
     groups[c]=d
    results[arm][m]={'by_class':groups,'macro_native_speed_w1_mps':float(np.mean([v['native_speed_w1_mps'] for v in groups.values()])) if all(v['native_speed_w1_mps'] is not None for v in groups.values()) else None,'macro_native_acceleration_w1_mps2':float(np.mean([v['native_acceleration_w1_mps2'] for v in groups.values()])) if all(v['native_acceleration_w1_mps2'] is not None for v in groups.values()) else None}
  save(out/'complete.json',{'complete':True,'scope':'20 OD development pilot, one reused G36 reference realization per OD; not route-choice frequency validation or confirmatory evidence','generation_sha256':digest(generated/'complete.json'),'reference_sha256':digest(reference/'complete.json'),'results':results})
 except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate']);p.add_argument('--catalog',type=Path);p.add_argument('--model',type=Path);p.add_argument('--teachers',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--generated',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if a.action=='generate':generate(a.catalog,a.model,a.teachers,a.cache,a.output)
 else:evaluate(a.generated,a.reference,a.output)
