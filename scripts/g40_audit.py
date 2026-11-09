"""Finite training-only GAF sensitivity and native decoder replay audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from g33_train import digest, save
from g33_prepare import encode as observed_encoding
from g33_model import encode as motion_encoding, decode as motion_decode, CLASSES
from g33_evaluate import curve_xy, extract_record
from g35c_teachers import load_dataset, load_cache, speed_profile
from g36_feedback import spatial_sequence
from g28_teachers import image_inputs
from cnr.original_gaf import encode_frozen
from g40_model import reader, split_reader
from g40_data import source_states


def variants(times, xy):
    native=observed_encoding(np.c_[times,xy]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    length=arc[-1];assert length>0
    decoded,flags=motion_decode(motion_encoding(native['progress'],times),times,0.,1.)
    result={'copy':xy.copy(),'replay':curve_xy(native['path_xy'],decoded)}
    s=times/times[-1]
    for strength in (.1,.3):
        target=times[-1]*(s+strength*np.sin(2*np.pi*s)/(2*np.pi));target[[0,-1]]=times[[0,-1]]
        q=np.interp(target,times,arc)/length
        result[f'warp-{strength}']=curve_xy(native['path_xy'],q)
        d=np.diff(arc).copy();n=len(d)//4
        block=d[:4*n].reshape(n,4);changed=block*np.array([1+strength,1-strength,1+strength,1-strength])
        changed*=block.sum(1,keepdims=True)/np.maximum(changed.sum(1,keepdims=True),1e-30)
        d[:4*n]=changed.flatten();q=np.r_[0.,np.cumsum(d)]/length;q[-1]=1.
        result[f'jitter-{strength}']=curve_xy(native['path_xy'],q)
    for width in (4,16):
        d=np.diff(arc);filtered=np.convolve(d,np.ones(width)/width,mode='full')[(width-1)//2:(width-1)//2+len(d)]
        q=np.r_[0.,np.cumsum(filtered)];q/=q[-1]
        result[f'smooth-{width}']=curve_xy(native['path_xy'],q)
    for name,value in result.items():
        assert value.shape==xy.shape and np.isfinite(value).all(),name
        assert np.allclose(value[[0,-1]],xy[[0,-1]],atol=1e-7,rtol=0),name
    return result


def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        source,raw=load_dataset(a.data);cr,index,stored_images=load_cache(a.cache,'spatial')
        assert all(np.array_equal(raw[k],index[k]) for k in ('ids','parent_ids','labels','partition'))
        states,shas=source_states(a.teachers,a.teachers,'gaf');models={};means={}
        for v in ('spatial','speed'):
            m=reader('gaf',v);m.load_state_dict(states[v]);m.cuda().eval();models[v]=split_reader(m,'gaf')
            path=a.teachers/(v+'-42')/'train-predictions.npz'
            tr=json.loads((a.teachers/(v+'-42')/'complete.json').read_text())
            assert digest(path)==tr['artifacts'][path.name]['sha256']
            with np.load(path,allow_pickle=False) as f:means[v]=f['features'].astype(np.float32).mean(0)
        selected=[]
        for c in range(5):
            eligible=np.flatnonzero((raw['partition']=='train')&(raw['labels']==c))
            order=sorted(eligible,key=lambda i:hashlib.sha256(str(raw['ids'][i]).encode()).digest());seen=set()
            for i in order:
                p=str(raw['parent_ids'][i])
                if p in seen:continue
                selected.append(i);seen.add(p)
                if len(seen)==100:break
        records=[];meta=cr['spatial']['frozen_parameters']
        for number,i in enumerate(selected):
            if time.monotonic()-started>1800:raise TimeoutError('30 minute bounded audit')
            lo,hi=raw['offset'][i:i+2];t=raw['time'][lo:hi];xy=raw['xy'][lo:hi]
            pairs={'original':xy,**variants(t,xy)};spatial=[];velocity=[];signals={}
            for name,value in pairs.items():
                seq,clipped=spatial_sequence(value,meta)
                im,_=encode_frozen(seq[None],np.asarray(meta['channel_min']),np.asarray(meta['channel_max']))
                v,_=speed_profile(t,value);spatial.append(im[0]);velocity.append(v)
                signals[name]=extract_record(np.c_[t,value])
            assert np.array_equal(spatial[0],stored_images[i]),'Source spatial bytes changed'
            assert np.array_equal(velocity[0],index['velocity'][i]),'Source speed profile changed'
            assert np.array_equal(spatial[0],spatial[1]) and np.array_equal(velocity[0],velocity[1])
            features={}
            with torch.no_grad():
                for view,values in [('spatial',np.stack(spatial)),('speed',np.stack(velocity))]:
                    prefix,tail,head=models[view];h=tail(prefix(image_inputs(view,values))).mean((2,3)).cpu().numpy()
                    h=h-means[view];h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12);features[view]=h
            original=signals['original'];label=int(raw['labels'][i])
            for j,name in enumerate(pairs):
                if name=='original':continue
                row={'class':CLASSES[label],'variant':name,'parent_hash':hashlib.sha256(str(raw['parent_ids'][i]).encode()).hexdigest(),
                    'declared_route_and_duration_unchanged':True,'spatial_changed_pixel_fraction':float(np.mean(spatial[j]!=spatial[0])),
                    'speed_profile_mean_abs_change':float(np.mean(np.abs(velocity[j]-velocity[0])))}
                for view,h in features.items():
                    row[view+'_feature_cosine_distance']=float(max(0.,1-np.dot(h[j],h[0])))
                    row[view+'_feature_l2_distance']=float(np.linalg.norm(h[j]-h[0]))
                for signal in ('speed','acceleration'):
                    x,y=original[signal],signals[name][signal]
                    row[signal+'_native_w1_change']=float(wasserstein_distance(x,y,original[signal+'_weight'],signals[name][signal+'_weight'])) if len(x) and len(y) else None
                for key,value in original['physical'].items():
                    alt=signals[name]['physical'][key]
                    row['physical_abs_change/'+key]=float(abs(alt-value)) if value is not None and alt is not None else None
                records.append(row)
            if number%25==0:
                save(out/'progress.json',{'parents_done':number+1,'total':len(selected),'seconds':time.monotonic()-started})
                print(json.dumps({'parents_done':number+1,'total':len(selected)}),flush=True)
        summary={}
        for c in CLASSES:
            summary[c]={}
            for name in pairs:
                if name=='original':continue
                rr=[r for r in records if r['class']==c and r['variant']==name]
                columns=[k for k in rr[0] if k not in ('class','variant','parent_hash','declared_route_and_duration_unchanged')]
                summary[c][name]={'n':len(rr),'metrics':{}}
                for k in columns:
                    values=[r[k] for r in rr if r[k] is not None]
                    summary[c][name]['metrics'][k]={'n':len(values),'mean':float(np.mean(values)) if values else None,
                        'quantiles_0_10_50_90_100':np.quantile(values,[0,.1,.5,.9,1]).tolist() if values else []}
        save(out/'records.json',{'rows':records})
        save(out/'complete.json',{'complete':True,'training_parents':len(selected),'summary':summary,'teacher_sha256':shas,
            'dataset_receipt_sha256':digest(a.data/'complete.json'),'cache_receipt_sha256':digest(a.cache/'complete.json'),
            'records_sha256':digest(out/'records.json'),'seconds':time.monotonic()-started,'no_fitting_or_validation_outcomes':True,
            'limits':['Controlled changes measure sensitivity, not a ranking of true realism.',
                      'Continuous route/T preserved; sampled coordinate-chord lengths may change on curves.',
                      'Native subsecond information may be invisible to the one-second/128-value speed encoder.']})
    except BaseException as error:save(out/'failed.json',{'error':repr(error),'seconds':time.monotonic()-started});raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','cache','teachers','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
