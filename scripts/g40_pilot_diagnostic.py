"""Training-side implementation diagnostics, never prospective quality evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from g33_train import digest, save
from g33_model import pack_motion
from g39_train import load_train
from g40_model import ConditionedMotion, predict
from g40_prior import load_prior, sample_code, sample_motion


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        r=json.loads((a.model/'complete.json').read_text());assert r['complete']
        assert digest(a.model/'model.pt')==r['model_sha256'] and digest(a.model/'train-codes.npz')==r['codes_sha256']
        state=torch.load(a.model/'model.pt',map_location='cpu',weights_only=False)
        model=ConditionedMotion(base=64).cuda().eval();model.load_state_dict(state['motion'])
        prior=load_prior(a.prior,r['model_sha256']);rows,clocks,targets,stats=load_train(a.data,'native_0_25s')
        assert stats==state['stats']
        with np.load(a.model/'train-codes.npz',allow_pickle=False) as f:
            assert all(np.array_equal(f[k],rows[k]) for k in ('ids','parent_ids','labels'))
            codes=f['code']
        indices=[]
        for c in range(5):
            candidates=sorted(np.flatnonzero(rows['labels']==c),key=lambda i:hashlib.sha256(str(rows['ids'][i]).encode()).digest())
            seen=set()
            for i in candidates:
                if rows['parent_ids'][i] in seen:continue
                indices.append(i);seen.add(rows['parent_ids'][i])
                if len(seen)==5:break
        details=[]
        with torch.no_grad():
            for j,i in enumerate(indices):
                if time.monotonic()-start>1800:raise TimeoutError('Pilot diagnostic time budget')
                batch=pack_motion([targets[i]],[clocks[i]],[rows['labels'][i]],[rows['paths'][i]],
                    [rows['length'][i]],[rows['duration'][i]],stats,'cuda')
                gen=torch.Generator(device='cuda').manual_seed(40010+j)
                noise=torch.randn(batch['x'].shape,generator=gen,device='cuda')*batch['mask']
                alpha=torch.cumprod(1-torch.linspace(.0001,.02,1000,device='cuda'),0)[500]
                noisy=alpha.sqrt()*batch['x']+(1-alpha).sqrt()*noise
                row={'class':int(rows['labels'][i]),'noise_mse':{}}
                true=torch.as_tensor(codes[i:i+1],device='cuda')
                classindices=indices[(j//5)*5:(j//5+1)*5];other=classindices[(j%5+1)%5]
                options={'train_trip_code':true,'other_train_trip_code':torch.as_tensor(codes[other:other+1],device='cuda'),
                         'zero_code':torch.zeros_like(true),'sampled_prior':sample_code(prior,batch,str(rows['ids'][i]),42)}
                predictions={}
                for key,code in options.items():
                    pred=predict(model,batch,noisy,torch.tensor([500],device='cuda'),None if r['arm']=='B' else code)
                    loss=((pred-noise).square()*batch['time_weights']).sum().item()
                    assert np.isfinite(loss);row['noise_mse'][key]=loss;predictions[key]=pred
                row['prediction_change_true_vs_other']=float((predictions['train_trip_code']-predictions['other_train_trip_code']).square().mean().sqrt())
                row['prediction_change_true_vs_zero']=float((predictions['train_trip_code']-predictions['zero_code']).square().mean().sqrt())
                code=options['sampled_prior'];row['sampled_code_abs_max']=float(code.abs().max()) if code is not None else 0.
                assert np.isfinite(row['sampled_code_abs_max'])
                if j%5==0:
                    draws=[]
                    for k in range(2):
                        request={'id':f'pilot-{j}-{k}','label':int(rows['labels'][i]),'length':float(rows['length'][i]),'path':rows['paths'][i]}
                        draw=sample_motion(model,prior,request,stats,42)
                        assert np.isfinite(draw['progress']).all() and np.all(np.diff(draw['progress'])>=0)
                        assert np.allclose(draw['progress'][[0,-1]],[0,1])
                        draws.append(np.interp(np.linspace(0,1,257),draw['time']/draw['time'][-1],draw['progress']))
                    row['different_draw_shape_mae']=float(np.mean(abs(draws[0]-draws[1])))
                details.append(row)
        save(a.output/'complete.json',{'complete':True,'arm':r['arm'],'training_parents':len(indices),
            'seconds':time.monotonic()-start,'model_sha256':r['model_sha256'],'prior_sha256':digest(a.prior/'model.pt'),
            'in_sample_implementation_diagnostic_only':True,'prospective_quality_evidence':False,'rows':details,
            'mean_noise_mse':{key:float(np.mean([d['noise_mse'][key] for d in details])) for key in options},
            'same_class_code_swap_prediction_change':float(np.mean([d['prediction_change_true_vs_other'] for d in details])),
            'zero_code_prediction_change':float(np.mean([d['prediction_change_true_vs_zero'] for d in details])),
            'source_sha256':digest(__file__)})
    except BaseException as error:save(a.output/'failed.json',{'error':repr(error)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('model','prior','data','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
