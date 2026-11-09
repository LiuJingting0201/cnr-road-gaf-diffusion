"""Final-model code intervention audit on fixed training parents, five noise levels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from g33_train import digest,save
from g33_model import pack_motion
from g39_train import load_train
from g40_model import ConditionedMotion,predict
from g40_prior import load_prior,sample_code

LEVELS=(50,250,500,750,950)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        r=json.loads((a.model/'complete.json').read_text());assert r['complete'] and r['steps']==5000
        assert digest(a.model/'model.pt')==r['model_sha256'] and digest(a.model/'train-codes.npz')==r['codes_sha256']
        state=torch.load(a.model/'model.pt',map_location='cpu',weights_only=False)
        model=ConditionedMotion(base=64).cuda().eval();model.load_state_dict(state['motion'])
        prior=load_prior(a.prior,r['model_sha256']);rows,clocks,targets,stats=load_train(a.data,'native_0_25s')
        assert stats==state['stats']
        with np.load(a.model/'train-codes.npz',allow_pickle=False) as f:
            assert all(np.array_equal(f[k],rows[k]) for k in ('ids','parent_ids','labels'))
            codes=f['code']
        chosen=[]
        for c in range(5):
            eligible=sorted(np.flatnonzero(rows['labels']==c),key=lambda i:hashlib.sha256(str(rows['ids'][i]).encode()).digest())
            seen=set()
            for i in eligible:
                if rows['parent_ids'][i] in seen:continue
                chosen.append(i);seen.add(rows['parent_ids'][i])
                if len(seen)==5:break
        schedule=torch.cumprod(1-torch.linspace(.0001,.02,1000,device='cuda'),0);records=[]
        with torch.no_grad():
            for j,i in enumerate(chosen):
                if time.monotonic()-start>1800:raise TimeoutError('30 minute final-model audit bound')
                b=pack_motion([targets[i]],[clocks[i]],[rows['labels'][i]],[rows['paths'][i]],
                    [rows['length'][i]],[rows['duration'][i]],stats,'cuda')
                own=torch.as_tensor(codes[i:i+1],device='cuda');group=chosen[(j//5)*5:(j//5+1)*5];other=group[(j%5+1)%5]
                sampled=sample_code(prior,b,str(rows['ids'][i]),r['seed'])
                options={'assigned_training_code':own,'same_class_other_code':torch.as_tensor(codes[other:other+1],device='cuda'),
                    'zero_code':torch.zeros_like(own),'sampled_prior':sampled}
                for level in LEVELS:
                    gen=torch.Generator(device='cuda').manual_seed(40010000+1000*r['seed']+20*j+LEVELS.index(level))
                    noise=torch.randn(b['x'].shape,generator=gen,device='cuda')*b['mask'];alpha=schedule[level]
                    noisy=alpha.sqrt()*b['x']+(1-alpha).sqrt()*noise;predictions={};losses={}
                    for name,code in options.items():
                        prediction=predict(model,b,noisy,torch.tensor([level],device='cuda'),None if r['arm']=='B' else code)
                        losses[name]=float(((prediction-noise).square()*b['time_weights']).sum());predictions[name]=prediction
                        assert np.isfinite(losses[name])
                    changes={name:float(((pred-predictions['assigned_training_code']).square()*b['time_weights']).sum().sqrt()) for name,pred in predictions.items()}
                    records.append({'class':int(rows['labels'][i]),'parent_order':j,'noise_index':level,'noise_mse':losses,
                        'prediction_rms_change':changes,'sampled_code_abs_max':float(sampled.abs().max()) if sampled is not None else 0.})
        summary={str(level):{name:float(np.mean([v['noise_mse'][name] for v in records if v['noise_index']==level])) for name in options} for level in LEVELS}
        save(a.output/'records.json',{'rows':records})
        save(a.output/'complete.json',{'complete':True,'arm':r['arm'],'seed':r['seed'],'seconds':time.monotonic()-start,
            'training_parents':25,'levels':list(LEVELS),'mean_noise_mse_by_level':summary,'records':records,
            'assigned_training_code_max_abs':float(np.abs(codes).max()),
            'model_source_sha256':r['model_sha256'],'prior_source_sha256':digest(a.prior/'model.pt'),
            'artifacts':{'records.json':digest(a.output/'records.json')},'source_sha256':digest(__file__),
            'scope':'In-sample condition-use audit; SH retains its assigned wrong-trip code. No future evaluation GAF, sampling intervention or fitted parameter changes.',
            'out_of_sample_quality_evidence':False})
    except BaseException as error:save(a.output/'failed.json',{'error':repr(error)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','model','prior','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
