"""Read-only prospective code-range and decoder audit after full G40 closure.

Uses recorded candidate metadata and training codes, never refits or reselects.
Exceeding an observed training maximum is descriptive, not a failure threshold.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ARMS=('B','F-SV','J-S','J-V','J-SV','J-RAW','J-SH','C')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def summarize(rows,training_max):
    assert rows
    code=np.asarray([r['code_abs_max'] for r in rows],float)
    bins=sum(r['motion_bins'] for r in rows)
    assert np.isfinite(code).all() and np.all(code>=0) and bins>0
    return {'candidates':len(rows),'code_abs_max_quantiles':dict(zip(
        ('min','median','p95','p99','max'),map(float,np.quantile(code,[0,.5,.95,.99,1])))),
        'training_code_abs_max':training_max,
        'exceed_training_max_count':int(np.sum(code>training_max)),
        'negative_decode_fraction':sum(r['negative_bins'] for r in rows)/bins,
        'upper_decode_fraction':sum(r['upper_clipped_bins'] for r in rows)/bins,
        'constant_speed_fallbacks':sum(r['constant_speed_fallback'] for r in rows)}


def main(a):
    start=time.monotonic();closed=read(a.root/'all-stages-complete.json')
    assert closed['complete'] and not (a.root/'failed.json').exists()
    assert digest(a.root/'complete.json')==closed['primary_receipt_sha256']
    sources={};training={};cells={};pooled={}
    for arm in ARMS:
        pooled[arm]=[]
        for seed in (42,43):
            name=f'{arm}-{seed}';model=a.root/'models'/name;receipt=read(model/'complete.json')
            assert receipt['complete'] and receipt['steps']==5000
            assert digest(model/'train-codes.npz')==receipt['codes_sha256']
            with np.load(model/'train-codes.npz',allow_pickle=False) as f:
                code=f['code'];assert np.isfinite(code).all();maximum=float(np.abs(code).max())
                class_max=[float(np.abs(code[f['labels']==c]).max()) for c in range(5)]
            training[name]={'all_code_abs_max':maximum,'class_code_abs_max':class_max,
                'code_sha256':receipt['codes_sha256']}
            sources[str(model/'complete.json')]=digest(model/'complete.json')
            for episode in range(400201,400209):
                if time.monotonic()-start>600:raise TimeoutError('Ten-minute metadata-audit bound')
                folder=a.root/str(episode)/'generated'/name;gen=read(folder/'complete.json')
                assert gen['complete'] and gen['model_sha256']==receipt['model_sha256']
                assert not gen['true_future_GAF_opened'] and not gen['reference_opened']
                assert digest(folder/'candidates.json')==gen['artifacts']['candidates.json']
                rows=read(folder/'candidates.json')['rows'];assert len(rows)==gen['candidates']
                result={'all':summarize(rows,maximum),'by_class':{
                    c:summarize([r for r in rows if r['label']==i],class_max[i])
                    for i,c in enumerate(receipt['classes'])}}
                cells[f'{episode}/{name}']=result
                pooled[arm].append({'episode':episode,'seed':seed,**result['all']})
                sources[str(folder/'complete.json')]=digest(folder/'complete.json')
    a.output.mkdir(parents=True,exist_ok=False)
    result={'complete':True,'seconds':time.monotonic()-start,'models':16,'cells':128,
        'training':training,'cells_by_name':cells,'by_arm_cells':pooled,
        'sources':sources,'all_stages_sha256':digest(a.root/'all-stages-complete.json'),
        'source_sha256':digest(__file__),
        'scope':'Recorded prospective code ranges and decoder accounting; all four draws. No new sampling, fitting, threshold-based exclusion or quality endpoint change.',
        'limits':'An observed training maximum is not the support boundary of a learned distribution. Exceedance alone is not proof of invalidity.'}
    p=a.output/'summary.json';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output/'complete.json').write_text(json.dumps({'complete':True,'models':16,'cells':128,
        'artifacts':{'summary.json':digest(p)},'source_sha256':digest(__file__)},indent=2)+'\n')
    print(json.dumps({'complete':True,'cells':128,'seconds':time.monotonic()-start}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
