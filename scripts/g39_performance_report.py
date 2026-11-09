"""Check numeric equivalence and report end-to-end performance, not GPU occupancy."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from g39_fast_inputs import checks


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a,b):
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype
        return {'exact':bool(torch.equal(a,b)),'max_abs':float((a.double()-b.double()).abs().max()) if a.numel() else 0.}
    if isinstance(a,np.ndarray):
        return {'exact':bool(np.array_equal(a,b)),'max_abs':float(np.max(np.abs(a.astype(float)-b))) if a.size else 0.}
    if isinstance(a,dict):
        assert set(a)==set(b);values=[compare(a[k],b[k]) for k in a]
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b);values=[compare(x,y) for x,y in zip(a,b)]
    else:return {'exact':bool(a==b),'max_abs':abs(a-b) if isinstance(a,(float,int)) else 0.}
    return {'exact':all(v['exact'] for v in values),'max_abs':max((v['max_abs'] for v in values),default=0.)}


def report(root):
    results={};identical=True
    for arm,seed in [('B',42),('SV',42),('B',43),('SV',43)]:
        loaded={};times={};rates={};memory={};resources={}
        for variant in ['original','cached']:
            times[variant]=[];rates[variant]=[];memory[variant]=[]
            for repeat in range(2):
                p=root/'fits'/f'{arm}-{seed}-{variant}-{repeat}'
                receipt=json.loads((p/'complete.json').read_text());assert receipt['complete']
                times[variant].append(receipt['elapsed_s'])
                history=[json.loads(line) for line in (p/'training.jsonl').read_text().splitlines()]
                first=next(x for x in history if x['step']==100);last=history[-1]
                rates[variant].append(200/(last['elapsed_s']-first['elapsed_s']))
                memory[variant].append(last['gpu_max_allocated_bytes'])
                resources[f'{variant}-{repeat}']=json.loads((root/'timing'/f'{arm}-{seed}-{variant}-{repeat}.json').read_text())
                loaded[(variant,repeat)]=torch.load(p/'checkpoint-latest.pt',map_location='cpu',weights_only=False)
        comparisons={}
        for key in [('original',1),('cached',0),('cached',1)]:
            a=loaded[('original',0)];b=loaded[key]
            values={k:compare(a[k],b[k]) for k in ['model','ema','optimizer','torch_rng','torch_cuda_rng','numpy_rng','stats','valid_tokens_seen']}
            comparisons[f'{key[0]}-{key[1]}']=values;identical &= all(v['exact'] for v in values.values())
        results[f'{arm}-{seed}']={'elapsed_s':times,'steady_updates_per_s':rates,'peak_allocated_bytes':memory,
            'steady_speedup':mean(rates['cached'])/mean(rates['original']),
            'whole_run_speedup':mean(times['original'])/mean(times['cached']),'comparisons':comparisons,
            'resource_observations':resources}
    evaluation={};evaluation_exact=True
    original=root/'evaluation/evaluate-original-0'
    for variant in ['original','cached']:
        seconds=[]
        for repeat in range(2):
            name=f'evaluate-{variant}-{repeat}';p=root/'evaluation'/name
            seconds.append(json.loads((root/'timing'/f'{name}.json').read_text())['wall_s'])
            for filename in ['complete.json','diagnostics.json']:
                equal=json.loads((p/filename).read_text())==json.loads((original/filename).read_text())
                evaluation_exact &= equal
        evaluation[variant]=seconds
    output={'complete':True,'training':results,'training_state_bitwise_equal':bool(identical),
            'evaluation_seconds':evaluation,'evaluation_json_exactly_equal':bool(evaluation_exact),
            'evaluation_speedup':mean(evaluation['original'])/mean(evaluation['cached']),
            'adoption_rule':'Cached training only if state/optimizer/RNG exact and measured faster; evaluation only if full metrics/diagnostics exact.',
            'limits':['300-update checks, not a replacement for full training-quality validation.',
                      'Two orders/repeats per arm and seed on four concurrent L20 GPUs.',
                      'No change to microbatch grouping, effective batch, model, teacher, precision or loss.',
                      'One fixed evaluation cell; speedup need not generalize uniformly.']}
    (root/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='training'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['checks','report']);p.add_argument('--root',type=Path)
    a=p.parse_args()
    if a.action=='checks':
        torch.manual_seed(391);torch.cuda.init();state=torch.cuda.get_rng_state().clone()
        checks('cuda');assert torch.equal(state,torch.cuda.get_rng_state()),'Cache consumed GPU RNG'
    else:report(a.root)
