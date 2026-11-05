"""Evaluate scorer transfer and ranking on entirely new fixed-path candidate pools."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from g12_fit import model,metrics,predict
from g8_common import tensor
from cnr.original_gaf import trajectory_sequence,encode_frozen
from cnr.path_diffusion import PathDenoiser,decode
root=Path('g12-data')
@torch.no_grad()
def run():
    cfg=json.loads(Path('runs/long-path-e3/seed42/config.json').read_text());meta=cfg['dataset']['source_protocol']
    q=np.load(root/'requests.npz');y=q['labels'];ref=np.load(root/'references.npz')
    assert np.array_equal(q['ids'],ref['ids'])
    def encode(x):
        seq=[];ok=[]
        for row in x:
            try:
                h,_=trajectory_sequence(row,np.array(meta['xy_min']),np.array(meta['xy_max']),eps=meta['hilbert_eps']);good=True
            except ValueError:h=np.zeros(224);good=False
            seq.append(h);ok.append(good)
        im,_=encode_frozen(seq,np.array(meta['channel_min']),np.array(meta['channel_max']),batch=16)
        return torch.from_numpy(im).cuda(),np.array(ok)
    chosen={};transfer={};oldnet=model();newnet=model();obs,valid=encode(ref['xy'])
    probs=lambda x:torch.tensor(x).softmax(-1).numpy()
    for seed in [44,43,42]:
        fits=[json.loads((root/f'{pref}fit{seed}.json').read_text()) for pref in ['', 'reg-']]
        prefix=['','reg-'][int(max(x[1] for x in fits[1]['curve'])>max(x[1] for x in fits[0]['curve']))]
        chosen[seed]=prefix;record=fits[int(bool(prefix))];T=record['T']
        oldnet.load_state_dict(torch.load(f'runs/gaf-duration-e2/length512-seed{seed}/final.pt',weights_only=True))
        newnet.load_state_dict(torch.load(root/f'{prefix}scorer{seed}.pt',weights_only=True))
        old= predict(oldnet,obs);new=predict(newnet,obs)
        transfer[seed]=dict(recipe=prefix or 'plain',T=T,valid=int(valid.sum()),original=metrics(old[valid],y[valid],range(1,6)),selected=metrics(new[valid],y[valid],range(1,6)),calibrated=metrics(new[valid]/T,y[valid],range(1,6)))
    # loop ends on the preregistered seed42 scorer, and corresponding temperature
    den=PathDenoiser().cuda().eval()
    den.load_state_dict(torch.load('runs/long-path-e3/seed42/latest.pt',weights_only=True)['ema'])
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    times=torch.linspace(999,0,100).long().tolist()
    arrays={k:[] for k in ['random','original','selected','calibrated']};caps={k:[] for k in arrays}
    diag=[];bad=0
    for i,c in enumerate(y):
        start=tensor(q['xy'][i,0]).expand(8,2);route=tensor(q['route'][i]).expand(8,-1,-1)
        L=tensor(q['route_length'][i]).expand(8);lab=torch.full((8,),int(c),device='cuda')
        sn=(start-tensor(cfg['dataset']['mean']))/tensor(cfg['dataset']['std']);cond=((route[:,::8]-start[:,None])/128).transpose(1,2)
        rng=torch.Generator(device='cuda').manual_seed(412003+int(i));z=torch.randn(8,1,511,device='cuda',generator=rng)
        for j,t in enumerate(times):
            e=den(z,torch.full_like(lab,t),lab,sn,cond);clean=(z-(1-alpha[t]).sqrt()*e)/alpha[t].sqrt()
            ap=alpha[times[j+1]] if j<99 else 1.;z=ap**.5*clean+(1-ap)**.5*e
        xy,p=decode(z,cfg['log_increment_mean'],cfg['log_increment_std'],route,L,start)
        x=xy.cpu().numpy();im,ok=encode(x);bad+=int((~ok).sum());lo=predict(oldnet,im);ln=predict(newnet,im)
        slot=np.random.default_rng(62401+i).permutation(8)
        elig=slot[ok[slot]];ix={'random':int((elig if len(elig) else slot)[0])}
        for key,logit in [('original',lo),('selected',ln),('calibrated',ln/T)]:
            ix[key]=int(elig[np.argmax(probs(logit)[elig,c])]) if len(elig) else ix['random']
        # Independent motion diagnostic: exact within-pool uniform mean versus selected.
        # Conditional observed futures are used here only after samples and scores exist.
        speed=np.linalg.norm(np.diff(x,axis=1),axis=-1)/.25;rv=np.linalg.norm(np.diff(ref['xy'][i],axis=0),axis=-1)/.25
        cost=np.array([[wasserstein_distance(v,rv),abs(v.sum()-rv.sum())*.25,abs((v<.1).mean()-(rv<.1).mean())] for v in speed])
        u=cost[elig].mean(0) if len(elig) else cost.mean(0)
        diag.append({key:(cost[v]-u).tolist() for key,v in ix.items()}|{'indices':ix,'episode':int(q['episode'][i]),'label':int(c)})
        for k,v in ix.items():arrays[k].append(x[v]);caps[k].append(bool(p[v,-1]>=L[v]))
    np.savez_compressed(root/'rank.npz',**arrays)
    (root/'scorer.json').write_text(json.dumps(dict(transfer=transfer,candidate_encoding_failures=bad,reference_encoding_failures=int(sum(~valid)),caps={k:sum(v) for k,v in caps.items()},within_pool=diag),indent=1))
if __name__=='__main__':torch.set_num_threads(2);run()
