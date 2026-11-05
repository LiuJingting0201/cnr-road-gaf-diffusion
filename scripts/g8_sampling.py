"""One late GAF resampling step on shared legal paths. No test inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from g8_common import assets,Scores,tensor,reference
from cnr.path_diffusion import PathDenoiser,decode
from cnr.topology import LaneConsistency
from cnr.data import CLASSES

@torch.no_grad()
def sample(seed,balanced=False,terminal_check=False):
    cfg,meta,d=assets()
    sc=Scores(meta); model=PathDenoiser().cuda().eval()
    model.load_state_dict(torch.load(f'runs/long-path-e3/seed{seed}/latest.pt',weights_only=True)['ema'])
    rng=np.random.default_rng(71809)
    noise=torch.Generator(device='cuda').manual_seed(71811)
    times=torch.linspace(999,0,100).long().tolist()
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    out={k:[] for k in ['identity','uniform','mid_gaf','wrong','post_gaf']}
    if terminal_check:
        out={k:[] for k in ['identity','post_gaf','post_max']}
    diagnostic=[];calls=0;gaf_seconds=0.;caps={k:[] for k in out};div={k:[] for k in out}
    for i,c in enumerate(d['labels']):
        start=tensor(d['xy'][i,0]).expand(8,2)
        route=tensor(d['route'][i]).expand(8,-1,-1)
        length=tensor(d['route_length'][i]).expand(8)
        route_cond=((route[:,::8]-start[:,None])/128).transpose(1,2)
        lab=torch.full((8,),int(c),device='cuda')
        def predict(z,j):
            t=times[j]
            eps=model(z,torch.full((8,),t,device='cuda'),lab,(start-tensor(cfg['dataset']['mean']))/tensor(cfg['dataset']['std']),route_cond)
            clean=(z-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            return eps,clean
        def coordinates(z):
            return decode(z,cfg['log_increment_mean'],cfg['log_increment_std'],route,length,start)
        def weights(p,ok,target):
            w=np.maximum(p[:,target],1e-6) if np.all(ok) else np.ones(8)
            return w/w.sum()
        z=torch.randn(8,1,511,device='cuda',generator=noise)
        for j in range(76):
            eps,clean=predict(z,j)
            at=alpha[times[j]];ap=alpha[times[j+1]]
            if j<75:z=ap.sqrt()*clean+(1-ap).sqrt()*eps
        begin=time.monotonic()
        p,ok,_=sc.image(coordinates(clean)[0].cpu().numpy())
        gaf_seconds+=time.monotonic()-begin;calls+=8
        u=rng.random(8)
        draw=lambda w:np.minimum(7,np.searchsorted(np.cumsum(w),u))
        if balanced:
            def draw(w):
                # Smallest shrinkage toward uniform satisfying the ESS floor.
                variance=np.sum((w-1/8)**2)
                scale=min(1.,np.sqrt((1/6-1/8)/max(variance,1e-15)))
                tempered=1/8+scale*(w-1/8)
                grid=(np.arange(8)+u[0])/8
                return np.minimum(7,np.searchsorted(tempered.cumsum(),grid))
        parents={'identity':np.arange(8),'uniform':draw(np.ones(8)/8),'mid_gaf':draw(weights(p,ok,c)),'wrong':draw(weights(p,ok,1+int(c)%5))}
        sigma=((1-ap)/(1-at)*(1-at/ap)).sqrt()
        innovation=torch.randn(z.shape,device='cuda',generator=noise)
        slot=int(rng.integers(8));post_u=rng.random()
        diagnostic.append(dict(label=int(c),mid_range=float(np.ptp(p[:,c])),valid=bool(ok.all()),distinct_mid=int(len(set(parents['mid_gaf']))),distinct_uniform=int(len(set(parents['uniform'])))))
        for arm,idx in parents.items():
            if terminal_check and arm!='identity':continue
            zz=ap.sqrt()*clean[idx]+(1-ap-sigma.square()).clamp_min(0).sqrt()*eps[idx]+sigma*innovation
            for j in range(76,100):
                ee,cc=predict(zz,j)
                prev=alpha[times[j+1]] if j<99 else 1.
                zz=prev**.5*cc+(1-prev)**.5*ee
            xx,prog=coordinates(zz);x=xx.cpu().numpy();cap=(prog[:,-1]>=length).cpu().numpy()
            dist=np.mean([np.linalg.norm(x[a]-x[b],axis=-1).mean() for a in range(8) for b in range(a)])
            out[arm].append(x[slot]);caps[arm].append(bool(cap[slot]));div[arm].append(float(dist))
            if arm=='identity':
                begin=time.monotonic()
                post,valid,_=sc.image(x);gaf_seconds+=time.monotonic()-begin;calls+=8
                k=min(7,int(np.searchsorted(weights(post,valid,c).cumsum(),post_u)))
                if terminal_check:
                    best=int(np.argmax(post[:,c])) if valid.all() else slot
                    out['post_max'].append(x[best]);caps['post_max'].append(bool(cap[best]));div['post_max'].append(float(dist))
                out['post_gaf'].append(x[k]);caps['post_gaf'].append(bool(cap[k]));div['post_gaf'].append(float(dist))
    if terminal_check:
        saved=np.load(f'pilot-output/generated-{seed}.npz')
        for name in ['identity','post_gaf']:
            np.testing.assert_allclose(out[name],saved[name],atol=.005,rtol=0)
    labels=d['labels']; real=reference(cfg,d)
    matcher=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64)
    def motion(x):
        v=np.linalg.norm(np.diff(x,axis=1),axis=-1)/.25
        return [v,np.diff(v,axis=1)/.25,v.sum(1)*.25]
    rv=motion(real);result={}
    for arm,xx in out.items():
        x=np.array(xx);v=motion(x);aud=sc.raw(x);p,ok,_=sc.image(x)
        graph=[matcher.match(y,CLASSES[int(c)]) for y,c in zip(x,labels)]
        per={}
        for c in np.unique(labels):
            ix=labels==c
            per[CLASSES[int(c)]]=[float(np.mean(aud[ix].argmax(1)==c)),*[wasserstein_distance(a[ix].ravel(),b[ix].ravel()) for a,b in zip(v,rv)],float(np.mean(p[ix,c]))]
        result[arm]=dict(columns=['tcn_agreement','speed_w1','acc_w1','travel_w1','gaf_target_probability'],per_class=per,macro=np.mean(list(per.values()),axis=0).tolist(),pair_distance=np.mean(div[arm]),selected_graph_pass=sum(g['graph_consistent'] and not g['candidate_truncation_warning'] for g in graph),selected_caps=sum(caps[arm]),selected_encoding_failures=int(sum(~ok)))
    prefix='terminal-' if terminal_check else ('balanced-' if balanced else '')
    np.savez_compressed(f'pilot-output/{prefix}generated-{seed}.npz',labels=labels,**out)
    Path(f'pilot-output/{prefix}seed{seed}.json').write_text(json.dumps(dict(seed=seed,conditions=len(labels),complete=True,arms=result,mid=diagnostic,shared_mid_plus_terminal_scoring_seconds=gaf_seconds,shared_sampling_cnn_images=calls),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--balanced',action='store_true');parser.add_argument('--terminal-check',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);sample(args.seed,args.balanced,args.terminal_check)
