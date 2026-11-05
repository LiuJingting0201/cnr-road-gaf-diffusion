import sys,json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from g8_common import assets,reference
from cnr.cluster_wasserstein import cluster_w1
from cnr.topology import LaneConsistency
from cnr.data import CLASSES
stage=int(sys.argv[1]);cfg,_,d=assets();real=reference(cfg,d)[:,None];y=d['labels'];root=Path('g9-output')
ds=[np.load(root/f'{s}-{stage}.npz') for s in (42,43,44)]
burden=ds[0]['burden'];high=np.zeros(len(y),bool)
for c in np.unique(y):
 ix=np.where(y==c)[0];high[ix[np.argsort(burden[ix],kind='stable')[len(ix)//2:]]]=True
def motion(x):
 v=np.linalg.norm(np.diff(x,axis=-2),axis=-1)/.25
 return [v,np.diff(v,axis=-1)/.25,v.sum(-1)*.25,(v<.1).mean(-1)]
m=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64);rv=motion(real);out={}; draws={}
for arm in ['mid','terminal','maximum','identity']:
 vs=[motion(z[arm]) for z in ds]; valid=0;cache={}
 if '--reuse-road' in sys.argv:
  valid=json.loads((root/f'report{stage}.json').read_text())[arm]['valid']
 for z in ([] if '--reuse-road' in sys.argv else ds):
  for row,c in zip(z[arm],y):
   for path in row:
    k=(int(c),path.tobytes())
    if k not in cache:
     g=m.match(path,CLASSES[int(c)]);cache[k]=g['graph_consistent'] and not g['candidate_truncation_warning']
    valid+=int(cache[k])
 out[arm]={'valid':valid,'caps':int(sum(z['cap_'+arm].sum() for z in ds)),'diversity':float(np.mean([np.linalg.norm(z[arm][:,a]-z[arm][:,b],axis=-1).mean() for z in ds for a,b in [(0,1),(0,2),(1,2)]]))}
 for group,mask in [('all',np.ones(len(y),bool)),('low',~high),('high',high)]:
  res=np.zeros((1001,4));rng=np.random.default_rng({'all':95001,'low':95002,'high':95003}[group]);per={}
  for c in np.unique(y):
   ix=(y==c)&mask;n=sum(ix);w=np.vstack([np.ones(n),rng.multinomial(n,np.ones(n)/n,size=1000)]);val=np.zeros_like(res)
   for v in vs:
    for j in range(4):
     a=v[j][ix].reshape(n,-1);b=rv[j][ix].reshape(n,-1)
     if j<3:val[:,j]+=cluster_w1(a,b,w)/3
     else:val[:,j]+=abs(w@(a.mean(1)-b.mean(1))/n)/3
   per[CLASSES[int(c)]]=val[0].tolist();res+=val/len(np.unique(y))
  out[arm][group]=res[0].tolist();out[arm][group+'_class']=per;draws[arm,group]=res
for arm in ['terminal','maximum','identity']:
 for g in ['all','low','high','interaction']:
  def delta(g):return draws['mid',g]-draws[arm,g]
  a=delta(g) if g!='interaction' else delta('high')-delta('low')
  out['mid-'+arm+'_'+g]={'point':a[0].tolist(),'CI':np.percentile(a[1:],[2.5,97.5],axis=0).T.tolist()}
out['burden']=burden.tolist();out['high']=high.tolist();out['encoding_failures']=[z['encoding_failures'].tolist() for z in ds]
(root/f'report{stage}.json').write_text(json.dumps(out,indent=1))
