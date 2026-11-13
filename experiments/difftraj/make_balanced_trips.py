"""Generate complete SUMO vehicle journeys, then export DiffTraj arrays by episode."""
import gzip,json,hashlib,os,subprocess,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict,Counter
from array import array
import xml.etree.ElementTree as ET
import numpy as np
import sumolib
base=Path('/workspace/cnr-experiments/gaf-inloop-v1')
src=base/'g30-job/g30-data/source/scenario/in'
binary=base/'g10-env/sumo/bin/sumo'
modes=['bus','motorcycle','passenger','bicycle','taxi']
def episode(seed):
    out=Path(str(seed));out.mkdir(exist_ok=True)
    pack=out/'trips.npz'
    if pack.exists():return
    start=time.monotonic()
    net=sumolib.net.readNet(str(src/'most.net.xml'))
    rng=np.random.default_rng(seed); r=ET.Element('routes')
    for m in modes: ET.SubElement(r,'vType',id=m,vClass=m)
    routes=ET.parse(src/'route/most.buses.flows.xml').getroot().findall('route')
    for route in routes:
        for stop in route.findall('stop'):stop.attrib.pop('until',None)
        r.append(route)
    jobs=[]
    for m in modes:
        edges=[e for e in net.getEdges() if e.allows(m) and e.getLength()>25]
        for k in range(100):
            ident=f'{seed}_{m}_{k}'; depart=int(rng.integers(900))
            if m=='bus':
                veh=ET.Element('vehicle',id=ident,type=m,depart=str(depart),route=routes[rng.integers(len(routes))].get('id'))
            else:
                while True:
                    a,b=(edges[j] for j in rng.choice(len(edges),2,replace=False))
                    if 1100<np.linalg.norm(np.subtract(a.getShape()[0],b.getShape()[-1]))<6500:break
                veh=ET.Element('trip',{'id':ident,'type':m,'depart':str(depart),'from':a.getID(),'to':b.getID(),'departLane':'best'})
            jobs.append((depart,ident,veh))
    for _,_,v in sorted(jobs):r.append(v)
    ET.ElementTree(r).write(out/'routes.xml')
    with (out/'log').open('w') as log:
        subprocess.run([str(binary),'-n',str(src/'most.net.xml'),'-r',str(out/'routes.xml'),'-a',str(src/'add/most.busstops.add.xml'),'--seed',str(seed),'--step-length','1','--end','7200','--time-to-teleport','-1','--ignore-route-errors','true','--fcd-output',str(out/'fcd.xml.gz'),'--fcd-output.attributes','x,y,speed','--tripinfo-output',str(out/'arrivals.xml'),'--no-step-log','true'],stdout=log,stderr=subprocess.STDOUT,check=True)
    arrivals={t.get('id'):(float(t.get('depart')),float(t.get('arrival'))) for t in ET.parse(out/'arrivals.xml').getroot().findall('tripinfo') if t.get('vaporized','') in ('','false','0')}
    obs=defaultdict(lambda:array('f'))
    with gzip.open(out/'fcd.xml.gz') as f:
        for _,step in ET.iterparse(f,events=['end']):
            if step.tag!='timestep':continue
            t=float(step.get('time'))
            for v in step:
                if v.get('id') in arrivals:obs[v.get('id')].extend([t,float(v.get('x')),float(v.get('y')),float(v.get('speed'))])
            step.clear()
    records=[];raw=[];headers=[];coords=[];ids=[];classes=[]
    for ident,buf in sorted(obs.items()):
        v=np.frombuffer(buf,dtype=np.float32).reshape(-1,4); t=v[:,0]; xy=v[:,1:3]; n=len(t);dep,arr=arrivals[ident]
        if n<120 or t[0]!=dep or arr-t[-1]!=1 or not np.all(np.diff(t)==1) or not np.isfinite(v).all():continue
        steps=np.linalg.norm(np.diff(xy,axis=0),axis=1);L=float(steps.sum());dur=float(t[-1]-t[0])
        if L<100 or steps.max()>65:continue
        idx=np.linspace(0,n-1,200);coords.append(np.column_stack([np.interp(idx,np.arange(n),col) for col in xy.T]))
        ids.append(ident);classes.append(ident.split('_')[1]);headers.append([int((seed%1440*60+dep)//300)%288,L,dur,n,L/(n-1),L/dur]);raw.append(v)
    np.savez_compressed(out/'pending.npz',ids=ids,classes=classes,h=headers,xy=np.asarray(coords,dtype='f'),raw=np.concatenate(raw))
    (out/'pending.npz').replace(pack); (out/'fcd.xml.gz').unlink()
    print(seed,dict(Counter(classes)),round(time.monotonic()-start,1),flush=True)
def main():
    state={}
    for split,target,start in [('train',10000,410000),('validation',1000,420000),('test',1000,430000)]:
        counts=Counter();used=[]; seed=start
        with ProcessPoolExecutor(8) as pool:
            while min((counts[m] for m in modes),default=0)<target:
                batch=list(range(seed,seed+8));list(pool.map(episode,batch))
                for s in batch:
                    with np.load(f'{s}/trips.npz') as d:counts.update(d['classes'].tolist())
                used+=batch;seed+=8
                if seed-start>800:raise RuntimeError(counts)
        state[split]=(used,target)
    seen=set()
    for split,(seeds,target) in state.items():
        xs=[];hs=[];labels=[];ids=[];count=Counter()
        for s in seeds:
            with np.load(f'{s}/trips.npz') as d:
                x,h,ls,keys=[d[k] for k in ('xy','h','classes','ids')]
                for i,m in enumerate(ls):
                    if count[m]>=target:continue
                    digest=hashlib.sha256(x[i].tobytes()).digest()
                    if digest in seen:continue
                    seen.add(digest);count[m]+=1;xs.append(x[i]);hs.append(h[i]);labels.append(m);ids.append(keys[i])
        assert all(count[m]==target for m in modes),count
        x=np.asarray(xs);h=np.asarray(hs)
        if split=='train':
            lo=x.min((0,1));hi=x.max((0,1));mu=x.mean((0,1),dtype='d');sd=x.std((0,1),dtype='d')
            hm=h[:,1:].mean(0);hd=h[:,1:].std(0).clip(1e-7)
            np.savez('normalization.npz',lo=lo,hi=hi,mean=mu,std=sd,header_mean=hm,header_std=hd)
        cells=np.clip(np.floor((x[:,[0,-1]]-lo)/(hi-lo)*16),0,15).astype(int)
        h[:,1:]=(h[:,1:]-hm)/hd;h=np.column_stack([h,cells[:,:,0]*16+cells[:,:,1]])
        p=Path(split);p.mkdir(exist_ok=True)
        np.save(p/'traj.npy',((x-mu)/sd).astype('f'));np.save(p/'head.npy',h.astype('f'));np.savez_compressed(p/'index.npz',ids=ids,classes=labels)
        (p/'counts.json').write_text(json.dumps(count))
    Path('COMPLETE').touch()
if __name__=='__main__':main()
