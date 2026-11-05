"""Generate fresh traffic episodes with independently randomized OD demand."""
import argparse, collections, hashlib, json, subprocess
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import sumolib
from cnr.data import CLASSES
from cnr.route_paths import RoutePaths
VC=['','bus','motorcycle','passenger','bicycle','taxi']
def simulate(seed,binary):
    out=Path('g10-data')/str(seed);out.mkdir(exist_ok=False)
    network=Path('data/maps/most.net.xml').resolve()
    net=sumolib.net.readNet(str(network));rng=np.random.default_rng(seed)
    types=E.Element('routes')
    for c in range(1,6):E.SubElement(types,'vType',id=f'type{c}',vClass=VC[c])
    jobs=[];stats={}
    for c in range(1,6):
        edges=[e for e in net.getEdges() if e.allows(VC[c]) and e.getLength()>20]
        attempts=0
        for j in range(140 if c==3 else 20):
            for trial in range(3000):
                a,b=rng.choice(len(edges),2,replace=False);attempts+=1
                path,L=net.getShortestPath(edges[a],edges[b],vClass=VC[c])
                if path and 2000<=L<=10000:break
            else:raise ValueError('No permitted OD pair')
            ident=f'{seed}_{c}_{j:03d}'
            jobs.append((float(rng.uniform(0,240)),c,ident,[e.getID() for e in path]))
        stats[c]=attempts
    for dep,c,ident,path in sorted(jobs):
        v=E.SubElement(types,'vehicle',id=ident,type=f'type{c}',depart=f'{dep:.2f}',departLane='best')
        E.SubElement(v,'route',edges=' '.join(path))
    routes=out/'demand.xml';E.ElementTree(types).write(routes)
    fcd=out/'fcd.xml'
    cmd=[binary,'-n',str(network),'-r',str(routes),'--seed',str(seed),'--step-length','.25','--end','650','--time-to-teleport','-1','--fcd-output',str(fcd),'--no-step-log','true','--duration-log.disable','true']
    with (out/'sumo.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    tracks=collections.defaultdict(list)
    for _,step in E.iterparse(fcd,events=['end']):
        if step.tag!='timestep':continue
        t=float(step.get('time'))
        for v in step.findall('vehicle'):
            ident=v.get('id')
            if int(ident.rsplit('_',1)[1])<20:
                tracks[ident].append((t,float(v.get('x')),float(v.get('y'))))
        step.clear()
    matcher=RoutePaths(str(network),radius=2.)
    kept=[];reject=collections.Counter()
    for _,c,ident,_ in sorted(jobs,key=lambda j:j[2]):
        if int(ident.rsplit('_',1)[1])>=20:continue
        a=np.array(tracks.get(ident,[]))
        if len(a)<512:reject['short_or_absent']+=1;continue
        left=(len(a)-512)//2;s=a[left:left+512]
        if not np.allclose(np.diff(s[:,0]),.25):reject['gap']+=1;continue
        xy=s[:,1:]
        if np.linalg.norm(xy[-1]-xy[0])<=5:reject['stationary']+=1;continue
        trace=matcher.trace(xy,CLASSES[c])
        rec=matcher.reconstruct(trace,CLASSES[c]) if trace is not None else None
        if rec is None or rec['progress'][-1]>=4084:reject['reconstruction']+=1;continue
        kept.append((ident,c,xy))
    np.savez_compressed(out/'eligible.npz',ids=[k[0] for k in kept],labels=[k[1] for k in kept],xy=[k[2] for k in kept])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(seed=seed,version=subprocess.check_output([binary,'--version'],text=True).splitlines()[0],net=sha(network),demand=sha(routes),fcd=sha(fcd),cmd=cmd,eligible=len(kept),rejections=reject,attempts=stats,inserted_test=len(tracks)),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('binary');a=p.parse_args();simulate(a.seed,a.binary)
