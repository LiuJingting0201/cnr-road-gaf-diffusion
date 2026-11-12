"""Uniform fixed-size reservoirs, in disjoint taxi splits, from official Porto train.csv."""
import csv, hashlib, io, json, random, shutil, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
def run(source, out):
    split={}
    pools=[[],[]]; counts=[0,0]; caps=[46470,3000]
    rng=[random.Random(20261112),random.Random(173)]
    with zipfile.ZipFile(source) as outer,tempfile.TemporaryFile() as tmp:
        with outer.open('train.csv.zip') as f: shutil.copyfileobj(f,tmp)
        with zipfile.ZipFile(tmp) as inner,inner.open('train.csv') as stream:
            for r in csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig')):
                if r['MISSING_DATA'].lower()=='true' or r['POLYLINE'].count('[')<3: continue
                t=int(r['TAXI_ID'])
                if t not in split: split[t]=int(hashlib.sha256(str(t).encode()).hexdigest()[:8],16)%10
                if split[t]==8: continue
                k=int(split[t]==9);counts[k]+=1
                j=counts[k]-1 if counts[k]<=caps[k] else rng[k].randrange(counts[k])
                if j>=caps[k]: continue
                track=np.array(json.loads(r['POLYLINE']))
                assert np.isfinite(track).all() and len(track)>=2
                values=np.rint(track*1e6).astype('i4')
                row=(np.diff(values,axis=0,prepend=np.zeros((1,2),'i4')),t,int(r['TIMESTAMP']))
                if len(pools[k])<caps[k]: pools[k].append(row)
                else: pools[k][j]=row
    for k,name in enumerate(['mixed-train','mixed-test']):
        tracks,taxis,starts=zip(*pools[k]);sizes=np.array(list(map(len,tracks)))
        np.savez_compressed(Path(out)/(name+'.npz'),delta=np.concatenate(tracks),length=sizes,taxi=taxis,start=starts)
        print(name,counts[k],len(sizes),float(np.median(sizes)),flush=True)
if __name__=='__main__': run(*sys.argv[1:])
