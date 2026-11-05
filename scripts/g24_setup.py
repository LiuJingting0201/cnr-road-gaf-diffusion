"""New OD episodes; choose eligible requests by order, before viewing generated scores."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
from g22_sim import episode,ROOT
from g22_hires import run as replay

if __name__=="__main__":
    tasks=[("cross",seed,16) for seed in range(276001,276009)]
    with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(episode,tasks))
    files=sorted((ROOT/"cross").glob("*/trips.npz"))
    with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(replay,[f.parent for f in files]))
    rows=[]
    refs=[]
    for f in files:
        with np.load(f) as d,np.load(f.parent/"quarter.npz") as r:
            assert np.array_equal(d["ids"],r["ids"])
            xy=r["txy"];off=r["offset"]
            for c in range(8):
                indices=sorted(np.flatnonzero(d["labels"]==c),key=lambda i:d["ids"][i])[:3]
                for i in indices:
                    rows.append((d["ids"][i],c,d["length"][i],d["paths"][i]))
                    refs.append((d["duration"][i],xy[off[i]:off[i+1]]))
    ids,label,length,path=map(np.array,zip(*rows));duration,xy=zip(*refs)
    out=Path("g24-data");out.mkdir(exist_ok=True)
    np.savez(out/"requests.npz",ids=ids,labels=label,length=length,paths=path)
    np.savez_compressed(out/"references.npz",ids=ids,duration=duration,txy=np.concatenate(xy),offset=np.r_[0,np.cumsum([len(x) for x in xy])])
    print(len(ids),flush=True)
