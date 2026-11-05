"""Create independent OD requests without exposing validation futures to training."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import numpy as np
from g22_sim import ROOT,episode
from g22_hires import run as replay


def main():
    flat=os.getenv("CNR_G26_LAYOUT")=="flat"
    first=278001 if flat else 277001
    tasks=[("fresh",s,48) for s in range(first,first+8)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(episode,tasks))
    files=sorted((ROOT/"fresh").glob("*/trips.npz"))
    assert len(files)==8
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(replay,[f.parent for f in files]))
    rows=[];refs=[]
    for f in files:
        with np.load(f) as q,np.load(f.parent/"quarter.npz") as r:
            assert np.array_equal(q["ids"],r["ids"])
            pos,off=r["txy"],r["offset"]
            for c in range(8):
                for i in sorted(np.flatnonzero(q["labels"]==c),key=lambda i:q["ids"][i])[:6]:
                    rows.append((q["ids"][i],c,q["length"][i],q["paths"][i]))
                    refs.append((q["duration"][i],pos[off[i]:off[i+1]]))
    ids,labels,length,paths=map(np.asarray,zip(*rows))
    T,xy=zip(*refs);out=Path("g27-data" if flat else "g26-data")
    np.savez(out/"requests.npz",ids=ids,labels=labels,length=length,paths=paths)
    np.savez_compressed(out/"references.npz",ids=ids,duration=T,txy=np.concatenate(xy),offset=np.r_[0,np.cumsum([len(x) for x in xy])])
    (out/"ready.txt").write_text(str(len(ids)))


if __name__=="__main__":main()
