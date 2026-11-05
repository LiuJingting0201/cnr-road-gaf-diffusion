"""Original Hilbert/GAF on each complete native-cadence trip, with no time crop."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from g19_gaf import images
from g22_diffusion import ROOT


def encode(file):
    out=file.parent/"quarter-images.npz"
    if out.exists():return
    with np.load(file) as q, np.load(file.parent/"trips.npz") as d:
        xy=q["txy"][:,1:];off=q["offset"];ids=q["ids"];y=d["labels"];paths=d["paths"]
        assert np.array_equal(ids,d["ids"])
    tracks=[np.vstack([path[0],xy[a:b],path[-1]]) for path,a,b in zip(paths,off,off[1:])]
    rgb=[];valid=[]
    for i in range(0,len(tracks),16):
        im,ok=images(tracks[i:i+16]);rgb.extend(im);valid.extend(ok)
    tmp=file.parent/"quarter-images.tmp.npz"
    np.savez_compressed(tmp,full=rgb,ok=valid,ids=ids,labels=y)
    tmp.replace(out)


if __name__=="__main__":
    files=[f for split in ("train","monitor","evaluation") for f in sorted((ROOT/split).glob("*/quarter.npz"))]
    with ProcessPoolExecutor(max_workers=10) as pool:list(pool.map(encode,files))
