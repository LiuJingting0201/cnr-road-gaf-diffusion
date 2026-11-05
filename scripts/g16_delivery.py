"""Export both fixed eight-class seeds; no observed reference positions included."""
import csv
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
from g16_data import BASE,NAMES


def export():
    req = np.load(BASE/"requests.npz")
    index = {ident:i for i,ident in enumerate(req["ids"])}
    path = BASE/"eight-class-synthetic.zip"
    meta = []
    with ZipFile(path,"w",compression=ZIP_DEFLATED) as z:
        with z.open("trajectories.csv","w") as binary, io.TextIOWrapper(binary,newline="") as text:
            w = csv.writer(text)
            w.writerow(["time","id","x","y","type"])
            for seed in [42,43]:
                d = np.load(BASE/f"C-{seed}"/"g15.npz")
                for j,ident in enumerate(d["ids"]):
                    i = index[ident]
                    for k,xy in enumerate(d["xy"][j]):
                        sid = f"G16_C_{seed}_{j}_{k}"
                        name = NAMES[req["labels"][i]]
                        w.writerows((f"{t*.25:.2f}",sid,f"{x:.3f}",f"{y:.3f}",name) for t,(x,y) in enumerate(xy))
                        meta.append(dict(id=sid,request=str(ident),model_seed=seed,noise_slot=k,
                                         mode=name,capped=bool(d["caps"][j,k])))
        z.writestr("conditions.json",json.dumps(meta))
        z.writestr("README.txt","Diffusion-generated candidates on G15 validation starts; do not use as training data for evaluation on this split. All four noise draws and both model seeds are included.\n")
    with ZipFile(path) as z:
        assert z.testzip() is None
        with z.open("trajectories.csv") as f:
            assert sum(1 for _ in f)==1+len(meta)*512
    return len(meta)


if __name__=="__main__":
    print(export())
