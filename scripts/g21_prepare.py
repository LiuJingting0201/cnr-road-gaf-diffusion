"""Build the unchanged Hilbert images and motion sequences on matching IDs."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from g16_data import csv_tracks
from g19_gaf import images, original_metadata

ROOT=Path(os.getenv("CNR_CLASSIFIER_DATA","g21-data"))


def pack_job(task):
    xy=task
    meta=original_metadata()
    if ROOT.name=="g21-fine":
        meta=dict(meta,hilbert_eps=meta["hilbert_eps"]/16)
    rgb,ok=images(xy,meta)
    step=np.linalg.norm(np.diff(xy.astype(float),axis=1),axis=-1)
    return rgb,ok,(4*step).astype(np.float32)


def pack(split,n):
    folder=ROOT/split
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():
        return
    rgb=np.lib.format.open_memmap(folder/"images.npy",mode="w+",dtype=np.uint8,shape=(n,224,224,3))
    v=np.lib.format.open_memmap(folder/"speed.npy",mode="w+",dtype=np.float32,shape=(n,511))
    ids,labels,valid=[],[],[]
    jobs=[]
    def save(pending):
        start,count,future=pending
        pixels,ok,speed=future.result()
        rgb[start:start+count]=pixels
        v[start:start+count]=speed
        valid.extend(ok)
    with ProcessPoolExecutor(max_workers=16) as pool:
        batch=[]
        for ident,c,xy in csv_tracks(split):
            ids.append(ident)
            labels.append(c)
            batch.append(xy)
            if len(batch)==16:
                jobs.append((len(ids)-len(batch),len(batch),pool.submit(pack_job,np.stack(batch))))
                batch=[]
                if len(jobs)>=32:
                    save(jobs.pop(0))
        if batch:
            jobs.append((len(ids)-len(batch),len(batch),pool.submit(pack_job,np.stack(batch))))
        for job in jobs:
            save(job)
    assert len(ids)==n==len(set(ids))==len(valid)
    np.savez(folder/"index.npz",ids=ids,labels=labels,valid=valid)
    rgb.flush()
    v.flush()
    (folder/"complete.json").write_text(json.dumps(dict(n=n,encoding_failures=int(n-np.sum(valid)))))


if __name__=="__main__":
    pack("train",64000)
    pack("validation",8000)
