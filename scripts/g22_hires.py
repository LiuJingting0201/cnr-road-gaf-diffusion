"""Replay full itineraries at 0.25 s; preserve every recorded position for CNN evaluation."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from g22_sim import ROOT,NET


def run(folder):
    out=folder/"quarter.npz"
    if out.exists():return
    with np.load(folder/"trips.npz") as z:
        ids=z["ids"];duration=z["duration"]
    wanted=set(ids);tracks=defaultdict(list)
    with TemporaryDirectory(dir=folder,prefix="quarter-") as scratch:
        fcd=Path(scratch)/"fcd.xml"
        cmd=["g10-env/sumo/bin/sumo","-n",str(NET),"-r",str(folder/"demand.xml"),"--seed",folder.name,
             "--step-length",".25","--end","2400","--time-to-teleport","-1","--fcd-output",str(fcd),
             "--device.fcd.period",".25","--fcd-output.attributes","x,y","--no-step-log","true"]
        with (folder/"quarter.log").open("w") as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        for _,step in ET.iterparse(fcd,events=["end"]):
            if step.tag!="timestep":continue
            time=float(step.get("time"))
            for item in step:
                ident=item.get("id")
                if ident in wanted:tracks[ident].append((time,float(item.get("x")),float(item.get("y"))))
            step.clear()
    arrays=[]
    for ident,T in zip(ids,duration):
        x=np.array(tracks[ident],dtype=np.float32)
        if len(x)<2 or not np.allclose(np.diff(x[:,0]),.25) or abs(len(x)*.25-T)>1.:
            raise ValueError(f"Replay not a complete trip: {ident}, {len(x)}, {T}")
        arrays.append(x)
    temp=folder/"quarter.tmp.npz"
    np.savez_compressed(temp,ids=ids,offset=np.r_[0,np.cumsum([len(x) for x in arrays])],txy=np.concatenate(arrays))
    temp.replace(out)


if __name__=="__main__":
    files=[p.parent for s in ("train","monitor","evaluation") for p in sorted((ROOT/s).glob("*/trips.npz"))]
    with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(run,files))
    (ROOT/"quarter-ready.txt").write_text(f"{len(files)} episodes\n")
